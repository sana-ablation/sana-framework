"""Decode variant directory names into structured axes.

`sana_evaluation.cli._variant_condition_label` is the authoritative encoder.
This module is the reader, and it deliberately owns its vocabulary rather than
importing the runtime: `sana_evaluation.runner.modes` pulls `strands`, `boto3`,
`lancedb` and `torch` (measured 1.26s) because it carries tool wiring alongside
axis resolution. `test/test_variant_vocabulary_contract.py` keeps the two
vocabularies honest without the runtime dependency.

Four naming generations reach this parser:

    gen 1  search_i_results_i_plani_computei_k5_skills_off
    gen 2  search_ideal__results_ideal__profile_ideal__compute_ideal__k5__skills_off
    gen 3  search_ideal__results_ideal__plan_ideal__compute_ideal__k5__skills_off
    gen 4  search_ideal__plan_ideal__compute_ideal__results_rich__k5__skills_off

Parsing is by prefix, never by position, so a future axis reorder is a
non-event. Gen 2 is the only generation that spells the planning axis
``profile``; it is absorbed here and no consumer repeats it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, FrozenSet, Iterable, List, Optional, Tuple

SEARCH_MODES: FrozenSet[str] = frozenset({"naive", "standard", "ideal", "preloaded", "web"})
PLAN_MODES: FrozenSet[str] = frozenset({"naive", "standard", "ideal"})
COMPUTE_MODES: FrozenSet[str] = frozenset({"standard", "ideal"})
RESULT_MODES: FrozenSet[str] = frozenset({"minimal", "rich"})

# Read-only. `naive`/`ideal` are the former spellings of the results axis; they
# exist on disk and are never written back out.
RESULT_MODE_ALIASES: Dict[str, str] = {"naive": "minimal", "ideal": "rich"}

# Mirrors sana_evaluation.config.AXIS_DEFAULTS. An absent segment means the
# encoder omitted a default, so the reader must supply the same one.
AXIS_DEFAULTS: Dict[str, str] = {
    "search": "standard",
    "plan": "standard",
    "compute": "standard",
    "results": "rich",
}

# Gen-1 wrote one letter per axis value.
_LETTER_TO_MODE: Dict[str, str] = {
    "n": "naive",
    "d": "standard",
    "i": "ideal",
    "p": "preloaded",
}

_KNOWN_FLAGS: FrozenSet[str] = frozenset({"free", "lessguide", "nos3"})

# Segment names that make a string a variant name at all.
_AXIS_KEYS: FrozenSet[str] = frozenset({"search", "results", "plan", "profile", "compute"})

_MATCHABLE_AXES: FrozenSet[str] = frozenset({"search", "results", "plan", "compute"})


def _resolve_mode(token: str) -> str:
    """Map a gen-1 letter to its word; leave anything else alone."""
    if len(token) == 1:
        return _LETTER_TO_MODE.get(token, token)
    return token


def _resolve_results(token: str) -> str:
    mode = _resolve_mode(token)
    return RESULT_MODE_ALIASES.get(mode, mode)


@dataclass(frozen=True)
class Variant:
    """One decoded variant directory name.

    `search`, `results`, `plan` and `compute` are always populated -- from the
    name, or from `AXIS_DEFAULTS` when the encoder omitted the segment. Values
    outside the known vocabulary are preserved rather than rejected, so a new
    arm is absent from a figure instead of raising inside one.
    """

    search: str
    results: str
    plan: str
    compute: str
    k: Optional[int]
    search_calls: Optional[int]
    flags: FrozenSet[str]
    skills: bool
    raw: str

    def matches(self, **axes: str) -> bool:
        """True when every named axis equals its argument.

        Unnamed axes are ignored, so `k`, `skills` and `flags` are never part of
        condition identity. An unknown axis name raises rather than quietly
        returning False -- a typo there would reintroduce exactly the silent
        mismatch this module exists to retire.
        """
        for name, expected in axes.items():
            if name not in _MATCHABLE_AXES:
                raise TypeError(
                    f"unknown axis {name!r}; expected one of "
                    f"{', '.join(sorted(_MATCHABLE_AXES))}"
                )
            if getattr(self, name) != expected:
                return False
        return True


def parse_variant(name: str) -> Variant:
    """Decode a variant directory name. Raises ValueError if it is not one."""
    text = str(name).strip()
    # Gen 1 separates with `_`, gens 2-4 with `__`. Collapsing first lets one
    # prefix scan read all four.
    tokens = [token for token in text.replace("__", "_").split("_") if token]

    axes: Dict[str, Optional[str]] = {
        "search": None,
        "results": None,
        "plan": None,
        "compute": None,
    }
    k: Optional[int] = None
    search_calls: Optional[int] = None
    flags: set = set()
    skills = False
    saw_axis = False

    index = 0
    while index < len(tokens):
        token = tokens[index]
        following = tokens[index + 1] if index + 1 < len(tokens) else None

        if token == "search" and following is not None:
            axes["search"] = _resolve_mode(following)
            saw_axis = True
        elif token == "results" and following is not None:
            axes["results"] = _resolve_results(following)
            saw_axis = True
        elif token in {"plan", "profile"} and following is not None:
            # Gen 2 spells this axis `profile`; it means the same thing.
            axes["plan"] = _resolve_mode(following)
            saw_axis = True
        elif token == "compute" and following is not None:
            axes["compute"] = _resolve_mode(following)
            saw_axis = True
        elif token.startswith("plan") and len(token) > 4:
            # Gen 1 glued the value on: `plani`.
            axes["plan"] = _resolve_mode(token[4:])
            saw_axis = True
        elif token.startswith("compute") and len(token) > 7:
            axes["compute"] = _resolve_mode(token[7:])
            saw_axis = True
        elif token == "skills" and following is not None:
            skills = following == "on"
        elif token == "debug":
            # `_with_debug_suffix` appends `__debug_<mode>`; not an axis.
            index += 2
            continue
        elif token.startswith("k") and token[1:].isdigit():
            k = int(token[1:])
        elif token.startswith("sc") and token[2:].isdigit():
            search_calls = int(token[2:])
        elif token in _KNOWN_FLAGS:
            flags.add(token)

        index += 1

    if not saw_axis:
        raise ValueError(
            f"not a variant name: {name!r} (no "
            f"{'/'.join(sorted(_AXIS_KEYS))} segment)"
        )

    return Variant(
        search=axes["search"] or AXIS_DEFAULTS["search"],
        results=axes["results"] or AXIS_DEFAULTS["results"],
        plan=axes["plan"] or AXIS_DEFAULTS["plan"],
        compute=axes["compute"] or AXIS_DEFAULTS["compute"],
        k=k,
        search_calls=search_calls,
        flags=frozenset(flags),
        skills=skills,
        raw=text,
    )


def try_parse_variant(name: str) -> Optional[Variant]:
    """`parse_variant`, but None instead of ValueError.

    Call sites that scan a directory tree meet non-variant entries routinely and
    should skip them, not crash.
    """
    try:
        return parse_variant(name)
    except ValueError:
        return None


# `short_name` drops these. `results`, `k` and `skills` are the three segments
# `Variant.matches` ignores, so they are not part of condition identity either.
_SHORTEN_DROPPED_PAIRS: FrozenSet[str] = frozenset({"results", "skills"})
_SHORTEN_DROPPED_K = re.compile(r"k\d+")


def short_name(name: str) -> str:
    """`name` with the segments that are not condition identity removed.

    `results`, `k` and `skills` go; everything else stays, including flags such
    as `nos3`, which are what distinguish the web arm from its no-S3 twin. A
    string this module cannot decode is returned unchanged.

    The decoder decides what to drop, but it filters the RAW tokens rather than
    re-rendering them, so a gen-1 name shortens exactly as it always has -- which
    is what `answer_failure.audit_runner` needs, because the result is a journal
    filename and journals written by earlier runs still have to resolve.
    Re-deriving the grammar instead of filtering produced trailing underscores on
    every gen-4 name.
    """
    if try_parse_variant(name) is None:
        return name

    tokens = [token for token in str(name).replace("__", "_").split("_") if token]
    kept: List[str] = []
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token in _SHORTEN_DROPPED_PAIRS:
            # `results rich`, `skills off`: the value travels with the key.
            index += 2
            continue
        if _SHORTEN_DROPPED_K.fullmatch(token):
            index += 1
            continue
        kept.append(token)
        index += 1
    return "_".join(kept)


def axis_codes(name: str) -> Dict[str, Optional[str]]:
    """Axis values keyed by short axis name, for label rendering.

    Values are resolved words (`ideal`, `standard`, `rich`), not gen-1
    letters -- there is nothing further to resolve them against.

    `run_mode_analysis._parse_variant_mode_codes` and
    `paper.delta_figures._parse_variant_codes` were byte-identical copies of
    this function; both now alias it.
    """
    decoded = try_parse_variant(str(name))
    if decoded is None:
        return {
            "search": None, "results": None, "plan": None,
            "compute": None, "skills": None, "k": None, "sc": None,
        }
    return {
        "search": decoded.search,
        "results": decoded.results,
        "plan": decoded.plan,
        "compute": decoded.compute,
        "skills": "on" if decoded.skills else "off",
        "k": str(decoded.k) if decoded.k is not None else None,
        "sc": str(decoded.search_calls) if decoded.search_calls is not None else None,
    }


# The seven conditions every figure and metric reports, in the order the paper
# reports them. `run_mode_analysis.TURN_WASTE_CONDITION_FIGURE_ORDER` and
# `answer_failure.combine_grouped_models.CONDITION_FIGURE_ORDER` held these same
# rows as variant literals; this is the single definition they now share.
#
# Only the three ablation axes take part in condition identity. `results`, `k`,
# `skills` and the flags are deliberately unconstrained, so a predicate cannot
# over-match on a modifier -- see `Variant.matches`.
CONDITION_ORDER: List[Tuple[str, Dict[str, str]]] = [
    ("No Plan", dict(search="ideal", plan="naive", compute="ideal")),
    ("Standard Plan", dict(search="ideal", plan="standard", compute="ideal")),
    ("BM25", dict(search="naive", plan="ideal", compute="ideal")),
    ("Pneuma Hybrid", dict(search="standard", plan="ideal", compute="ideal")),
    ("Standard Computation", dict(search="ideal", plan="ideal", compute="standard")),
    ("Ideal", dict(search="ideal", plan="ideal", compute="ideal")),
    ("Preloaded", dict(search="preloaded", plan="ideal", compute="ideal")),
]

_CONDITION_AXES: Dict[str, Dict[str, str]] = {label: axes for label, axes in CONDITION_ORDER}

# The variant directory names present in `experiments/` as of 2026-09-08.
# `experiments/` is gitignored, so the round-trip test cannot read the tree; this
# is the committed corpus it reads instead. Refresh with:
#   find experiments -type d -name 'search_*' | sed 's|.*/||' | sort -u
DISK_VARIANT_NAMES: Tuple[str, ...] = (
    "search_ideal__plan_ideal__compute_ideal__results_rich__k5__skills_off",
    "search_ideal__plan_ideal__compute_standard__results_rich__k5__skills_off",
    "search_ideal__plan_naive__compute_ideal__results_rich__k5__skills_off",
    "search_ideal__plan_standard__compute_ideal__results_rich__k5__skills_off",
    "search_ideal__plan_standard__compute_standard__results_minimal__skills_off",
    "search_naive__plan_ideal__compute_ideal__results_rich__k5__skills_off",
    "search_naive__plan_naive__compute_standard__results_rich__k5__skills_off",
    "search_naive__plan_standard__compute_standard__results_minimal__skills_off",
    "search_preloaded__plan_ideal__compute_ideal__results_rich__k5__skills_off",
    "search_standard__plan_ideal__compute_ideal__results_rich__k5__skills_off",
    "search_standard__plan_standard__compute_standard__results_minimal__skills_off",
    "search_standard__plan_standard__compute_standard__results_rich__k5__skills_off",
    "search_web__plan_standard__compute_standard__results_minimal__nos3__skills_off",
)


def find_variant(names: Iterable[str], **axes: str) -> Optional[str]:
    """The one observed name in `names` matching `axes`, or None.

    Returns the name as observed, so callers can use it both as a key into
    on-disk lookups and as a value written to output. Raises ValueError when
    more than one name matches: picking one silently is the failure this module
    exists to retire.
    """
    matched = []
    for name in names:
        variant = try_parse_variant(name)
        if variant is not None and variant.matches(**axes):
            matched.append(name)
    if not matched:
        return None
    if len(matched) > 1:
        raise ValueError(
            f"axes {axes!r} match {len(matched)} variants: {sorted(matched)}"
        )
    return matched[0]


def select_conditions(
    names: Iterable[str], labels: Optional[Iterable[str]] = None
) -> List[Tuple[str, str]]:
    """Pair each condition present in `names` with its observed variant name.

    Ordered by CONDITION_ORDER regardless of the order of `names` or `labels`.
    Conditions with no matching name are omitted rather than yielded blank.
    """
    pool = list(names)
    wanted = set(labels) if labels is not None else None
    if wanted is not None:
        unknown = wanted - set(_CONDITION_AXES)
        if unknown:
            raise KeyError(f"unknown condition label(s): {sorted(unknown)}")
    selected: List[Tuple[str, str]] = []
    for label, axes in CONDITION_ORDER:
        if wanted is not None and label not in wanted:
            continue
        found = find_variant(pool, **axes)
        if found is not None:
            selected.append((label, found))
    return selected
