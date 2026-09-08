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
