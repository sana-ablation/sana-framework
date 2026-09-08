"""Byte-level snapshot of every reachable composed prompt.

Regenerate deliberately with:  SANA_UPDATE_GOLDEN=1 pytest test/test_prompt_corpus.py
A diff here means the composed prompt changed. That is only expected in the
commit that decomposes prompts into fragments.
"""
import itertools
import os
from pathlib import Path

import pytest

from sana_evaluation.helper import prompting as P

GOLDEN = Path(__file__).parent / "golden_prompts"

PROFILES = ("naive", "standard", "ideal")
SEARCHES = ("naive", "preloaded", "standard", "ideal", "web")
BENCHMARKS = ("lakeqa", "kramabench")


def _reachable():
    for profile, search, benchmark, skills in itertools.product(
        PROFILES, SEARCHES, BENCHMARKS, (True, False)
    ):
        if skills and profile == "naive":
            continue          # --skills on requires profile standard|ideal
        if search == "web" and profile == "ideal":
            continue          # web rejects every ideal axis
        yield profile, search, benchmark, skills


def _compose(profile, search, benchmark, skills):
    if benchmark == "kramabench":
        return P.compose_kramabench_prompt(search, include_skills=skills)
    if profile == "naive":
        return P.compose_baseline_prompt(search)
    return P.compose_managed_prompt(search, include_skills=skills)


def _name(profile, search, benchmark, skills):
    return f"{profile}__{search}__{benchmark}__skills-{'on' if skills else 'off'}.txt"


CASES = list(_reachable())


def test_corpus_is_complete():
    assert len(CASES) == 46, f"expected 46 reachable combinations, got {len(CASES)}"


@pytest.mark.parametrize("case", CASES, ids=lambda c: _name(*c)[:-4])
def test_composed_prompt_matches_golden(case):
    actual = _compose(*case)
    path = GOLDEN / _name(*case)
    if os.getenv("SANA_UPDATE_GOLDEN"):
        path.parent.mkdir(exist_ok=True)
        path.write_text(actual)
        return
    assert path.is_file(), f"missing golden file {path}; regenerate with SANA_UPDATE_GOLDEN=1"
    assert actual == path.read_text(), f"composed prompt changed for {path.name}"
