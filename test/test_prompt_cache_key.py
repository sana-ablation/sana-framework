"""prompt_cache_key must fit OpenAI's 64-character cap.

/v1/chat/completions accepts an over-long key; /v1/responses answers HTTP 400,
so every model-tier variant label silently exceeded the cap until a Responses-API
model was added.
"""
from sana_evaluation.run_eval import _MAX_PROMPT_CACHE_KEY, _bounded_cache_key

VARIANTS = [
    "search_ideal__results_ideal__profile_ideal__compute_ideal__k5__skills_off",
    "search_naive__results_ideal__profile_ideal__compute_ideal__k5__skills_off",
    "search_standard__results_ideal__profile_ideal__compute_ideal__k5__skills_off",
    "search_preloaded__results_ideal__profile_ideal__compute_ideal__k5__skills_off",
    "search_ideal__results_ideal__profile_naive__compute_ideal__k5__skills_off",
    "search_ideal__results_ideal__profile_standard__compute_ideal__k5__skills_off",
    "search_ideal__results_ideal__profile_ideal__compute_standard__k5__skills_off",
    "search_web__results_naive__profile_standard__compute_standard__nos3__skills_off",
]
MODELS = ["openai_gpt-5.6-luna", "openai_gpt-5-mini", "openai_gpt-5.2", "openai_gpt-5.4-nano"]


def _keys():
    return [_bounded_cache_key(f"{m}:{v}") for m in MODELS for v in VARIANTS]


def test_every_real_variant_key_fits_the_cap():
    for key in _keys():
        assert len(key) <= _MAX_PROMPT_CACHE_KEY, key


def test_variants_stay_distinct_after_bounding():
    keys = _keys()
    assert len(set(keys)) == len(keys), "bounding collapsed two distinct variants"


def test_short_keys_pass_through_unchanged():
    assert _bounded_cache_key("openai_gpt-5-mini:reference") == "openai_gpt-5-mini:reference"


def test_bounding_is_stable_across_calls():
    long = "openai_gpt-5.6-luna:" + "x" * 200
    assert _bounded_cache_key(long) == _bounded_cache_key(long)


def test_labels_differing_only_in_the_tail_get_different_keys():
    # The failure plain truncation would produce: these differ at the very end.
    base = "openai_gpt-5.6-luna:search_ideal__results_ideal__profile_ideal__compute_"
    assert _bounded_cache_key(base + "ideal__k5__skills_off") != _bounded_cache_key(
        base + "standard__k5__skills_off")
