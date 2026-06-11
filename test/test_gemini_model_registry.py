from sana_evaluation.config import resolve_model
from sana_evaluation.helper.constants import MODEL_PRICING


def test_gemini_flash_lite_model_registry_entry() -> None:
    provider, model_id = resolve_model("gemini/gemini-3.1-flash-lite")

    assert provider == "gemini"
    assert model_id == "gemini-3.1-flash-lite"
    assert MODEL_PRICING["gemini/gemini-3.1-flash-lite"]["input"] == 0.25
    assert MODEL_PRICING["gemini/gemini-3.1-flash-lite"]["output"] == 1.50
