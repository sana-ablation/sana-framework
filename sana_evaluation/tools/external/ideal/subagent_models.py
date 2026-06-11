"""Model selection for hidden ideal-mode helper agents."""

from __future__ import annotations

import os

MAIN_MODEL_ENV = "SANA_MAIN_MODEL"
IDEAL_SUBAGENT_MODEL_ENV = "SANA_IDEAL_SUBAGENT_MODEL"
SEARCH_IDEAL_SUBAGENT_MODEL_ENV = "SANA_SEARCH_IDEAL_SUBAGENT_MODEL"
SEMANTIC_IDEAL_SUBAGENT_MODEL_ENV = "SANA_SEMANTIC_IDEAL_SUBAGENT_MODEL"
REPAIR_IDEAL_SUBAGENT_MODEL_ENV = "SANA_REPAIR_IDEAL_SUBAGENT_MODEL"


def _model_from_env(*names: str) -> str:
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    raise RuntimeError(
        "Ideal subagent model is not configured. Run through setup_run/run_mode_eval "
        "or set SANA_IDEAL_SUBAGENT_MODEL."
    )


def search_model_name() -> str:
    return _model_from_env(
        SEARCH_IDEAL_SUBAGENT_MODEL_ENV,
        IDEAL_SUBAGENT_MODEL_ENV,
        MAIN_MODEL_ENV,
    )


def semantic_model_name() -> str:
    return _model_from_env(
        SEMANTIC_IDEAL_SUBAGENT_MODEL_ENV,
        IDEAL_SUBAGENT_MODEL_ENV,
        MAIN_MODEL_ENV,
    )


def repair_model_name() -> str:
    return _model_from_env(
        REPAIR_IDEAL_SUBAGENT_MODEL_ENV,
        IDEAL_SUBAGENT_MODEL_ENV,
        MAIN_MODEL_ENV,
    )
