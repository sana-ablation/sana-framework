"""
AgentConfig — settings dataclass for the Strands evaluation runner.

To create a model from this config, use:
    from sana_evaluation.llm.llm_factory import build_model
    model = build_model(config)

Short model names (e.g. "bedrock/claude-sonnet-4.5") are resolved via MODEL_REGISTRY
into the canonical provider string and actual model ID required by the Strands SDK.
They also serve as the key for pricing lookups in models.py MODEL_PRICING.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple
from sana_evaluation.models import MODEL_REGISTRY


def resolve_model(model_name: str) -> Tuple[str, str]:
    """Return (provider, model_id) for a short model name, or raise ValueError."""
    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model name '{model_name}'. "
            f"Available: {', '.join(MODEL_REGISTRY)}"
        )
    return MODEL_REGISTRY[model_name]


@dataclass
class ConditionConfig:
    condition: str = "baseline"
    base_condition: Optional[str] = None # optional explicit base condition for variant labels
    sparse_backend: str = "bm25"         # "bm25" or "splade"
    trace_output_dir: str = "results/traces"


# The four experiment axes have exactly one set of defaults, keyed by RunConfig
# field name and consumed by every site that used to spell them out: the CLI
# parser and `cli._resolve_mode_axes`, `runner.modes.build_mode_bundle`,
# `preflight.run_preflight`, and the batch worker's backend setup. Sharing them
# is what makes `RunConfig()` and `cli.parse([])` describe the same run *by
# construction* rather than by three copies happening to agree -- the property
# the deleted no-axes agent path silently violated.
AXIS_DEFAULTS = {
    "search_tool_mode": "standard",
    "search_results_mode": "rich",
    "plan_mode": "standard",
    "computation_tool_mode": "standard",
}


@dataclass
class RunConfig:
    results_output_dir: str = "results"
    logs_output_dir: str = "logs"
    debug_mode: Optional[str] = None
    max_tool_calls: int = 30
    conversation_manager_strategy: str = "summarizing"
    sliding_window_k: int = 40
    summary_ratio: float = 0.4
    preserve_recent_messages: int = 12
    timeout_seconds: int = 600
    submit_grace_seconds: int = 30
    submit_only_max_tokens: int = 2048
    tool_executor: str = "sequential"    # or "concurrent"
    condition_config: ConditionConfig = field(default_factory=ConditionConfig)
    max_consecutive_category: int = 6    # CategoryStagnationHandler threshold (0 = disabled)
    search_k: Optional[int] = None
    search_calls_limit: Optional[int] = None
    search_descriptions: str = "naive"
    search_db_path: Optional[str] = None
    # Optional[str] because callers may still pass None explicitly; every reader
    # coalesces it back to the same AXIS_DEFAULTS entry.
    search_tool_mode: Optional[str] = AXIS_DEFAULTS["search_tool_mode"]
    search_results_mode: Optional[str] = AXIS_DEFAULTS["search_results_mode"]
    plan_mode: Optional[str] = AXIS_DEFAULTS["plan_mode"]
    skills_enabled: bool = False
    computation_tool_mode: Optional[str] = AXIS_DEFAULTS["computation_tool_mode"]
    plan_skills_enabled: bool = False
    search_free: bool = False
    no_s3: bool = False
    benchmark: Optional[str] = None


@dataclass
class AgentConfig:
    # ------------------------------------------------------------------
    # Preferred: pass a short model name from MODEL_REGISTRY, e.g.:
    #   AgentConfig(model_name="bedrock/claude-sonnet-4.5")
    # provider and model_id are resolved automatically.
    #
    # For models not in the registry, set provider + model_id directly.
    # ------------------------------------------------------------------
    model_name: Optional[str] = None          # short canonical key (used for pricing)
    provider: str = "bedrock"
    model_id: str = "us.anthropic.claude-sonnet-4-5-20250929-v1:0"
    
    temperature: float = 0.0
    max_tokens: int = 8096
    streaming: bool = False

    bedrock_region: Optional[str] = None      # falls back to AWS_DEFAULT_REGION
    anthropic_api_key: Optional[str] = None   # or set ANTHROPIC_API_KEY
    # Microsoft Foundry (Azure) serves Claude; resource and base_url are
    # mutually exclusive, and one of them is required.
    foundry_api_key: Optional[str] = None     # or set ANTHROPIC_FOUNDRY_API_KEY
    foundry_resource: Optional[str] = None    # or set ANTHROPIC_FOUNDRY_RESOURCE
    foundry_base_url: Optional[str] = None    # or set ANTHROPIC_FOUNDRY_BASE_URL
    openai_api_key: Optional[str] = None      # or set OPENAI_API_KEY
    openai_base_url: Optional[str] = None     # for Azure / vLLM / compatible APIs
    openai_prompt_cache_key: Optional[str] = None        # optional stable OpenAI prompt cache key
    openai_prompt_cache_retention: Optional[str] = None  # optional OpenAI retention e.g. "24h"
    gemini_api_key: Optional[str] = None      # or set GEMINI_API_KEY
    ollama_host: str = "http://localhost:11434"
    llama_api_key: Optional[str] = None       # or set LLAMA_API_KEY
    litellm_base_url: Optional[str] = None    # or set LITELLM_BASE_URL
    litellm_api_key: Optional[str] = None     # or set LITELLM_API_KEY
    mistral_api_key: Optional[str] = None     # or set MISTRAL_API_KEY
    cohere_api_key: Optional[str] = None      # or set COHERE_API_KEY
    sagemaker_endpoint: Optional[str] = None  # deployed endpoint name
    sagemaker_region: Optional[str] = None
    writer_api_key: Optional[str] = None      # or set WRITER_API_KEY
    extra_model_kwargs: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.model_name:
            self.provider, self.model_id = resolve_model(self.model_name)
