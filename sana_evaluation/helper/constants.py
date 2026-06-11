from typing import Dict, Tuple
import os 
from dotenv import load_dotenv

# This physically reads your .env file and injects it into os.environ
load_dotenv()

HAIKU_ARN = os.getenv("BEDROCK_HAIKU_ARN", "us.anthropic.claude-haiku-4-5-20251001-v1:0")
SONNET_ARN = os.getenv("BEDROCK_SONNET_ARN", "us.anthropic.claude-sonnet-4-5-20250929-v1:0")
LLAMA_ARN = os.getenv("BEDROCK_LLAMA_ARN", "us.meta.llama3-3-70b-instruct-v1:0")
QWEN_ARN = os.getenv("BEDROCK_QWEN_ARN", "qwen.qwen3-vl-235b-a22b")

# ---------------------------------------------------------------------------
# Model registry — short name -> (provider, actual_model_id)
# Use these short names in AgentConfig(model_name=...) and for pricing lookups.
# ---------------------------------------------------------------------------
MODEL_REGISTRY: Dict[str, Tuple[str, str]] = {
    # Anthropic via Bedrock (cross-region inference)
    "bedrock/claude-opus-4.5":     ("bedrock", "us.anthropic.claude-opus-4-5-20251101-v1:0"),
    "bedrock/claude-sonnet-4.5":   ("bedrock", "us.anthropic.claude-sonnet-4-5-20250929-v1:0"),
    "bedrock/claude-haiku-4.5":    ("bedrock", "us.anthropic.claude-haiku-4-5-20251001-v1:0"),

    # Custom application inference profiles (ARN-based)
    "bedrock/claude-haiku-4.5-arn":  ("bedrock", HAIKU_ARN),
    "bedrock/claude-sonnet-4.5-arn": ("bedrock", SONNET_ARN),
    "bedrock/llama-3.3-70b-arn":     ("bedrock", LLAMA_ARN),
    "bedrock/qwen3-235b-arn":        ("bedrock", QWEN_ARN),
    # Llama via Bedrock
    "bedrock/llama4-maverick":     ("bedrock", "us.meta.llama4-maverick-17b-instruct-v1:0"),
    "bedrock/llama4-scout":        ("bedrock", "us.meta.llama4-scout-17b-instruct-v1:0"),
    "bedrock/llama3.3-70b":        ("bedrock", "us.meta.llama3-3-70b-instruct-v1:0"),
    "bedrock/llama3.1-70b":        ("bedrock", "us.meta.llama3-1-70b-instruct-v1:0"),

    # Mistral via Bedrock
    "bedrock/mistral-large":       ("bedrock", "mistral.mistral-large-3-675b-instruct"),
    "bedrock/mixtral-8x7b":        ("bedrock", "mistral.mixtral-8x7b-instruct-v0:1"),

    # Qwen via Bedrock
    "bedrock/qwen3-32b":           ("bedrock", "qwen.qwen3-32b-v1:0"),
    "bedrock/qwen3-80b":           ("bedrock", "qwen.qwen3-next-80b-a3b"),
    "bedrock/qwen3-235b":          ("bedrock", "qwen.qwen3-vl-235b-a22b"),

    # DeepSeek via Bedrock
    "bedrock/deepseek-r1":         ("bedrock", "us.deepseek.r1-v1:0"),

    # OpenAI
    "openai/gpt-5.2":              ("openai", "gpt-5.2"),
    "openai/gpt-5-nano":           ("openai", "gpt-5-nano"),
    "openai/gpt-5-mini":           ("openai", "gpt-5-mini"),
    "openai/gpt-5.4":              ("openai", "gpt-5.4"),
    "openai/gpt-5.4-nano":         ("openai", "gpt-5.4-nano"),

    # Google Gemini API
    "gemini/gemini-3.1-flash-lite": ("gemini", "gemini-3.1-flash-lite"),

    # Anthropic direct API
    "anthropic/claude-opus-4.6":   ("anthropic", "claude-opus-4-6-20251101"),
    "anthropic/claude-sonnet-4.5": ("anthropic", "claude-sonnet-4-5-20251001"),
    "anthropic/claude-haiku-4.5":  ("anthropic", "claude-haiku-4-5-20251001"),
}

MODEL_PRICING = {
        # OpenAI text-token pricing per 1M tokens.
        # Source: https://platform.openai.com/docs/pricing/
        "gpt-5.2": {"input": 1.75, "cache_read_input": 0.175, "output": 14.00},
        "gpt-5-nano": {"input": 0.05, "cache_read_input": 0.005, "output": 0.40},
        "gpt-5-mini": {"input": 0.25, "cache_read_input": 0.025, "output": 2.00},
        "gpt-5.4": {"input": 2.50, "cache_read_input": 0.25, "output": 15.00},
        "gpt-5.4-nano": {"input": 0.20, "cache_read_input": 0.02, "output": 1.25},
        # Alias keys for provider-prefixed model_name values
        "openai/gpt-5.2": {"input": 1.75, "cache_read_input": 0.175, "output": 14.00},
        "openai/gpt-5-nano": {"input": 0.05, "cache_read_input": 0.005, "output": 0.40},
        "openai/gpt-5-mini": {"input": 0.25, "cache_read_input": 0.025, "output": 2.00},
        "openai/gpt-5.4": {"input": 2.50, "cache_read_input": 0.25, "output": 15.00},
        "openai/gpt-5.4-nano": {"input": 0.20, "cache_read_input": 0.02, "output": 1.25},

        # Google Gemini API
        "gemini/gemini-3.1-flash-lite": {"input": 0.25, "output": 1.50},

        # AWS Bedrock Claude models
        "bedrock/claude-opus-4.5": {"input": 5.00, "output": 25.00},
        "bedrock/claude-sonnet-4.5": {"input": 3.00, "output": 15.00},
        "bedrock/claude-haiku-4.5": {"input": 1.00, "output": 5.00},
        "bedrock/claude-haiku-4.5-arn": {"input": 1.00, "output": 5.00},
        "bedrock/claude-sonnet-4.5-arn": {"input": 3.00, "output": 15.00},

        # AWS Bedrock Llama models
        "bedrock/llama4-maverick": {"input": 0.24, "output": 0.97},
        "bedrock/llama4-scout": {"input": 0.17, "output": 0.66},
        "bedrock/llama3.3-70b": {"input": 0.72, "output": 0.72},
        "bedrock/llama3.1-70b": {"input": 0.72, "output": 0.72},
        "bedrock/llama-3.3-70b-arn": {"input": 0.72, "output": 0.72},

        # AWS Bedrock Mistral models
        "bedrock/mistral-large": {"input": 0.50, "output": 1.50},
        "bedrock/mixtral-8x7b": {"input": 0.45, "output": 0.70},

        # AWS Bedrock Qwen models
        "bedrock/qwen3-32b": {"input": 0.15, "output": 0.60},
        "bedrock/qwen3-80b": {"input": 0.15, "output": 1.20},
        "bedrock/qwen3-235b": {"input": 0.53, "output": 2.66},
        "bedrock/qwen3-235b-arn": {"input": 0.53, "output": 2.66},

        # AWS Bedrock DeepSeek
        "bedrock/deepseek-r1": {"input": 1.35, "output": 5.40},

        # Together AI
        "together/qwen3-235b": {"input": 0.65, "output": 3.00},
        "together/qwen3-32b": {"input": 0.50, "output": 1.50},
        "together/llama4-maverick": {"input": 0.27, "output": 0.85},
        "together/deepseek-r1": {"input": 3.00, "output": 7.00},

        # Fireworks AI
        "fireworks/qwen3-235b": {"input": 0.22, "output": 0.88},
        "fireworks/llama4-maverick": {"input": 0.17, "output": 0.17},

        # OpenRouter
        "openrouter/qwen3-235b": {"input": 1.00, "output": 1.00},
    }
