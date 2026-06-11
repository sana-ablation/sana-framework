import os
import unittest
from unittest.mock import patch

from sana_evaluation.config import AgentConfig
from sana_evaluation.llm.llm_factory import build_model


class OpenAILlmFactoryTests(unittest.TestCase):
    def test_prompt_cache_config_is_forwarded_into_openai_params(self) -> None:
        config = AgentConfig(
            provider="openai",
            model_id="gpt-5.4",
            openai_api_key="test-key",
            openai_prompt_cache_key="assistant-v3:tools-v1",
            openai_prompt_cache_retention="24h",
            max_tokens=1000,
        )

        with patch("sana_evaluation.llm.openai_cached_model.OpenAICachedUsageModel") as mock_model:
            build_model(config)

        _, kwargs = mock_model.call_args
        self.assertEqual(kwargs["model_id"], "gpt-5.4")
        self.assertEqual(kwargs["client_args"]["api_key"], "test-key")
        self.assertEqual(kwargs["params"]["max_completion_tokens"], 1000)
        self.assertEqual(kwargs["params"]["prompt_cache_key"], "assistant-v3:tools-v1")
        self.assertEqual(kwargs["params"]["prompt_cache_retention"], "24h")
        self.assertNotIn("temperature", kwargs["params"])

    def test_prompt_cache_env_defaults_apply_when_config_is_unset(self) -> None:
        config = AgentConfig(
            provider="openai",
            model_id="gpt-5.4",
            openai_api_key="test-key",
            max_tokens=512,
        )

        with patch.dict(
            os.environ,
            {
                "OPENAI_PROMPT_CACHE_KEY": "env-cache-key",
                "OPENAI_PROMPT_CACHE_RETENTION": "24h",
            },
            clear=False,
        ):
            with patch("sana_evaluation.llm.openai_cached_model.OpenAICachedUsageModel") as mock_model:
                build_model(config)

        _, kwargs = mock_model.call_args
        self.assertEqual(kwargs["params"]["prompt_cache_key"], "env-cache-key")
        self.assertEqual(kwargs["params"]["prompt_cache_retention"], "24h")

    def test_explicit_params_override_agent_prompt_cache_defaults(self) -> None:
        config = AgentConfig(
            provider="openai",
            model_id="gpt-5.4",
            openai_api_key="test-key",
            openai_prompt_cache_key="config-cache-key",
            openai_prompt_cache_retention="24h",
            extra_model_kwargs={
                "params": {
                    "prompt_cache_key": "explicit-cache-key",
                    "prompt_cache_retention": "1h",
                    "max_completion_tokens": 333,
                }
            },
        )

        with patch("sana_evaluation.llm.openai_cached_model.OpenAICachedUsageModel") as mock_model:
            build_model(config)

        _, kwargs = mock_model.call_args
        self.assertEqual(kwargs["params"]["prompt_cache_key"], "explicit-cache-key")
        self.assertEqual(kwargs["params"]["prompt_cache_retention"], "1h")
        self.assertEqual(kwargs["params"]["max_completion_tokens"], 333)


if __name__ == "__main__":
    unittest.main()
