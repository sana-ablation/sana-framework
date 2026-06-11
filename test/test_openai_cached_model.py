from types import SimpleNamespace
import unittest

from sana_evaluation.llm.openai_cached_model import (
    OpenAICachedUsageModel,
    OpenAIResponsesCachedUsageModel,
    extract_chat_cached_input_tokens,
    extract_responses_cached_input_tokens,
)


class OpenAICachedModelTests(unittest.TestCase):
    def test_extract_chat_cached_input_tokens(self) -> None:
        usage = SimpleNamespace(prompt_tokens_details=SimpleNamespace(cached_tokens=123))
        self.assertEqual(extract_chat_cached_input_tokens(usage), 123)

    def test_extract_responses_cached_input_tokens(self) -> None:
        usage = SimpleNamespace(input_tokens_details=SimpleNamespace(cached_tokens=456))
        self.assertEqual(extract_responses_cached_input_tokens(usage), 456)

    def test_chat_metadata_chunk_includes_cache_read_tokens(self) -> None:
        model = OpenAICachedUsageModel(model_id="gpt-5.2")
        event = {
            "chunk_type": "metadata",
            "data": SimpleNamespace(
                prompt_tokens=1000,
                completion_tokens=200,
                total_tokens=1200,
                prompt_tokens_details=SimpleNamespace(cached_tokens=750),
            ),
        }

        chunk = model.format_chunk(event)
        self.assertEqual(chunk["metadata"]["usage"]["inputTokens"], 1000)
        self.assertEqual(chunk["metadata"]["usage"]["cacheReadInputTokens"], 750)

    def test_responses_metadata_chunk_includes_cache_read_tokens(self) -> None:
        model = OpenAIResponsesCachedUsageModel(model_id="gpt-5.2")
        event = {
            "chunk_type": "metadata",
            "data": SimpleNamespace(
                input_tokens=1200,
                output_tokens=300,
                total_tokens=1500,
                input_tokens_details=SimpleNamespace(cached_tokens=800),
            ),
        }

        chunk = model._format_chunk(event)
        self.assertEqual(chunk["metadata"]["usage"]["inputTokens"], 1200)
        self.assertEqual(chunk["metadata"]["usage"]["cacheReadInputTokens"], 800)


if __name__ == "__main__":
    unittest.main()
