"""Prompt templates for description artifact generation."""

PROMPT_TEMPLATE = """Generate dataset metadata and a concise description for this dataset file.

S3 URI: {dataset_uri}
Metadata: {metadata}
Content sample (first {max_words} words): {dataset_sample}

Requirements:
1) Return JSON with exactly these keys: "generated_metadata", "description".
2) "generated_metadata" should be a cleaned, human-readable metadata phrase (around 5-20 words).
3) "description" must be <= 120 words, factual, and readable.
4) If sample quality is low/noisy, mention uncertainty briefly.
"""

