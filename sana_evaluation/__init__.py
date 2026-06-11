"""
Evaluation Framework for Data Lake Benchmark

This package provides:
- LLM Factory: Unified interface for multiple model providers via Strands
- Agent Tools: Functions for accessing the S3 data lake
- DataLakeAgent / BatchRunner: Strands-native runners for benchmark tasks
- Metrics: Evaluation metrics for comparing answers

Usage:
    from sana_evaluation.agent_with_mode import DataLakeAgent, BatchRunner
    from sana_evaluation.config import AgentConfig, RunConfig

    agent = DataLakeAgent(AgentConfig())
    result = agent.run("What is the capital of France?")

    batch = BatchRunner(AgentConfig())
    results = batch.run_from_files(["tasks/task_1.json", "tasks/task_2.json"])
"""

from .agent_with_mode import DataLakeAgent, BatchRunner
from .config import AgentConfig, RunConfig
from .helper.result import AgentResult
from .llm.llm_factory import build_model
from .helper.metrics import (
    compute_exact_match,
    compute_f1_score,
    normalize_text,
)

__all__ = [
    "DataLakeAgent",
    "BatchRunner",
    "AgentResult",
    "AgentConfig",
    "RunConfig",
    "build_model",
    "compute_exact_match",
    "compute_f1_score",
    "normalize_text",
]
