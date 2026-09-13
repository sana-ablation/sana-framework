# test/test_orchestration_runner_injection.py
"""The BatchRunner seam existed for a second caller that no longer exists."""
import inspect

from sana_evaluation.runner import orchestration


def test_batch_runner_is_not_a_module_global():
    assert not hasattr(orchestration, "BatchRunner"), (
        "orchestration must not carry a monkey-patchable BatchRunner global"
    )


def test_run_evaluation_takes_the_runner_as_a_parameter():
    sig = inspect.signature(orchestration.run_evaluation)
    assert "batch_runner_cls" in sig.parameters
