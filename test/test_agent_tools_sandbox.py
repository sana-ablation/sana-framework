import importlib.util
import sys
import types
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory


REPO_ROOT = Path(__file__).resolve().parents[1]

_MODULE_NAMES = [
    "strands",
    "boto3",
    "botocore",
    "botocore.config",
    "botocore.exceptions",
    "duckdb",
    "dotenv",
    "requests",
    "sana_evaluation",
    "sana_evaluation.tools",
    "sana_evaluation.tools.helper",
    "sana_evaluation.tools.helper.detect",
    "sana_evaluation.tools.lake",
]


class _Config:
    def __init__(self, *_args, **_kwargs):
        pass


class _ClientError(Exception):
    pass


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _install_import_stubs():
    previous = {name: sys.modules.get(name) for name in _MODULE_NAMES}

    fake_strands = types.ModuleType("strands")
    fake_strands.tool = lambda func: func
    sys.modules["strands"] = fake_strands

    fake_boto3 = types.ModuleType("boto3")
    fake_boto3.client = lambda *_args, **_kwargs: object()
    sys.modules["boto3"] = fake_boto3

    fake_botocore = types.ModuleType("botocore")
    fake_botocore.UNSIGNED = "UNSIGNED"
    sys.modules["botocore"] = fake_botocore

    fake_config = types.ModuleType("botocore.config")
    fake_config.Config = _Config
    sys.modules["botocore.config"] = fake_config

    fake_exceptions = types.ModuleType("botocore.exceptions")
    fake_exceptions.ClientError = _ClientError
    sys.modules["botocore.exceptions"] = fake_exceptions

    fake_dotenv = types.ModuleType("dotenv")
    fake_dotenv.load_dotenv = lambda: None
    sys.modules["dotenv"] = fake_dotenv

    fake_requests = types.ModuleType("requests")
    sys.modules["requests"] = fake_requests

    fake_duckdb = types.ModuleType("duckdb")
    fake_duckdb.DuckDBPyConnection = object
    fake_duckdb.connect = lambda *_args, **_kwargs: None
    sys.modules["duckdb"] = fake_duckdb

    # Stand in for the real packages so ``lake``'s relative imports resolve
    # without executing ``sana_evaluation/__init__.py`` (which pulls in the
    # whole agent stack).
    for name, path in (
        ("sana_evaluation", REPO_ROOT / "sana_evaluation"),
        ("sana_evaluation.tools", REPO_ROOT / "sana_evaluation" / "tools"),
        ("sana_evaluation.tools.helper", REPO_ROOT / "sana_evaluation" / "tools" / "helper"),
    ):
        package = types.ModuleType(name)
        package.__path__ = [str(path)]
        sys.modules[name] = package

    _load_module(
        "sana_evaluation.tools.helper.detect",
        REPO_ROOT / "sana_evaluation" / "tools" / "helper" / "detect.py",
    )

    return previous


def _restore_modules(previous):
    for name, old_value in previous.items():
        if old_value is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = old_value


def _load_lake_module():
    previous = _install_import_stubs()
    try:
        return _load_module(
            "sana_evaluation.tools.lake",
            REPO_ROOT / "sana_evaluation" / "tools" / "lake.py",
        )
    finally:
        _restore_modules(previous)


class AgentToolsSandboxTests(unittest.TestCase):
    def setUp(self):
        self.mod = _load_lake_module()

    def test_cleanup_sandbox_clears_override_after_deleting_pinned_sandbox(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            sandbox = root / "sandbox"
            sandbox.mkdir()
            (sandbox / "data.txt").write_text("x")
            self.mod.SANDBOX_BASE_DIR = root / "base"
            self.mod.set_sandbox_dir(sandbox)

            result = self.mod.cleanup_sandbox()
            next_sandbox = self.mod._get_sandbox_dir()

            self.assertEqual(result.get("status"), "cleaned")
            self.assertIsNone(self.mod._SANDBOX_OVERRIDE)
            self.assertNotEqual(next_sandbox, sandbox)
            self.assertTrue(next_sandbox.exists())


if __name__ == "__main__":
    unittest.main()
