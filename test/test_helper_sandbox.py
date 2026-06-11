import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from sana_evaluation.helper.sandbox import _cleanup_isolated_sandbox


class IsolatedSandboxCleanupTests(unittest.TestCase):
    def test_cleanup_propagates_rmtree_failures(self):
        with TemporaryDirectory() as tmpdir:
            sandbox = Path(tmpdir) / "sandbox"
            sandbox.mkdir()

            with patch("sana_evaluation.helper.sandbox.shutil.rmtree", side_effect=OSError("busy")):
                with self.assertRaisesRegex(OSError, "busy"):
                    _cleanup_isolated_sandbox(sandbox)


if __name__ == "__main__":
    unittest.main()
