"""Lightweight version discovery must not initialize the generation runtime."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import agent_api_server as server


class CoreVersionTests(unittest.TestCase):
    def test_reads_version_without_executing_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "wgp.py").write_text(
                'raise RuntimeError("must not import")\nWanGP_version = "13.141"\n',
                encoding="utf-8",
            )
            with patch.object(server, "WANGP_ROOT", root):
                self.assertEqual(server._wangp_version(), "13.141")

    def test_missing_or_unrecognized_version(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.object(server, "WANGP_ROOT", root):
                self.assertIsNone(server._wangp_version())
                (root / "wgp.py").write_text("# no version\n", encoding="utf-8")
                self.assertIsNone(server._wangp_version())


if __name__ == "__main__":
    unittest.main()
