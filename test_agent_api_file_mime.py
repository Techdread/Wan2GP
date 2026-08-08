#!/usr/bin/env python3
"""Focused regression tests for browser media responses from /api/file."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from agent_api_server import _content_type_for_path


class ContentTypeTests(unittest.TestCase):
    def test_audio_types_are_playable(self):
        expected = {
            "voice.wav": "audio/wav",
            "voice.mp3": "audio/mpeg",
            "voice.flac": "audio/flac",
            "voice.ogg": "audio/ogg",
            "voice.opus": "audio/ogg",
            "voice.m4a": "audio/mp4",
            "voice.aac": "audio/aac",
            "voice.weba": "audio/webm",
        }
        for filename, content_type in expected.items():
            with self.subTest(filename=filename):
                self.assertEqual(_content_type_for_path(Path(filename)), content_type)

    def test_unknown_file_falls_back_to_octet_stream(self):
        self.assertEqual(
            _content_type_for_path(Path("artifact.unknown-wan-output")),
            "application/octet-stream",
        )


if __name__ == "__main__":
    unittest.main()
