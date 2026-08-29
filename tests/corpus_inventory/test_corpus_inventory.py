# Tests raw-corpus metadata inventory without using copyrighted source text.

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from tools.corpus_inventory import build_inventory, write_inventory


class CorpusInventoryTests(unittest.TestCase):
    def test_inventory_is_recursive_and_contains_metadata_only(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            first_bytes = "alpha  beta\n gamma".encode("utf-8")
            second_bytes = "one\ttwo\nthree".encode("utf-8")
            (root / "first.txt").write_bytes(first_bytes)
            (root / "nested").mkdir()
            (root / "nested" / "second.txt").write_bytes(second_bytes)
            (root / "ignored.md").write_text("not included", encoding="utf-8")

            output_path = root / "manifest.json"
            write_inventory(root, output_path)
            inventory = json.loads(output_path.read_text(encoding="utf-8"))

            self.assertEqual(
                inventory,
                {
                    "files": [
                        {
                            "relative_path": "first.txt",
                            "sha256": hashlib.sha256(first_bytes).hexdigest(),
                            "byte_size": len(first_bytes),
                            "word_count": 3,
                        },
                        {
                            "relative_path": "nested/second.txt",
                            "sha256": hashlib.sha256(second_bytes).hexdigest(),
                            "byte_size": len(second_bytes),
                            "word_count": 3,
                        },
                    ],
                    "totals": {
                        "file_count": 2,
                        "byte_size": len(first_bytes) + len(second_bytes),
                        "word_count": 6,
                    },
                },
            )
            manifest_text = output_path.read_text(encoding="utf-8")
            self.assertNotIn("alpha", manifest_text)
            self.assertNotIn("one", manifest_text)

    def test_invalid_utf8_fails_before_writing_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            (root / "invalid.txt").write_bytes(b"valid prefix\xff")
            output_path = root / "manifest.json"

            with self.assertRaises(UnicodeDecodeError):
                write_inventory(root, output_path)

            self.assertFalse(output_path.exists())

    def test_build_inventory_rejects_missing_root(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            missing_root = Path(temporary_directory) / "missing"

            with self.assertRaises(NotADirectoryError):
                build_inventory(missing_root)


if __name__ == "__main__":
    unittest.main()
