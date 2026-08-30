# Tests deterministic block- and record-preserving corpus unitization without source books.

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from tools.corpus_unitization import (
    normalize_source_text,
    unitize_corpus,
    unitize_text,
)


class CorpusUnitizationTests(unittest.TestCase):
    def test_normalization_changes_only_approved_characters(self) -> None:
        source = "  A\u00adB\tC\u200bD\u200c E\u00a0F\uf0a7  ."

        self.assertEqual(
            normalize_source_text(source),
            "  AB\tCD E F•  .",
        )

    def test_sentence_cases_are_split_deterministically(self) -> None:
        document = unitize_text(
            'Dr. Smith said, "Wait..." Then she left. He stayed.',
        )
        units = document["blocks"][0]["records"][0]["units"]

        self.assertEqual(len(units), 3)
        self.assertIn("Dr. Smith", units[0]["text"])
        self.assertIn("Wait...", units[0]["text"])
        self.assertEqual(units[1]["text"].strip(), "Then she left.")
        self.assertEqual(units[2]["text"].strip(), "He stayed.")

        ellipsis_document = unitize_text("Flip . . . flip . . .")
        ellipsis_record = ellipsis_document["blocks"][0]["records"][0]
        self.assertEqual(
            "".join(unit["text"] for unit in ellipsis_record["units"]),
            ellipsis_record["text"],
        )

    def test_each_nonblank_line_is_a_record(self) -> None:
        examples = [
            "Part of me wanted to leave. I stayed.",
            "Book the room. Then call me.",
            "Notes fell from the desk. He picked them up.",
        ]

        source = "\n".join(examples + ["\uf0a7 First item", "\uf0a7 Second item"])
        records = unitize_text(source)["blocks"][0]["records"]

        self.assertEqual(len(records), 5)
        self.assertEqual([len(record["units"]) for record in records[:3]], [2, 2, 2])
        self.assertEqual([record["text"] for record in records[3:]], ["• First item", "• Second item"])

    def test_block_record_and_separator_reconstruction_is_exact(self) -> None:
        source = "  First line.\r\nSecond line.\r\n\r\n"
        document = unitize_text(source)
        block = document["blocks"][0]

        self.assertEqual(
            "".join(
                record["text"] + record["line_separator"]
                for record in block["records"]
            ),
            "  First line.\r\nSecond line.\r\n",
        )
        self.assertEqual(block["separator_after"], "\r\n")
        self.assertEqual(document["leading_separator"], "")

    def test_leading_separator_is_preserved(self) -> None:
        document = unitize_text("\n\nFirst line.")

        self.assertEqual(document["leading_separator"], "\n\n")
        self.assertEqual(document["blocks"][0]["records"][0]["text"], "First line.")

    def test_ids_are_unique_sequential_and_reset_per_book(self) -> None:
        first = unitize_text("One. Two.")
        second = unitize_text("Another book.")

        first_ids = [
            unit["id"]
            for block in first["blocks"]
            for record in block["records"]
            for unit in record["units"]
        ]
        second_ids = [
            unit["id"]
            for block in second["blocks"]
            for record in block["records"]
            for unit in record["units"]
        ]
        self.assertEqual(first_ids, ["000001", "000002"])
        self.assertEqual(len(first_ids), len(set(first_ids)))
        self.assertEqual(second_ids, ["000001"])

    def test_unit_text_preserves_normalized_record_text(self) -> None:
        source = "  Alpha  beta.\tGamma\n\nHeading\u00a0two"
        document = unitize_text(source)

        expected_records = ["  Alpha  beta.\tGamma", "Heading two"]
        actual_records = [
            record["text"]
            for block in document["blocks"]
            for record in block["records"]
        ]
        self.assertEqual(actual_records, expected_records)
        for block in document["blocks"]:
            for record in block["records"]:
                self.assertEqual(
                    "".join(unit["text"] for unit in record["units"]),
                    record["text"],
                )

    def test_blocks_reconstruct_normalized_source_with_separators(self) -> None:
        source = "\n  First.\nSecond.\n\n\nThird.\n"
        document = unitize_text(source)
        reconstructed = document["leading_separator"]
        for block in document["blocks"]:
            reconstructed += "".join(
                record["text"] + record["line_separator"]
                for record in block["records"]
            )
            reconstructed += block["separator_after"]

        self.assertEqual(reconstructed, normalize_source_text(source))
        for block in document["blocks"]:
            for record in block["records"]:
                self.assertEqual(
                    "".join(unit["text"] for unit in record["units"]),
                    record["text"],
                )

    def test_corpus_is_recursive_mirrored_and_does_not_modify_raw(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            raw_root = root / "data" / "raw"
            output_root = root / "data" / "processed" / "unitized"
            first = raw_root / "TRAIN" / "Series" / "volume-01.txt"
            second = raw_root / "VALIDATION" / "Other Series" / "volume-02.txt"
            first.parent.mkdir(parents=True)
            second.parent.mkdir(parents=True)
            first.write_text("First book.", encoding="utf-8")
            second.write_text("Second book.", encoding="utf-8")
            before = {
                path: hashlib.sha256(path.read_bytes()).hexdigest()
                for path in (first, second)
            }

            output_paths = unitize_corpus(raw_root, output_root)

            self.assertEqual(
                [path.relative_to(output_root).as_posix() for path in output_paths],
                ["TRAIN/Series/volume-01.json", "VALIDATION/Other Series/volume-02.json"],
            )
            self.assertEqual(
                [
                    json.loads((output_root / relative).read_text(encoding="utf-8"))["source_path"]
                    for relative in (Path("TRAIN/Series/volume-01.json"), Path("VALIDATION/Other Series/volume-02.json"))
                ],
                ["TRAIN/Series/volume-01.txt", "VALIDATION/Other Series/volume-02.txt"],
            )
            self.assertEqual(
                before,
                {
                    path: hashlib.sha256(path.read_bytes()).hexdigest()
                    for path in (first, second)
                },
            )

    def test_repeated_unitization_has_identical_json(self) -> None:
        source = "A scene.\n\nChapter 2\n\nAnother scene..."
        first = json.dumps(unitize_text(source), ensure_ascii=False, indent=2)
        second = json.dumps(unitize_text(source), ensure_ascii=False, indent=2)

        self.assertEqual(first, second)

    def test_invalid_utf8_fails_before_creating_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            raw_root = root / "raw"
            output_root = root / "unitized"
            raw_root.mkdir()
            invalid = raw_root / "invalid.txt"
            invalid.write_bytes(b"valid prefix\xff")

            with self.assertRaises(UnicodeDecodeError):
                unitize_corpus(raw_root, output_root)
            self.assertFalse(output_root.exists())


if __name__ == "__main__":
    unittest.main()
