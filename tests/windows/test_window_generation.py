# Tests deterministic literary window construction without copyrighted source prose.

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tools.window_generation import (
    ASSISTANT_RESERVE_TOKENS,
    Unit,
    UnitizedBook,
    generate_book_records,
    partition_targets,
    process_corpus,
    render_units,
    render_user_message,
    serialized_input_token_count,
)


class CharacterTokenizer:
    """Small deterministic tokenizer stub with a chat-template serialization."""

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return list(range(len(text)))

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> list[int]:
        self.last_messages = messages
        assert tokenize is True
        assert add_generation_prompt is True
        serialized = "".join(
            f"<{message['role']}>{message['content']}"
            for message in messages
        )
        serialized += "<assistant>"
        return list(range(len(serialized)))


def make_units(count: int = 8) -> tuple[Unit, ...]:
    return tuple(
        Unit(f"{index:06d}", f"Unit {index}.", 1 if index < 5 else 2)
        for index in range(1, count + 1)
    )


def make_book(path: Path, units: tuple[Unit, ...]) -> UnitizedBook:
    path.write_text("synthetic unitized book", encoding="utf-8")
    return UnitizedBook("SYNTHETIC/book.txt", path, units)


class WindowGenerationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tokenizer = CharacterTokenizer()
        self.system = "SYSTEM PROMPT"

    def test_target_regions_are_exhaustive_consecutive_and_independent_of_context(self) -> None:
        units = make_units()
        target_budget = len(render_units(units[:2]))
        first = partition_targets(units, self.tokenizer, target_budget)
        second = partition_targets(units, self.tokenizer, target_budget)

        self.assertEqual(first, second)
        self.assertEqual(first[0].start, 0)
        self.assertEqual(first[-1].end, len(units))
        self.assertEqual(
            [units[index].id for target in first for index in range(target.start, target.end)],
            [unit.id for unit in units],
        )
        for previous, current in zip(first, first[1:]):
            self.assertEqual(previous.end, current.start)

    def test_exact_user_rendering_preserves_source_block_breaks(self) -> None:
        units = make_units(3)
        actual = render_user_message(units[:1], units[1:2], units[2:])
        expected = (
            "PAST\n[000001] Unit 1.\n\n"
            "TARGET\n[000002] Unit 2.\n\n"
            "FUTURE\n[000003] Unit 3."
        )
        self.assertEqual(actual, expected)

    def test_render_units_uses_blank_lines_only_between_source_blocks(self) -> None:
        units = (
            Unit("000001", "First.", 1),
            Unit("000002", "Second.", 1),
            Unit("000003", "Third.", 2),
        )
        self.assertEqual(
            render_units(units),
            "[000001] First.\n[000002] Second.\n\n[000003] Third.",
        )

    def test_context_overlap_and_edge_behavior(self) -> None:
        units = tuple(Unit(f"{index:06d}", f"U{index}", 1) for index in range(1, 7))
        target_budget = len(render_units(units[:2]))
        targets = partition_targets(units, self.tokenizer, target_budget)
        target_records = []
        for target in targets:
            target_user = render_user_message((), units[target.start : target.end], ())
            target_records.append(
                serialized_input_token_count(self.tokenizer, self.system, target_user)
            )
        max_length = max(target_records) + ASSISTANT_RESERVE_TOKENS + 25
        with tempfile.TemporaryDirectory() as directory:
            book = make_book(Path(directory) / "book.json", units)
            records = generate_book_records(
                book,
                self.tokenizer,
                "synthetic-tokenizer",
                self.system,
                "synthetic-v1",
                "prompt-hash",
                max_length,
                target_budget,
            )

        self.assertEqual(records[0]["metadata"]["past_range"]["unit_count"], 0)
        self.assertGreater(records[0]["metadata"]["future_range"]["unit_count"], 0)
        self.assertEqual(records[-1]["metadata"]["future_range"]["unit_count"], 0)
        for previous, current in zip(records, records[1:]):
            self.assertEqual(
                current["metadata"]["past_range"]["end_id"],
                previous["metadata"]["target_range"]["end_id"],
            )
            previous_future_end = previous["metadata"]["future_range"]["end_id"]
            current_target_start = current["metadata"]["target_range"]["start_id"]
            if previous_future_end is not None:
                future_ids = [
                    f"{index:06d}"
                    for index in range(1, 7)
                    if previous["metadata"]["future_range"]["start_id"]
                    <= f"{index:06d}"
                    <= previous_future_end
                ]
                self.assertEqual(future_ids[0], current_target_start)

    def test_oversized_unit_is_kept_whole(self) -> None:
        units = (
            Unit("000001", "a very long complete unit", 1),
            Unit("000002", "short", 1),
        )
        budget = len(render_units(units[:1])) - 1
        targets = partition_targets(units, self.tokenizer, budget)
        self.assertEqual(
            [(target.start, target.end) for target in targets],
            [(0, 1), (1, 2)],
        )

    def test_serialized_upper_bound_never_exceeds_configured_budget(self) -> None:
        units = make_units(10)
        target_budget = len(render_units(units[:2]))
        max_length = 1000
        with tempfile.TemporaryDirectory() as directory:
            book = make_book(Path(directory) / "book.json", units)
            records = generate_book_records(
                book,
                self.tokenizer,
                "synthetic-tokenizer",
                self.system,
                "synthetic-v1",
                "prompt-hash",
                max_length,
                target_budget,
            )
        self.assertTrue(records)
        self.assertTrue(
            all(
                record["metadata"]["serialized_upper_bound_token_count"] <= max_length
                for record in records
            )
        )

    def test_process_corpus_is_deterministic_and_keeps_metadata_outside_messages(self) -> None:
        units = make_units(5)
        document = {
            "schema_version": 1,
            "leading_separator": "",
            "blocks": [
                {
                    "block_index": 1,
                    "records": [
                        {
                            "record_index": 1,
                            "text": "synthetic",
                            "line_separator": "\n",
                            "units": [
                                {"id": unit.id, "text": unit.text}
                                for unit in units
                            ],
                        }
                    ],
                    "separator_after": "",
                }
            ],
            "source_path": "SYNTHETIC/book.txt",
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_root = root / "unitized"
            output_root = root / "windows"
            input_root.mkdir()
            source = input_root / "TRAIN" / "book.json"
            source.parent.mkdir()
            source.write_text(json.dumps(document), encoding="utf-8")
            prompt_path = root / "scene-segmentation-v1.txt"
            prompt_path.write_text(self.system, encoding="utf-8")
            target_budget = len(render_units(units[:2]))
            first = process_corpus(
                input_root,
                output_root,
                self.tokenizer,
                "synthetic-tokenizer",
                self.system,
                prompt_path,
                1000,
                target_budget,
            )
            first_bytes = (output_root / "TRAIN" / "book.jsonl").read_bytes()
            second = process_corpus(
                input_root,
                output_root,
                self.tokenizer,
                "synthetic-tokenizer",
                self.system,
                prompt_path,
                1000,
                target_budget,
            )
            second_bytes = (output_root / "TRAIN" / "book.jsonl").read_bytes()

        self.assertEqual(first_bytes, second_bytes)
        self.assertEqual(first["aggregate"], second["aggregate"])
        record = json.loads(first_bytes.splitlines()[0])
        self.assertEqual(record["messages"][0]["content"], self.system)
        self.assertNotIn("source_path", record["messages"][1]["content"])
        self.assertIn("source_path", record["metadata"])


if __name__ == "__main__":
    unittest.main()
