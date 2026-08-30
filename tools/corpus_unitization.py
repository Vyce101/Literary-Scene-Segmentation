# Unitizes raw UTF-8 books into deterministic, block- and record-preserving JSON.

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Iterable

from pysbd import Segmenter


DEFAULT_RAW_ROOT = Path("data/raw")
DEFAULT_OUTPUT_ROOT = Path("data/processed/unitized")
UNIT_ID_WIDTH = 6


def normalize_source_text(text: str) -> str:
    """Apply only the approved source-character normalizations."""

    return text.translate(
        {
            ord("\u00ad"): None,
            ord("\u200b"): None,
            ord("\u200c"): None,
            ord("\uf0a7"): "•",
            ord("\u00a0"): " ",
        }
    )


def _text_files(root: Path) -> Iterable[Path]:
    return sorted(
        (
            path
            for path in root.rglob("*.txt")
            if path.is_file()
        ),
        key=lambda path: path.relative_to(root).as_posix(),
    )


def _split_line_ending(line: str) -> tuple[str, str]:
    if line.endswith("\r\n"):
        return line[:-2], "\r\n"
    if line.endswith(("\r", "\n")):
        return line[:-1], line[-1]
    return line, ""


def _split_blocks(text: str) -> tuple[str, list[dict[str, object]]]:
    leading_separator = ""
    blocks: list[dict[str, object]] = []
    records: list[tuple[str, str]] = []
    block_separator = ""

    for line in text.splitlines(keepends=True):
        record_text, line_separator = _split_line_ending(line)
        if record_text.strip():
            records.append((record_text, line_separator))
            continue

        if records:
            blocks.append(
                {
                    "records": records,
                    "separator_after": block_separator + line,
                }
            )
            records = []
            block_separator = ""
        elif blocks:
            blocks[-1]["separator_after"] = (
                str(blocks[-1]["separator_after"]) + line
            )
        else:
            leading_separator += line

    if records:
        blocks.append({"records": records, "separator_after": ""})

    return leading_separator, blocks


def _segment_prose(text: str, segmenter: Segmenter | None = None) -> list[str]:
    active_segmenter = segmenter or Segmenter(
        language="en",
        clean=False,
        char_span=True,
    )
    spans = list(active_segmenter.segment(text))
    if not spans:
        return []

    # Use source offsets so every emitted unit remains an exact source slice.
    # Overlapping spans are merged conservatively instead of inventing a
    # sentence boundary from an ambiguous library result.
    lossless_segments: list[str] = []
    cursor = 0
    for span in spans:
        if span.start < cursor:
            continue
        start = min(span.start, len(text))
        end = min(max(span.end, start), len(text))
        if start > cursor and lossless_segments:
            lossless_segments[-1] += text[cursor:start]
        if start > cursor and not lossless_segments:
            start = cursor
        if end > start:
            lossless_segments.append(text[start:end])
        cursor = end

    if lossless_segments and cursor < len(text):
        lossless_segments[-1] += text[cursor:]
    elif not lossless_segments:
        lossless_segments.append(text)
    if "".join(lossless_segments) != text:
        raise ValueError("sentence segmentation did not preserve normalized source text")
    return lossless_segments


def unitize_text(text: str, source_path: str | None = None) -> dict[str, object]:
    """Return a deterministic nested block/record/unit representation of one book."""

    normalized_text = normalize_source_text(text)
    segmenter = Segmenter(language="en", clean=False, char_span=True)
    leading_separator, source_blocks = _split_blocks(normalized_text)
    blocks: list[dict[str, object]] = []
    next_unit_number = 1

    for block_index, source_block in enumerate(source_blocks, start=1):
        records: list[dict[str, object]] = []
        for record_index, (record_text, line_separator) in enumerate(
            source_block["records"], start=1
        ):
            units = []
            for unit_text in _segment_prose(record_text, segmenter):
                units.append(
                    {
                        "id": f"{next_unit_number:0{UNIT_ID_WIDTH}d}",
                        "text": unit_text,
                    }
                )
                next_unit_number += 1
            if "".join(unit["text"] for unit in units) != record_text:
                raise ValueError(
                    f"unitization changed record text at block {block_index}, "
                    f"record {record_index}"
                )
            records.append(
                {
                    "record_index": record_index,
                    "text": record_text,
                    "line_separator": line_separator,
                    "units": units,
                }
            )
        block_text = "".join(
            str(record["text"]) + str(record["line_separator"])
            for record in records
        )
        expected_block_text = "".join(
            record_text + line_separator
            for record_text, line_separator in source_block["records"]
        )
        if block_text != expected_block_text:
            raise ValueError(f"block reconstruction failed at block {block_index}")
        blocks.append(
            {
                "block_index": block_index,
                "records": records,
                "separator_after": source_block["separator_after"],
            }
        )

    result: dict[str, object] = {
        "schema_version": 1,
        "leading_separator": leading_separator,
        "blocks": blocks,
    }
    if source_path is not None:
        result["source_path"] = source_path
    return result


def _read_utf8(path: Path) -> str:
    return path.read_bytes().decode("utf-8")


def unitize_book(raw_path: Path, raw_root: Path = DEFAULT_RAW_ROOT) -> dict[str, object]:
    relative_path = raw_path.relative_to(raw_root).as_posix()
    return unitize_text(_read_utf8(raw_path), source_path=relative_path)


def write_unitized_book(
    raw_path: Path,
    raw_root: Path = DEFAULT_RAW_ROOT,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
) -> Path:
    relative_path = raw_path.relative_to(raw_root)
    output_path = output_root / relative_path.with_suffix(".json")
    document = unitize_book(raw_path, raw_root)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(document, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return output_path


def unitize_corpus(
    raw_root: Path = DEFAULT_RAW_ROOT,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
) -> list[Path]:
    """Unitize every raw text file and return the generated paths."""

    if not raw_root.is_dir():
        raise NotADirectoryError(raw_root)
    raw_paths = list(_text_files(raw_root))
    # Validate the complete input set before creating derived output, without
    # retaining all normalized books in memory during the subsequent write pass.
    for raw_path in raw_paths:
        _read_utf8(raw_path)
    return [
        write_unitized_book(raw_path, raw_root, output_root)
        for raw_path in raw_paths
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args()

    output_paths = unitize_corpus(args.raw_root, args.output_root)
    print(f"Unitized {len(output_paths)} books.")


if __name__ == "__main__":
    main()
