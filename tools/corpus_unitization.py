# Unitizes raw UTF-8 books into deterministic, paragraph-preserving JSON records.

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

_PARAGRAPH_SEPARATOR = re.compile(
    r"(?:\r\n|\r|\n)[ \t]*(?:(?:\r\n|\r|\n)[ \t]*)+"
)
_STRUCTURAL_LABEL = re.compile(
    r"^\s*(?:"
    r"chapter|part|book|volume|prologue|epilogue|appendix|preface|"
    r"foreword|afterword|acknowledg(?:e)?ments?|dedication|contents|"
    r"table\s+of\s+contents|copyright|publication|bibliography|index|notes"
    r")\b",
    re.IGNORECASE,
)
_TOC_RECORD = re.compile(r"^\s*\S.+?(?:\.{2,}|\s{2,})\s*\d+\s*$")
_PAGE_RECORD = re.compile(r"^\s*(?:page\s+)?\d+\s*$", re.IGNORECASE)
_ORNAMENT_RECORD = re.compile(r"^\s*[•*_=~—-]{2,}\s*$")


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


def _split_paragraphs(text: str) -> list[str]:
    paragraphs = [paragraph for paragraph in _PARAGRAPH_SEPARATOR.split(text) if paragraph]
    return paragraphs


def _is_structural_line(line: str) -> bool:
    content = line.strip()
    if not content:
        return False
    return bool(
        _STRUCTURAL_LABEL.match(content)
        or _TOC_RECORD.match(content)
        or _PAGE_RECORD.match(content)
        or _ORNAMENT_RECORD.match(content)
    )


def _split_structural_lines(paragraph: str) -> list[str] | None:
    lines = paragraph.splitlines(keepends=True)
    if len(lines) == 1:
        return [paragraph] if _is_structural_line(lines[0]) else None
    if not any(_is_structural_line(line) for line in lines):
        return None

    units: list[str] = []
    prose_lines: list[str] = []

    def flush_prose() -> None:
        if prose_lines:
            units.extend(_segment_prose("".join(prose_lines)))
            prose_lines.clear()

    for line in lines:
        if _is_structural_line(line):
            flush_prose()
            units.append(line)
        else:
            prose_lines.append(line)
    flush_prose()
    return units


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
    """Return a deterministic nested paragraph/unit representation of one book."""

    normalized_text = normalize_source_text(text)
    segmenter = Segmenter(language="en", clean=False, char_span=True)
    paragraphs: list[dict[str, object]] = []
    next_unit_number = 1

    for paragraph_index, paragraph_text in enumerate(
        _split_paragraphs(normalized_text), start=1
    ):
        structural_units = _split_structural_lines(paragraph_text)
        paragraph_units = structural_units or _segment_prose(paragraph_text, segmenter)
        units = []
        for unit_text in paragraph_units:
            units.append(
                {
                    "id": f"{next_unit_number:0{UNIT_ID_WIDTH}d}",
                    "text": unit_text,
                }
            )
            next_unit_number += 1
        if "".join(unit["text"] for unit in units) != paragraph_text:
            raise ValueError(
                f"unitization changed paragraph text at paragraph {paragraph_index}"
            )
        paragraphs.append(
            {
                "paragraph_index": paragraph_index,
                "text": paragraph_text,
                "units": units,
            }
        )

    result: dict[str, object] = {
        "schema_version": 1,
        "paragraphs": paragraphs,
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
