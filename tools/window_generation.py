# Generates deterministic overlapping PAST/TARGET/FUTURE windows from unitized books.

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Iterable, Mapping
from typing import Any, Sequence


DEFAULT_INPUT_ROOT = Path("data/processed/unitized")
DEFAULT_OUTPUT_ROOT = Path("data/processed/windows")
DEFAULT_PROMPT_PATH = Path("prompts/scene-segmentation-v1.txt")
DEFAULT_TOKENIZER_NAME = "Qwen/Qwen3.5-0.8B-Base"
ASSISTANT_RESERVE_TOKENS = 512
TARGET_BOUNDARY_BATCH_SIZE = 64
UNIT_ID_WIDTH = 6
UNIT_ID_PATTERN = f"{{:0{UNIT_ID_WIDTH}d}}"


class WindowGenerationError(ValueError):
    """Raised when a unitized book cannot produce valid windows."""


@dataclass(frozen=True)
class Unit:
    id: str
    text: str
    block_index: int


@dataclass(frozen=True)
class TargetRange:
    start: int
    end: int
    token_count: int = -1


@dataclass(frozen=True)
class UnitizedBook:
    source_path: str
    unitized_path: Path
    units: tuple[Unit, ...]


def _require_list(value: Any, name: str) -> list[Any]:
    if not isinstance(value, list):
        raise WindowGenerationError(f"unitized book field {name!r} must be a list")
    return value


def flatten_unitized_book(document: dict[str, Any], unitized_path: Path) -> UnitizedBook:
    """Validate and flatten a unitized document while retaining source block ownership."""

    if document.get("schema_version") != 1:
        raise WindowGenerationError(
            f"unsupported unitized schema in {unitized_path}: "
            f"{document.get('schema_version')!r}"
        )
    source_path = document.get("source_path")
    if not isinstance(source_path, str) or not source_path:
        raise WindowGenerationError(f"missing source_path in {unitized_path}")

    units: list[Unit] = []
    for block_position, block in enumerate(
        _require_list(document.get("blocks"), "blocks"),
        start=1,
    ):
        if not isinstance(block, dict):
            raise WindowGenerationError(f"block {block_position} is not an object")
        block_index = block.get("block_index")
        if not isinstance(block_index, int) or block_index < 1:
            raise WindowGenerationError(
                f"invalid block_index at block position {block_position}"
            )
        records = _require_list(block.get("records"), f"blocks[{block_position}].records")
        for record_position, record in enumerate(records, start=1):
            if not isinstance(record, dict):
                raise WindowGenerationError(
                    f"record {record_position} in block {block_index} is not an object"
                )
            record_units = _require_list(
                record.get("units"),
                f"blocks[{block_position}].records[{record_position}].units",
            )
            for unit in record_units:
                if not isinstance(unit, dict):
                    raise WindowGenerationError("unit entry is not an object")
                unit_id = unit.get("id")
                unit_text = unit.get("text")
                expected_id = UNIT_ID_PATTERN.format(len(units) + 1)
                if unit_id != expected_id:
                    raise WindowGenerationError(
                        f"expected unit ID {expected_id} in {unitized_path}, got {unit_id!r}"
                    )
                if not isinstance(unit_text, str):
                    raise WindowGenerationError(
                        f"unit {unit_id} in {unitized_path} has non-string text"
                    )
                units.append(Unit(unit_id, unit_text, block_index))

    if not units:
        raise WindowGenerationError(f"unitized book contains no units: {unitized_path}")
    return UnitizedBook(source_path, unitized_path, tuple(units))


def load_unitized_book(path: Path) -> UnitizedBook:
    """Load one UTF-8 unitized JSON book."""

    try:
        document = json.loads(path.read_bytes().decode("utf-8"))
    except json.JSONDecodeError as error:
        raise WindowGenerationError(f"invalid JSON in {path}: {error}") from error
    if not isinstance(document, dict):
        raise WindowGenerationError(f"unitized document is not an object: {path}")
    return flatten_unitized_book(document, path)


def render_units(units: Sequence[Unit]) -> str:
    """Render unit lines, using one blank line between different source blocks."""

    if not units:
        return ""
    rendered: list[str] = [f"[{units[0].id}] {units[0].text}"]
    for previous, unit in zip(units, units[1:]):
        separator = "\n\n" if previous.block_index != unit.block_index else "\n"
        rendered.append(separator + f"[{unit.id}] {unit.text}")
    return "".join(rendered)


def render_user_message(
    past: Sequence[Unit], target: Sequence[Unit], future: Sequence[Unit]
) -> str:
    """Render the exact model-facing USER message."""

    return (
        f"PAST\n{render_units(past)}\n\n"
        f"TARGET\n{render_units(target)}\n\n"
        f"FUTURE\n{render_units(future)}"
    )


def _token_ids(value: Any) -> Any:
    if isinstance(value, Mapping):
        if "input_ids" not in value:
            raise WindowGenerationError("tokenizer result does not contain input_ids")
        return value["input_ids"]
    return value


def _token_count(tokenizer: Any, text: str) -> int:
    backend = getattr(tokenizer, "_tokenizer", None)
    if backend is not None:
        return len(backend.encode(text, add_special_tokens=False).ids)
    encoded = tokenizer.encode(text, add_special_tokens=False)
    return len(_token_ids(encoded))


def _token_counts(tokenizer: Any, texts: Sequence[str]) -> tuple[int, ...]:
    """Count several strings in one tokenizer batch when supported."""

    if not texts:
        return ()
    backend = getattr(tokenizer, "_tokenizer", None)
    if backend is not None:
        return tuple(
            len(item.ids)
            for item in backend.encode_batch(list(texts), add_special_tokens=False)
        )
    if hasattr(tokenizer, "__call__"):
        encoded = tokenizer(list(texts), add_special_tokens=False)
        input_ids = _token_ids(encoded)
        return tuple(len(item) for item in input_ids)
    return tuple(_token_count(tokenizer, text) for text in texts)


def serialized_input_token_count(tokenizer: Any, system_message: str, user_message: str) -> int:
    """Count the exact chat-template serialization including generation prompt."""

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message},
    ]
    backend = getattr(tokenizer, "_tokenizer", None)
    if backend is not None:
        rendered = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        return len(backend.encode(rendered, add_special_tokens=False).ids)

    serialized = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
    )
    return len(_token_ids(serialized))


def partition_targets(
    units: Sequence[Unit], tokenizer: Any, target_token_budget: int
) -> tuple[TargetRange, ...]:
    """Partition all units into consecutive whole-unit TARGET regions."""

    if target_token_budget < 1:
        raise WindowGenerationError("target token budget must be positive")
    if not units:
        return ()

    line_costs = _unit_line_token_costs(tokenizer, units)
    newline_cost, blank_line_cost = _token_counts(tokenizer, ("\n", "\n\n"))
    targets: list[TargetRange] = []
    start = 0
    while start < len(units):
        end = start
        approximate_count = 0
        while end < len(units):
            separator_cost = 0
            if end > start:
                separator_cost = (
                    blank_line_cost
                    if units[end - 1].block_index != units[end].block_index
                    else newline_cost
                )
            next_count = approximate_count + line_costs[end] + separator_cost
            if next_count <= target_token_budget or end == start:
                approximate_count = next_count
                end += 1
                continue
            break

        # Correct the approximate position using exact rendered-region counts.
        # Batch forward candidates because the official tokenizer is much
        # faster when it receives a group of strings at once.
        while end > start + 1 and _token_count(
            tokenizer, render_units(units[start:end])
        ) > target_token_budget:
            end -= 1
        while end < len(units):
            batch_end = min(len(units), end + TARGET_BOUNDARY_BATCH_SIZE)
            candidates = tuple(
                render_units(units[start:candidate_end])
                for candidate_end in range(end + 1, batch_end + 1)
            )
            candidate_counts = _token_counts(tokenizer, candidates)
            exceeded = next(
                (
                    offset
                    for offset, candidate_count in enumerate(candidate_counts)
                    if candidate_count > target_token_budget
                ),
                None,
            )
            if exceeded is not None:
                end += exceeded
                break
            end = batch_end
        if end == start:
            raise WindowGenerationError("failed to create a non-empty TARGET region")
        exact_count = _token_count(tokenizer, render_units(units[start:end]))
        targets.append(TargetRange(start, end, exact_count))
        start = end
    return tuple(targets)


def _range_metadata(units: Sequence[Unit]) -> dict[str, Any]:
    if not units:
        return {"start_id": None, "end_id": None, "unit_count": 0}
    return {
        "start_id": units[0].id,
        "end_id": units[-1].id,
        "unit_count": len(units),
    }


def _candidate_window(
    tokenizer: Any,
    system_message: str,
    units: Sequence[Unit],
    target_range: TargetRange,
    past_start: int,
    future_end: int,
    max_sequence_length: int,
) -> tuple[str, int, int, int, int] | None:
    past = units[past_start : target_range.start]
    target = units[target_range.start : target_range.end]
    future = units[target_range.end : future_end]
    user_message = render_user_message(past, target, future)
    serialized_count = serialized_input_token_count(
        tokenizer, system_message, user_message
    )
    total_reserved_count = serialized_count + ASSISTANT_RESERVE_TOKENS
    if total_reserved_count > max_sequence_length:
        return None
    return (
        user_message,
        serialized_count,
        _token_count(tokenizer, render_units(past)),
        _token_count(tokenizer, render_units(future)),
        total_reserved_count,
    )


def _unit_line_token_costs(tokenizer: Any, units: Sequence[Unit]) -> tuple[int, ...]:
    """Count individually rendered unit lines for fast context-size estimates."""

    return _token_counts(
        tokenizer,
        tuple(f"[{unit.id}] {unit.text}" for unit in units),
    )


def select_context(
    units: Sequence[Unit],
    target_range: TargetRange,
    tokenizer: Any,
    system_message: str,
    max_sequence_length: int,
    unit_line_token_costs: Sequence[int] | None = None,
) -> tuple[int, int, str, int, int, int, int]:
    """Fill nearest context units while balancing serialized PAST/FUTURE sizes."""

    if max_sequence_length <= ASSISTANT_RESERVE_TOKENS:
        raise WindowGenerationError(
            "maximum sequence length must exceed the assistant reserve"
        )

    past_start = target_range.start
    future_end = target_range.end
    initial = _candidate_window(
        tokenizer,
        system_message,
        units,
        target_range,
        past_start,
        future_end,
        max_sequence_length,
    )
    if initial is None:
        raise WindowGenerationError(
            f"TARGET {units[target_range.start].id}-{units[target_range.end - 1].id} "
            "cannot fit within the configured serialized sequence length"
        )

    line_costs = tuple(unit_line_token_costs or _unit_line_token_costs(tokenizer, units))
    newline_cost, blank_line_cost = _token_counts(tokenizer, ("\n", "\n\n"))
    separator_costs = tuple(
        blank_line_cost if previous.block_index != current.block_index else newline_cost
        for previous, current in zip(units, units[1:])
    )
    past_approximate_tokens = 0
    future_approximate_tokens = 0
    initial_serialized_count = initial[1]

    while True:
        candidates: list[tuple[str, int, int, int]] = []
        if past_start > 0:
            added_cost = line_costs[past_start - 1]
            if past_start < target_range.start:
                added_cost += separator_costs[past_start - 1]
            candidate_past_tokens = past_approximate_tokens + added_cost
            if (
                initial_serialized_count
                + candidate_past_tokens
                + future_approximate_tokens
                + ASSISTANT_RESERVE_TOKENS
                <= max_sequence_length
            ):
                candidates.append(
                    ("past", past_start - 1, candidate_past_tokens, future_approximate_tokens)
                )
        if future_end < len(units):
            added_cost = line_costs[future_end]
            if future_end > target_range.end:
                added_cost += separator_costs[future_end - 1]
            candidate_future_tokens = future_approximate_tokens + added_cost
            if (
                initial_serialized_count
                + past_approximate_tokens
                + candidate_future_tokens
                + ASSISTANT_RESERVE_TOKENS
                <= max_sequence_length
            ):
                candidates.append(
                    ("future", future_end + 1, past_approximate_tokens, candidate_future_tokens)
                )
        if not candidates:
            break

        def candidate_key(
            item: tuple[str, int, int, int]
        ) -> tuple[int, int, int]:
            side, _, candidate_past_tokens, candidate_future_tokens = item
            side_order = 0 if side == "past" else 1
            return (
                abs(candidate_past_tokens - candidate_future_tokens),
                -(candidate_past_tokens + candidate_future_tokens),
                side_order,
            )

        selected_side, selected_end, selected_past_tokens, selected_future_tokens = min(
            candidates, key=candidate_key
        )
        if selected_side == "past":
            past_start = selected_end
        else:
            future_end = selected_end
        past_approximate_tokens = selected_past_tokens
        future_approximate_tokens = selected_future_tokens

    final = _candidate_window(
        tokenizer,
        system_message,
        units,
        target_range,
        past_start,
        future_end,
        max_sequence_length,
    )
    if final is None:
        # Approximate line costs intentionally avoid expensive full-template
        # calls during greedy selection. Remove the farthest context units
        # until the exact serialized count satisfies the hard limit.
        while final is None and (past_start < target_range.start or future_end > target_range.end):
            if past_start == target_range.start:
                future_end -= 1
            elif future_end == target_range.end:
                past_start += 1
            elif past_approximate_tokens >= future_approximate_tokens:
                past_start += 1
            else:
                future_end -= 1
            final = _candidate_window(
                tokenizer,
                system_message,
                units,
                target_range,
                past_start,
                future_end,
                max_sequence_length,
            )
        if final is None:
            raise WindowGenerationError("context selection produced an invalid window")
    user_message, serialized_count, past_tokens, future_tokens, total_reserved_count = final
    target_tokens = target_range.token_count
    if target_tokens < 0:
        target_tokens = _token_count(
            tokenizer, render_units(units[target_range.start : target_range.end])
        )
    return (
        past_start,
        future_end,
        user_message,
        past_tokens,
        target_tokens,
        future_tokens,
        serialized_count,
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _window_record(
    book: UnitizedBook,
    units: Sequence[Unit],
    target_range: TargetRange,
    window_index: int,
    tokenizer: Any,
    tokenizer_name: str,
    system_message: str,
    prompt_version: str,
    prompt_sha256: str,
    max_sequence_length: int,
    target_token_budget: int,
    unit_line_token_costs: Sequence[int],
    unitized_sha256: str,
) -> dict[str, Any]:
    (
        past_start,
        future_end,
        user_message,
        past_tokens,
        target_tokens,
        future_tokens,
        serialized_count,
    ) = select_context(
        units,
        target_range,
        tokenizer,
        system_message,
        max_sequence_length,
        unit_line_token_costs,
    )
    past = units[past_start : target_range.start]
    target = units[target_range.start : target_range.end]
    future = units[target_range.end : future_end]
    return {
        "metadata": {
            "schema_version": 1,
            "source_path": book.source_path,
            "unitized_path": book.unitized_path.as_posix(),
            "unitized_sha256": unitized_sha256,
            "window_index": window_index,
            "target_range": _range_metadata(target),
            "past_range": _range_metadata(past),
            "future_range": _range_metadata(future),
            "past_token_count": past_tokens,
            "target_token_count": target_tokens,
            "future_token_count": future_tokens,
            "serialized_input_token_count": serialized_count,
            "reserved_assistant_token_count": ASSISTANT_RESERVE_TOKENS,
            "serialized_upper_bound_token_count": serialized_count
            + ASSISTANT_RESERVE_TOKENS,
            "max_sequence_length": max_sequence_length,
            "target_token_budget": target_token_budget,
            "tokenizer": tokenizer_name,
            "prompt_version": prompt_version,
            "prompt_sha256": prompt_sha256,
        },
        "messages": [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ],
    }


def generate_book_records(
    book: UnitizedBook,
    tokenizer: Any,
    tokenizer_name: str,
    system_message: str,
    prompt_version: str,
    prompt_sha256: str,
    max_sequence_length: int,
    target_token_budget: int,
) -> list[dict[str, Any]]:
    """Generate all deterministic window records for one book."""

    targets = partition_targets(book.units, tokenizer, target_token_budget)
    unit_line_token_costs = _unit_line_token_costs(tokenizer, book.units)
    unitized_sha256 = _sha256(book.unitized_path)
    return [
        _window_record(
            book,
            book.units,
            target_range,
            window_index,
            tokenizer,
            tokenizer_name,
            system_message,
            prompt_version,
            prompt_sha256,
            max_sequence_length,
            target_token_budget,
            unit_line_token_costs,
            unitized_sha256,
        )
        for window_index, target_range in enumerate(targets, start=1)
    ]


def _statistic(values: Sequence[int]) -> dict[str, int | float]:
    if not values:
        return {"count": 0, "min": 0, "mean": 0, "max": 0}
    return {
        "count": len(values),
        "min": min(values),
        "mean": round(statistics.fmean(values), 2),
        "max": max(values),
    }


def _summarize_metadata(metadata: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Return stable token statistics for generated metadata."""

    return {
        "book_count": 1,
        "window_count": len(metadata),
        "statistics": {
            name: _statistic([int(item[name]) for item in metadata])
            for name in (
                "past_token_count",
                "target_token_count",
                "future_token_count",
                "serialized_input_token_count",
                "serialized_upper_bound_token_count",
            )
        },
    }


def summarize_records(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Return stable token statistics for generated records."""

    return _summarize_metadata([record["metadata"] for record in records])


def _jsonl_bytes(records: Iterable[dict[str, Any]]) -> bytes:
    return b"".join(
        (
            json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
        ).encode("utf-8")
        for record in records
    )


def _unitized_paths(input_root: Path) -> list[Path]:
    if not input_root.is_dir():
        raise NotADirectoryError(input_root)
    return sorted(
        (path for path in input_root.rglob("*.json") if path.is_file()),
        key=lambda path: path.relative_to(input_root).as_posix(),
    )


def _book_summary(records: Sequence[dict[str, Any]], book: UnitizedBook) -> dict[str, Any]:
    summary = summarize_records(records)
    summary["source_path"] = book.source_path
    return summary


def process_corpus(
    input_root: Path,
    output_root: Path,
    tokenizer: Any,
    tokenizer_name: str,
    system_message: str,
    prompt_path: Path,
    max_sequence_length: int,
    target_token_budget: int,
    report_only: bool = False,
) -> dict[str, Any]:
    """Generate one JSONL file per book or return the report without writing files."""

    prompt_sha256 = hashlib.sha256(prompt_path.read_bytes()).hexdigest()
    prompt_version = prompt_path.stem
    book_summaries: list[dict[str, Any]] = []
    all_metadata: list[dict[str, Any]] = []
    paths = _unitized_paths(input_root)
    if not paths:
        raise WindowGenerationError(f"no unitized JSON books found under {input_root}")

    for input_path in paths:
        book = load_unitized_book(input_path)
        records = generate_book_records(
            book,
            tokenizer,
            tokenizer_name,
            system_message,
            prompt_version,
            prompt_sha256,
            max_sequence_length,
            target_token_budget,
        )
        if not report_only:
            relative_output = input_path.relative_to(input_root).with_suffix(".jsonl")
            output_path = output_root / relative_output
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(_jsonl_bytes(records))
        book_summaries.append(_book_summary(records, book))
        all_metadata.extend(record["metadata"] for record in records)

    aggregate = _summarize_metadata(all_metadata)
    aggregate["book_count"] = len(book_summaries)
    return {
        "input_root": input_root.as_posix(),
        "output_root": output_root.as_posix(),
        "report_only": report_only,
        "tokenizer": tokenizer_name,
        "prompt_version": prompt_version,
        "max_sequence_length": max_sequence_length,
        "target_token_budget": target_token_budget,
        "reserved_assistant_token_count": ASSISTANT_RESERVE_TOKENS,
        "books": book_summaries,
        "aggregate": aggregate,
    }


def _load_tokenizer(name: str) -> Any:
    try:
        from transformers import AutoTokenizer
    except ImportError as error:
        raise SystemExit(
            "The Qwen tokenizer requires transformers in the project environment. "
            "Run `uv sync` with the repository .venv active."
        ) from error
    return AutoTokenizer.from_pretrained(name)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--prompt-path", type=Path, default=DEFAULT_PROMPT_PATH)
    parser.add_argument("--tokenizer", default=DEFAULT_TOKENIZER_NAME)
    parser.add_argument("--max-sequence-length", type=int, required=True)
    parser.add_argument("--target-token-budget", type=int, required=True)
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="generate and measure windows without writing copyrighted prose",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.max_sequence_length < 1:
        raise SystemExit("--max-sequence-length must be positive")
    if args.target_token_budget < 1:
        raise SystemExit("--target-token-budget must be positive")
    try:
        system_message = args.prompt_path.read_bytes().decode("utf-8")
        tokenizer = _load_tokenizer(args.tokenizer)
        report = process_corpus(
            args.input_root,
            args.output_root,
            tokenizer,
            args.tokenizer,
            system_message,
            args.prompt_path,
            args.max_sequence_length,
            args.target_token_budget,
            args.report_only,
        )
    except (OSError, UnicodeError, WindowGenerationError) as error:
        raise SystemExit(f"window generation failed: {error}") from error
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
