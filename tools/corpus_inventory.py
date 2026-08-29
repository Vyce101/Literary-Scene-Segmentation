# Inventories UTF-8 text files in the raw corpus without storing source text.

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterable


DEFAULT_RAW_ROOT = Path("data/raw")
DEFAULT_OUTPUT_PATH = Path("manifests/corpus-inventory.json")


def _text_files(root: Path) -> Iterable[Path]:
    paths = (
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() == ".txt"
    )
    return sorted(paths, key=lambda path: path.relative_to(root).as_posix())


def build_inventory(root: Path) -> dict[str, object]:
    """Build metadata for every UTF-8 text file below *root*."""
    if not root.is_dir():
        raise NotADirectoryError(root)

    files: list[dict[str, object]] = []
    total_byte_size = 0
    total_word_count = 0

    for path in _text_files(root):
        file_bytes = path.read_bytes()
        try:
            text = file_bytes.decode("utf-8")
        except UnicodeDecodeError as error:
            relative_path = path.relative_to(root).as_posix()
            raise UnicodeDecodeError(
                error.encoding,
                error.object,
                error.start,
                error.end,
                f"{relative_path}: {error.reason}",
            ) from error

        byte_size = len(file_bytes)
        word_count = len(text.split())
        files.append(
            {
                "relative_path": path.relative_to(root).as_posix(),
                "sha256": hashlib.sha256(file_bytes).hexdigest(),
                "byte_size": byte_size,
                "word_count": word_count,
            }
        )
        total_byte_size += byte_size
        total_word_count += word_count

    return {
        "files": files,
        "totals": {
            "file_count": len(files),
            "byte_size": total_byte_size,
            "word_count": total_word_count,
        },
    }


def write_inventory(root: Path = DEFAULT_RAW_ROOT, output_path: Path = DEFAULT_OUTPUT_PATH) -> dict[str, object]:
    """Write the raw-corpus inventory and return its metadata."""
    inventory = build_inventory(root)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(inventory, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return inventory


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    args = parser.parse_args()

    inventory = write_inventory(args.root, args.output)
    print(f"Inventoried {inventory['totals']['file_count']} text files.")


if __name__ == "__main__":
    main()
