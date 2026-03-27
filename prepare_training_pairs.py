from __future__ import annotations

import argparse
import csv
from pathlib import Path

from training.pokemon_pairs import build_pairs


def main() -> None:
    parser = argparse.ArgumentParser(description="Create original/distorted training pair manifest.")
    parser.add_argument("--original-root", default="pokemon", help="Root containing original Pokemon PNG files")
    parser.add_argument(
        "--encoded-root",
        default="pokemon_distorted",
        help="Directory containing distorted PNG files",
    )
    parser.add_argument(
        "--out-csv",
        default="models/pokemon_pairs.csv",
        help="Output pair manifest CSV path",
    )
    args = parser.parse_args()

    original_root = Path(args.original_root)
    encoded_root = Path(args.encoded_root)
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    pairs = build_pairs(original_root, encoded_root)
    if not pairs:
        raise RuntimeError("No training pairs found. Generate distorted images first.")

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["name", "original_path", "encoded_path"])
        for item in pairs:
            writer.writerow([item.name, str(item.original_path), str(item.encoded_path)])

    print(f"Wrote {len(pairs)} pairs to {out_csv}")


if __name__ == "__main__":
    main()
