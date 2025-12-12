"""Prepare multimodal training pairs by applying random invertible transforms."""
from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import List, Tuple

import torch

import sys

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import alshicrypt  # noqa: E402

ROOT = Path(__file__).resolve().parent
SAMPLES_DIR = ROOT / "samples"
DEFAULT_IMAGE_DIR = SAMPLES_DIR / "images"
DEFAULT_AUDIO_DIR = SAMPLES_DIR / "audio" / "wav"
DATASET_DIR = ROOT / "dataset"


def pick_files(directory: Path, pattern: str, count: int) -> List[Path]:
    files = list(directory.rglob(pattern))
    if len(files) < count:
        raise ValueError(f"Requested {count} files from {directory}, but only found {len(files)}")
    random.shuffle(files)
    return files[:count]


def build_pairs(paths: List[Path], transform: alshicrypt.MediaTransform) -> List[Tuple[alshicrypt.MediaTensor, alshicrypt.MediaTensor]]:
    pairs = []
    for path in paths:
        media = alshicrypt.tensorfy(path)
        encrypted = transform.apply(media)
        pairs.append((media, encrypted))
    return pairs


def serialize_pairs(
    pairs: List[Tuple[alshicrypt.MediaTensor, alshicrypt.MediaTensor]],
    target_file: Path,
    *,
    append: bool = False,
) -> None:
    data = {
        "original": [pair[0] for pair in pairs],
        "encrypted": [pair[1] for pair in pairs],
    }
    torch.save(data, target_file)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, default=DEFAULT_IMAGE_DIR, help="Directory containing image files.")
    parser.add_argument("--audio", type=Path, default=DEFAULT_AUDIO_DIR, help="Directory containing WAV files.")
    parser.add_argument("--num-images", type=int, default=200)
    parser.add_argument("--num-audio", type=int, default=200)
    parser.add_argument("--out", type=Path, default=DATASET_DIR / "train.pt")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    image_files = pick_files(args.images, "*.png", args.num_images)
    audio_files = pick_files(args.audio, "*.wav", args.num_audio)

    transform = alshicrypt.random_media_transform()

    image_pairs = build_pairs(image_files, transform)
    audio_pairs = build_pairs(audio_files, transform)
    DATASET_DIR.mkdir(exist_ok=True)
    serialize_pairs(image_pairs + audio_pairs, args.out)
    print(f"Wrote {len(image_pairs) + len(audio_pairs)} pairs to {args.out}")


if __name__ == "__main__":
    main()
