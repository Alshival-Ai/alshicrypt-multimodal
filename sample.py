"""Example script showing how to load an image and convert it to a tensor via the pipeline."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import alshicrypt  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "image",
        type=Path,
        nargs="?",
        help="Path to an image file sampled from the CIFAR export (e.g., samples/images/dog/data_batch_1_...).",
    )
    parser.add_argument("--image-size", type=int, default=32, help="Resize shorter edge to this size before tensor conversion.")
    parser.add_argument("--no-normalize", action="store_true", help="Skip CIFAR normalization stage.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for DataLoader example.")
    parser.add_argument("--root", type=Path, default=Path("samples/images"), help="Root directory that holds class folders.")
    parser.add_argument("--audio-sample-rate", type=int, default=alshicrypt.DEFAULT_AUDIO_SAMPLE_RATE, help="Resample audio to this rate.")
    parser.add_argument("--audio-num-samples", type=int, default=alshicrypt.DEFAULT_AUDIO_NUM_SAMPLES, help="Pad/crop audio to this many samples.")
    parser.add_argument("--demo-transform", action="store_true", help="Apply a random invertible transform and show recovery statistics.")
    return parser.parse_args()


def _find_default_image(root: Path) -> Path:
    root = root.expanduser()
    if not root.exists():
        raise FileNotFoundError(f"Could not find dataset directory at {root}.")
    for path in root.rglob("*.png"):
        if path.is_file():
            return path
    raise FileNotFoundError(f"No PNG files were discovered under {root}. Did you export CIFAR images first?")


def main() -> None:
    args = parse_args()

    image_path = args.image or _find_default_image(args.root)
    if args.image is None:
        print(f"No image argument provided; defaulting to {image_path}")

    media = alshicrypt.tensorfy(
        image_path,
        image_size=args.image_size,
        normalize=not args.no_normalize,
        audio_sample_rate=args.audio_sample_rate,
        audio_num_samples=args.audio_num_samples,
    )
    print(
        f"Image tensor shape={media.image.shape}, has_image={media.has_image}, "
        f"dtype={media.image.dtype}, min={media.image.min():.3f}, max={media.image.max():.3f}"
    )
    print(
        f"Audio tensor shape={media.audio.shape}, has_audio={media.has_audio}, "
        f"dtype={media.audio.dtype}, min={media.audio.min():.3f}, max={media.audio.max():.3f}"
    )

    if args.demo_transform:
        transform = alshicrypt.random_media_transform(
            image_shape=tuple(media.image.shape) if media.has_image else None,
            audio_shape=tuple(media.audio.shape) if media.has_audio else None,
        )
        encrypted = transform.apply(media)
        recovered = transform.invert(encrypted)
        img_diff = (recovered.image - media.image).abs().max().item()
        aud_diff = (recovered.audio - media.audio).abs().max().item()
        print(
            f"Transform demo -> image max|diff|={img_diff:.6f}, audio max|diff|={aud_diff:.6f}"
        )

    pipeline = alshicrypt.ImageTensorPipeline(
        root=args.root,
        batch_size=args.batch_size,
        image_size=args.image_size,
        normalize=not args.no_normalize,
    )
    loader = pipeline.dataloader()
    batch_images, batch_labels = next(iter(loader))
    print(
        f"Batch images tensor shape={batch_images.shape}, labels shape={batch_labels.shape},"
        f" dtype={batch_images.dtype}"
    )


if __name__ == "__main__":
    main()
