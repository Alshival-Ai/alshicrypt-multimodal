"""End-to-end pipeline for generating transforms, datasets, and training models."""
from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

import torch
from torch.utils.data import DataLoader, Dataset

from .data import DEFAULT_AUDIO_NUM_SAMPLES, MediaTensor, tensorfy_media
from .model import ModelConfig, build_model, reconstruction_loss
from .transforms import random_media_transform


# Dataset ---------------------------------------------------------------------


class MediaPairDataset(Dataset):
    """Wraps (input, target) media tensors for model training."""

    def __init__(self, pairs: Sequence[Tuple[MediaTensor, MediaTensor]]) -> None:
        self.pairs = list(pairs)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        source, target = self.pairs[idx]
        return {
            "image_in": source.image,
            "audio_in": source.audio,
            "image_target": target.image,
            "audio_target": target.audio,
            "has_image": torch.tensor(1.0 if source.has_image else 0.0, dtype=torch.float32),
            "has_audio": torch.tensor(1.0 if source.has_audio else 0.0, dtype=torch.float32),
        }


# Utility functions -----------------------------------------------------------


def _pick_files(directory: Path, pattern: str, total: int) -> List[Path]:
    files = [path for path in directory.rglob(pattern) if path.is_file()]
    if len(files) < total:
        raise ValueError(f"Requested {total} files from {directory}, but only found {len(files)}.")
    random.shuffle(files)
    return files[:total]


def _split_files(directory: Path, pattern: str, train_count: int, test_count: int) -> Tuple[List[Path], List[Path]]:
    files = _pick_files(directory, pattern, train_count + test_count)
    return files[:train_count], files[train_count:]


def _build_pairs(paths: Iterable[Path], transform, *, inverse: bool = False) -> List[Tuple[MediaTensor, MediaTensor]]:
    pairs = []
    for path in paths:
        media = tensorfy_media(path)
        encrypted = transform.apply(media)
        if inverse:
            pairs.append((encrypted, media))
        else:
            pairs.append((media, encrypted))
    return pairs


def _batch_to_device(batch, device: torch.device):
    return {k: v.to(device) for k, v in batch.items()}


def _train_model(
    model,
    train_dataset: MediaPairDataset,
    test_dataset: MediaPairDataset,
    *,
    lr: float = 1e-3,
    batch_size: int = 8,
    max_epochs: int = 200,
    tol: float = 1e-3,
    name: str = "model",
) -> Tuple[int, float]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    for epoch in range(1, max_epochs + 1):
        model.train()
        for batch in loader:
            batch = _batch_to_device(batch, device)
            pred_image, pred_audio = model(batch["image_in"], batch["audio_in"])

            total_loss = torch.tensor(0.0, device=device)
            image_mask = batch["has_image"].view(-1).bool()
            audio_mask = batch["has_audio"].view(-1).bool()

            if image_mask.any():
                loss_img = reconstruction_loss(pred_image[image_mask], batch["image_target"][image_mask])
                total_loss = total_loss + loss_img
            if audio_mask.any():
                loss_audio = reconstruction_loss(pred_audio[audio_mask], batch["audio_target"][audio_mask])
                total_loss = total_loss + loss_audio

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

        acc = evaluate_accuracy(model, test_dataset, tol=tol)
        print(f"{name} epoch {epoch}: accuracy={acc:.3f}")
        if acc >= 1.0:
            return epoch, acc
    return max_epochs, acc


def evaluate_accuracy(model, dataset: MediaPairDataset, tol: float = 1e-3) -> float:
    device = next(model.parameters()).device
    loader = DataLoader(dataset, batch_size=1)
    correct = 0
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = _batch_to_device(batch, device)
            pred_image, pred_audio = model(batch["image_in"], batch["audio_in"])
            ok = True
            if batch["has_image"].item() > 0:
                diff = (pred_image - batch["image_target"]).abs().max().item()
                ok = ok and diff <= tol
            if batch["has_audio"].item() > 0:
                diff = (pred_audio - batch["audio_target"]).abs().max().item()
                ok = ok and diff <= tol
            correct += 1 if ok else 0
    return correct / max(1, len(dataset))


# Generation entrypoint -------------------------------------------------------


@dataclass
class GenerationResult:
    transform: object
    encryptor: object
    decryptor: object
    encrypt_stats: Tuple[int, float]
    decrypt_stats: Tuple[int, float]


def generate(
    *,
    image_dir: Path,
    audio_dir: Path,
    num_train_images: int = 200,
    num_train_audio: int = 200,
    num_test_images: int = 10,
    num_test_audio: int = 10,
    lr: float = 1e-3,
    batch_size: int = 8,
    max_epochs: int = 200,
    tol: float = 1e-3,
    seed: int = 0,
) -> GenerationResult:
    random.seed(seed)
    torch.manual_seed(seed)

    image_train, image_test = _split_files(image_dir, "*.png", num_train_images, num_test_images)
    audio_train, audio_test = _split_files(audio_dir, "*.wav", num_train_audio, num_test_audio)

    transform = random_media_transform()

    train_pairs_forward = _build_pairs(image_train + audio_train, transform, inverse=False)
    test_pairs_forward = _build_pairs(image_test + audio_test, transform, inverse=False)
    train_pairs_inverse = _build_pairs(image_train + audio_train, transform, inverse=True)
    test_pairs_inverse = _build_pairs(image_test + audio_test, transform, inverse=True)

    encryptor = build_model(ModelConfig(audio_num_samples=DEFAULT_AUDIO_NUM_SAMPLES))
    decryptor = build_model(ModelConfig(audio_num_samples=DEFAULT_AUDIO_NUM_SAMPLES))

    encrypt_stats = _train_model(
        encryptor,
        MediaPairDataset(train_pairs_forward),
        MediaPairDataset(test_pairs_forward),
        lr=lr,
        batch_size=batch_size,
        max_epochs=max_epochs,
        tol=tol,
        name="encryptor",
    )
    decrypt_stats = _train_model(
        decryptor,
        MediaPairDataset(train_pairs_inverse),
        MediaPairDataset(test_pairs_inverse),
        lr=lr,
        batch_size=batch_size,
        max_epochs=max_epochs,
        tol=tol,
        name="decryptor",
    )

    return GenerationResult(
        transform=transform,
        encryptor=encryptor,
        decryptor=decryptor,
        encrypt_stats=encrypt_stats,
        decrypt_stats=decrypt_stats,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a transform, dataset, and train models.")
    parser.add_argument("--image-dir", type=Path, default=Path("samples/images"))
    parser.add_argument("--audio-dir", type=Path, default=Path("samples/audio/wav"))
    parser.add_argument("--num-train-images", type=int, default=200)
    parser.add_argument("--num-train-audio", type=int, default=200)
    parser.add_argument("--num-test-images", type=int, default=10)
    parser.add_argument("--num-test-audio", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-epochs", type=int, default=200)
    parser.add_argument("--tol", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    result = generate(
        image_dir=args.image_dir,
        audio_dir=args.audio_dir,
        num_train_images=args.num_train_images,
        num_train_audio=args.num_train_audio,
        num_test_images=args.num_test_images,
        num_test_audio=args.num_test_audio,
        lr=args.lr,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        tol=args.tol,
        seed=args.seed,
    )
    print(
        f"Encryptor epochs={result.encrypt_stats[0]}, accuracy={result.encrypt_stats[1]:.3f}; "
        f"Decryptor epochs={result.decrypt_stats[0]}, accuracy={result.decrypt_stats[1]:.3f}"
    )


if __name__ == "__main__":
    main()
