"""End-to-end pipeline for generating transforms, datasets, and training models."""
from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple, Union

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


def _build_pairs(
    paths: Iterable[Path],
    transform,
    *,
    tensor_kwargs: dict | None = None,
    inverse: bool = False,
) -> List[Tuple[MediaTensor, MediaTensor]]:
    pairs = []
    tensor_kwargs = tensor_kwargs or {}
    for path in paths:
        media = tensorfy_media(path, **tensor_kwargs)
        encrypted = transform.apply(media)
        if inverse:
            pairs.append((encrypted, media))
        else:
            pairs.append((media, encrypted))
    return pairs


def _resolve_device(device: Union[str, torch.device, None] = None) -> torch.device:
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but no GPU is available.")
    return resolved


def _batch_to_device(batch, device: torch.device):
    non_blocking = device.type == "cuda"
    return {k: v.to(device, non_blocking=non_blocking) for k, v in batch.items()}


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
    device: torch.device,
) -> Tuple[int, float]:
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=device.type == "cuda",
    )
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
    loader = DataLoader(dataset, batch_size=1, pin_memory=device.type == "cuda")
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
    image_dir: Path | str | None = Path("samples/images"),
    audio_dir: Path | str | None = Path("samples/audio/wav"),
    num_train_images: int = 200,
    num_train_audio: int = 200,
    num_test_images: int = 10,
    num_test_audio: int = 10,
    lr: float = 1e-3,
    batch_size: int = 8,
    max_epochs: int = 200,
    tol: float = 1e-3,
    seed: int = 0,
    device: Union[str, torch.device, None] = None,
    audio_num_samples: int = DEFAULT_AUDIO_NUM_SAMPLES,
    image_latent_dim: int = 2048,
    audio_latent_dim: int = 2048,
    model_layers: int = 4,
    residual: bool = True,
    dropout: float = 0.02,
    expansion: int = 2,
) -> GenerationResult:
    image_dir = Path(image_dir or "samples/images")
    audio_dir = Path(audio_dir or "samples/audio/wav")
    random.seed(seed)
    torch.manual_seed(seed)
    device = _resolve_device(device)
    tensor_kwargs = {"audio_num_samples": audio_num_samples}

    image_train, image_test = _split_files(image_dir, "*.png", num_train_images, num_test_images)
    audio_train, audio_test = _split_files(audio_dir, "*.wav", num_train_audio, num_test_audio)

    transform = random_media_transform(audio_shape=(1, audio_num_samples))

    train_pairs_forward = _build_pairs(image_train + audio_train, transform, tensor_kwargs=tensor_kwargs, inverse=False)
    test_pairs_forward = _build_pairs(image_test + audio_test, transform, tensor_kwargs=tensor_kwargs, inverse=False)
    train_pairs_inverse = _build_pairs(image_train + audio_train, transform, tensor_kwargs=tensor_kwargs, inverse=True)
    test_pairs_inverse = _build_pairs(image_test + audio_test, transform, tensor_kwargs=tensor_kwargs, inverse=True)

    model_config = ModelConfig(
        audio_num_samples=audio_num_samples,
        image_latent_dim=image_latent_dim,
        audio_latent_dim=audio_latent_dim,
        num_layers=model_layers,
        residual=residual,
        dropout=dropout,
        expansion=expansion,
    )
    encryptor = build_model(model_config)
    decryptor = build_model(model_config)

    encrypt_stats = _train_model(
        encryptor,
        MediaPairDataset(train_pairs_forward),
        MediaPairDataset(test_pairs_forward),
        lr=lr,
        batch_size=batch_size,
        max_epochs=max_epochs,
        tol=tol,
        name="encryptor",
        device=device,
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
        device=device,
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
    parser.add_argument(
        "--audio-num-samples",
        type=int,
        default=DEFAULT_AUDIO_NUM_SAMPLES,
        help="Pad/crop audio waveforms to this many samples before training.",
    )
    parser.add_argument(
        "--image-latent-dim",
        type=int,
        default=2048,
        help="Hidden width for the image MLP (must be >= flattened image dim).",
    )
    parser.add_argument(
        "--audio-latent-dim",
        type=int,
        default=2048,
        help="Hidden width for the audio MLP (must be >= padded audio dim).",
    )
    parser.add_argument(
        "--model-layers",
        type=int,
        default=4,
        help="Number of linear layers per modality network.",
    )
    parser.add_argument(
        "--no-residual",
        action="store_true",
        help="Disable residual skip connections inside the modality MLPs.",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.02,
        help="Dropout probability applied inside the modality residual blocks.",
    )
    parser.add_argument(
        "--expansion",
        type=int,
        default=2,
        help="Hidden-width multiplier inside each modality block (>=1).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch device to train on (e.g. 'cuda', 'cuda:0', or 'cpu'). Defaults to GPU when available.",
    )
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
        audio_num_samples=args.audio_num_samples,
        image_latent_dim=args.image_latent_dim,
        audio_latent_dim=args.audio_latent_dim,
        model_layers=args.model_layers,
        residual=not args.no_residual,
        dropout=args.dropout,
        expansion=args.expansion,
        device=args.device,
    )
    print(
        f"Encryptor epochs={result.encrypt_stats[0]}, accuracy={result.encrypt_stats[1]:.3f}; "
        f"Decryptor epochs={result.decrypt_stats[0]}, accuracy={result.decrypt_stats[1]:.3f}"
    )


if __name__ == "__main__":
    main()
