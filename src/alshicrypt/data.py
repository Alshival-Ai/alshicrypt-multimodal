"""Data loading utilities for converting CIFAR images into tensors."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import soundfile as sf
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torchaudio

# Channel statistics for CIFAR-10 as reported by torchvision docs.
CIFAR10_MEAN: Sequence[float] = (0.4914, 0.4822, 0.4465)
CIFAR10_STD: Sequence[float] = (0.2470, 0.2435, 0.2616)

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}
AUDIO_EXTENSIONS = {".wav", ".wave"}
DEFAULT_AUDIO_SAMPLE_RATE = 16000
DEFAULT_AUDIO_NUM_SAMPLES = 16000


@dataclass
class MediaTensor:
    """Container for multimodal tensors."""

    image: torch.Tensor
    audio: torch.Tensor
    has_image: bool
    has_audio: bool


def _ensure_root(path: Path) -> Path:
    path = path.expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Dataset root '{path}' was not found. Did you extract CIFAR into samples/images?")
    return path


def image_to_tensor(image_path: Path | str, *, image_size: int = 32, normalize: bool = True) -> torch.Tensor:
    """Load a single image file and convert it into a tensor ready for the models."""

    transform_ops = [transforms.Resize((image_size, image_size)), transforms.ToTensor()]
    if normalize:
        transform_ops.append(transforms.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD))
    transform = transforms.Compose(transform_ops)
    tensor = transform(Image.open(image_path).convert("RGB"))
    return tensor


def audio_to_tensor(
    audio_path: Path | str,
    *,
    sample_rate: int = DEFAULT_AUDIO_SAMPLE_RATE,
    num_samples: int | None = DEFAULT_AUDIO_NUM_SAMPLES,
    mono: bool = True,
) -> torch.Tensor:
    """Load a WAV file into a tensor and normalize its length."""

    data, file_sr = sf.read(str(audio_path), dtype="float32")
    waveform = torch.from_numpy(data.T)
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    if mono and waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if file_sr != sample_rate:
        waveform = torchaudio.functional.resample(waveform, file_sr, sample_rate)
    if num_samples is not None:
        if waveform.shape[-1] < num_samples:
            pad = num_samples - waveform.shape[-1]
            waveform = F.pad(waveform, (0, pad))
        else:
            waveform = waveform[..., :num_samples]
    return waveform


@dataclass
class ImageTensorPipeline:
    """Builds `torch.utils.data.DataLoader` objects for CIFAR image tensors."""

    root: Path | str = Path("samples/images")
    batch_size: int = 128
    num_workers: int = 0
    image_size: int = 32
    normalize: bool = True
    augment: bool = False
    pin_memory: bool = False

    def _build_transforms(self) -> transforms.Compose:
        ops = []
        if self.augment:
            ops.extend(
                [
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomCrop(self.image_size, padding=4),
                ]
            )
        if self.image_size != 32:
            ops.append(transforms.Resize((self.image_size, self.image_size)))
        ops.append(transforms.ToTensor())
        if self.normalize:
            ops.append(transforms.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD))
        return transforms.Compose(ops)

    def dataset(self) -> datasets.ImageFolder:
        root = _ensure_root(Path(self.root))
        return datasets.ImageFolder(root=str(root), transform=self._build_transforms())

    def dataloader(self, *, shuffle: bool = True, drop_last: bool = False) -> DataLoader:
        dataset = self.dataset()
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=drop_last,
        )


def tensorfy_media(
    media_path: Path | str,
    *,
    image_size: int = 32,
    normalize: bool = True,
    audio_sample_rate: int = DEFAULT_AUDIO_SAMPLE_RATE,
    audio_num_samples: int = DEFAULT_AUDIO_NUM_SAMPLES,
    mono: bool = True,
) -> MediaTensor:
    """Convert a media file into a multimodal tensor representation."""

    path = Path(media_path)
    suffix = path.suffix.lower()
    image_tensor = torch.zeros((3, image_size, image_size), dtype=torch.float32)
    audio_tensor = torch.zeros((1, audio_num_samples), dtype=torch.float32)
    has_image = False
    has_audio = False

    if suffix in IMAGE_EXTENSIONS:
        image_tensor = image_to_tensor(path, image_size=image_size, normalize=normalize)
        has_image = True
    elif suffix in AUDIO_EXTENSIONS:
        audio_tensor = audio_to_tensor(
            path, sample_rate=audio_sample_rate, num_samples=audio_num_samples, mono=mono
        )
        has_audio = True
    else:
        raise ValueError(f"Unhandled media extension '{suffix}' for file {path}")

    return MediaTensor(image=image_tensor, audio=audio_tensor, has_image=has_image, has_audio=has_audio)


if __name__ == "__main__":
    pipeline = ImageTensorPipeline()
    loader = pipeline.dataloader()
    images, labels = next(iter(loader))
    print(f"Loaded batch: {images.shape} tensor, dtype={images.dtype}, labels shape={labels.shape}")
