"""Alshicrypt: utilities for tensor-based encryption experiments."""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Union

from .data import (
    AUDIO_EXTENSIONS,
    DEFAULT_AUDIO_NUM_SAMPLES,
    DEFAULT_AUDIO_SAMPLE_RATE,
    IMAGE_EXTENSIONS,
    ImageTensorPipeline,
    MediaTensor,
    audio_to_tensor,
    image_to_tensor,
    tensorfy_media,
)
from .transforms import (
    MediaTransform,
    TensorPermutationTransform,
    random_media_transform,
)
from .model import (
    ModelConfig,
    AlshiCryptAutoencoder,
    build_model,
    reconstruction_loss,
)
from .generate import generate

__all__ = [
    "AUDIO_EXTENSIONS",
    "DEFAULT_AUDIO_NUM_SAMPLES",
    "DEFAULT_AUDIO_SAMPLE_RATE",
    "IMAGE_EXTENSIONS",
    "ImageTensorPipeline",
    "MediaTensor",
    "MediaTransform",
    "TensorPermutationTransform",
    "audio_to_tensor",
    "image_to_tensor",
    "random_media_transform",
    "ModelConfig",
    "AlshiCryptAutoencoder",
    "build_model",
    "reconstruction_loss",
    "generate",
    "tensorfy_media",
    "tensorfy",
]

if TYPE_CHECKING:
    import torch


def tensorfy(
    media_path: Union[str, Path],
    *,
    image_size: int = 32,
    normalize: bool = True,
    audio_sample_rate: int = DEFAULT_AUDIO_SAMPLE_RATE,
    audio_num_samples: int = DEFAULT_AUDIO_NUM_SAMPLES,
    mono: bool = True,
) -> MediaTensor:
    """Public convenience wrapper to load media into a multimodal tensor."""

    return tensorfy_media(
        media_path,
        image_size=image_size,
        normalize=normalize,
        audio_sample_rate=audio_sample_rate,
        audio_num_samples=audio_num_samples,
        mono=mono,
    )
