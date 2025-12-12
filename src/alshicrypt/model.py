"""Learnable parameterizations of the media transformation."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class ModelConfig:
    image_shape: Sequence[int] = (3, 32, 32)
    audio_shape: Optional[Sequence[int]] = None
    audio_num_samples: int = 16000
    latent_dim: int = 2048
    num_layers: int = 3

    def __post_init__(self) -> None:
        if self.audio_shape is None:
            self.audio_shape = (1, self.audio_num_samples)


class ModalityMLP(nn.Module):
    """Fully-connected network that maps tensors to tensors of the same shape."""

    def __init__(self, shape: Sequence[int], latent_dim: int, num_layers: int) -> None:
        super().__init__()
        self.shape = tuple(shape)
        self.input_dim = int(math.prod(self.shape))
        layers = []
        in_features = self.input_dim
        for _ in range(max(1, num_layers - 1)):
            layers.append(nn.Linear(in_features, latent_dim))
            layers.append(nn.GELU())
            in_features = latent_dim
        layers.append(nn.Linear(in_features, self.input_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]
        flat = x.reshape(batch, self.input_dim)
        transformed = self.net(flat)
        return transformed.reshape(batch, *self.shape)


class MediaTransformer(nn.Module):
    """Applies learnable tensor-to-tensor mappings for image/audio modalities."""

    def __init__(self, config: ModelConfig = ModelConfig()) -> None:
        super().__init__()
        self.config = config
        self.image_net = ModalityMLP(config.image_shape, config.latent_dim, config.num_layers)
        self.audio_net = ModalityMLP(config.audio_shape, config.latent_dim, config.num_layers)

    def forward(
        self,
        image: Optional[torch.Tensor] = None,
        audio: Optional[torch.Tensor] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        image_out = self.image_net(image) if image is not None else None
        audio_out = self.audio_net(audio) if audio is not None else None
        return image_out, audio_out


AlshiCryptAutoencoder = MediaTransformer  # Backwards compatibility alias.


def reconstruction_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(pred, target)


def build_model(config: Optional[ModelConfig] = None) -> MediaTransformer:
    return MediaTransformer(config or ModelConfig())
