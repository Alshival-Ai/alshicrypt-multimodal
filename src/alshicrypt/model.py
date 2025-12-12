"""Learnable parameterizations of the media transformation."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import torch
from torch import nn
from torch.nn import functional as F

from .data import DEFAULT_AUDIO_NUM_SAMPLES


@dataclass
class ModelConfig:
    image_shape: Sequence[int] = (3, 32, 32)
    audio_shape: Optional[Sequence[int]] = None
    audio_num_samples: int = DEFAULT_AUDIO_NUM_SAMPLES
    latent_dim: int = 2048
    image_latent_dim: Optional[int] = None
    audio_latent_dim: Optional[int] = None
    num_layers: int = 4
    residual: bool = True
    dropout: float = 0.02
    use_layernorm: bool = True
    expansion: int = 2

    def __post_init__(self) -> None:
        if self.audio_shape is None:
            self.audio_shape = (1, self.audio_num_samples)
        if self.image_latent_dim is None:
            self.image_latent_dim = self.latent_dim
        if self.audio_latent_dim is None:
            self.audio_latent_dim = self.latent_dim
        image_dim = int(math.prod(self.image_shape))
        audio_dim = int(math.prod(self.audio_shape))
        self.image_latent_dim = max(self.image_latent_dim, image_dim)
        self.audio_latent_dim = max(self.audio_latent_dim, audio_dim)


class ResidualMLPBlock(nn.Module):
    """Residual feed-forward block with optional normalization and dropout."""

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        *,
        dropout: float = 0.0,
        use_layernorm: bool = True,
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim) if use_layernorm else nn.Identity()
        self.linear1 = nn.Linear(dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, dim)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.norm(x)
        y = self.linear1(y)
        y = F.gelu(y)
        y = self.dropout(y)
        y = self.linear2(y)
        return y


class ModalityMLP(nn.Module):
    """Fully-connected network that preserves tensor dimensionality."""

    def __init__(
        self,
        shape: Sequence[int],
        latent_dim: int,
        num_layers: int,
        *,
        residual: bool = True,
        dropout: float = 0.0,
        use_layernorm: bool = True,
        expansion: int = 1,
    ) -> None:
        super().__init__()
        self.shape = tuple(shape)
        self.input_dim = int(math.prod(self.shape))
        self.residual = residual

        hidden_dim = max(latent_dim, self.input_dim) * max(1, expansion)
        self.direct = nn.Linear(self.input_dim, self.input_dim)
        self.blocks = nn.ModuleList(
            ResidualMLPBlock(
                self.input_dim,
                hidden_dim,
                dropout=dropout,
                use_layernorm=use_layernorm,
            )
            for _ in range(max(0, num_layers))
        )
        self._init_direct()

    def _init_direct(self) -> None:
        if self.direct.weight.shape[0] == self.direct.weight.shape[1]:
            nn.init.eye_(self.direct.weight)
        else:
            nn.init.xavier_uniform_(self.direct.weight)
        nn.init.zeros_(self.direct.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]
        flat = x.reshape(batch, self.input_dim)
        out = self.direct(flat)
        for block in self.blocks:
            update = block(out)
            out = out + update if self.residual else update
        return out.reshape(batch, *self.shape)


class MediaTransformer(nn.Module):
    """Applies learnable tensor-to-tensor mappings for image/audio modalities."""

    def __init__(self, config: ModelConfig = ModelConfig()) -> None:
        super().__init__()
        self.config = config
        self.image_net = ModalityMLP(
            config.image_shape,
            config.image_latent_dim,
            config.num_layers,
            residual=config.residual,
            dropout=config.dropout,
            use_layernorm=config.use_layernorm,
            expansion=config.expansion,
        )
        self.audio_net = ModalityMLP(
            config.audio_shape,
            config.audio_latent_dim,
            config.num_layers,
            residual=config.residual,
            dropout=config.dropout,
            use_layernorm=config.use_layernorm,
            expansion=config.expansion,
        )

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
