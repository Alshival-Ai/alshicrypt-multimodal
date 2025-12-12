"""Invertible transformations for multimodal tensors."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence

import torch

from .data import DEFAULT_AUDIO_NUM_SAMPLES, MediaTensor


def _safe_rand_scale(size: int, min_scale: float = 0.5, max_scale: float = 2.0, device: Optional[torch.device] = None) -> torch.Tensor:
    """Generate random scaling factors bounded away from zero for stability."""

    scale = torch.empty(size, device=device).uniform_(min_scale, max_scale)
    signs = torch.randint(0, 2, (size,), device=device, dtype=torch.int8) * 2 - 1
    return scale * signs.float()


@dataclass
class TensorPermutationTransform:
    """Linear invertible tensor transform built from a permutation and element-wise scaling."""

    shape: Sequence[int]
    permutation: torch.Tensor
    scale: torch.Tensor

    @classmethod
    def random(
        cls,
        shape: Sequence[int],
        *,
        min_scale: float = 0.5,
        max_scale: float = 2.0,
        device: Optional[torch.device] = None,
    ) -> "TensorPermutationTransform":
        numel = int(math.prod(shape))
        permutation = torch.randperm(numel, device=device)
        scale = _safe_rand_scale(numel, min_scale=min_scale, max_scale=max_scale, device=device)
        return cls(shape=tuple(shape), permutation=permutation, scale=scale)

    def _reshape(self, tensor: torch.Tensor) -> torch.Tensor:
        if tuple(tensor.shape) != tuple(self.shape):
            raise ValueError(f"Expected tensor with shape {self.shape}, got {tuple(tensor.shape)}")
        return tensor.reshape(-1)

    def apply(self, tensor: torch.Tensor) -> torch.Tensor:
        flat = self._reshape(tensor)
        transformed = flat[self.permutation] * self.scale
        return transformed.reshape(self.shape)

    def invert(self, tensor: torch.Tensor) -> torch.Tensor:
        flat = self._reshape(tensor)
        inv_flat = torch.empty_like(flat)
        inv_flat[self.permutation] = flat / self.scale
        return inv_flat.reshape(self.shape)


@dataclass
class RandomMLP:
    """Lightweight two-layer MLP with random weights."""

    w1: torch.Tensor
    b1: torch.Tensor
    w2: torch.Tensor
    b2: torch.Tensor

    @classmethod
    def random(cls, in_features: int, out_features: int, hidden_features: Optional[int] = None) -> "RandomMLP":
        hidden = hidden_features or min(512, max(32, in_features, out_features))
        w1 = torch.randn(hidden, in_features) / math.sqrt(max(1, in_features))
        b1 = torch.zeros(hidden)
        w2 = torch.randn(out_features, hidden) / math.sqrt(max(1, hidden))
        b2 = torch.zeros(out_features)
        return cls(w1=w1, b1=b1, w2=w2, b2=b2)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.tanh(self.w1 @ x + self.b1)
        return self.w2 @ h + self.b2


@dataclass
class RealNVPFlow:
    """Two-stage affine coupling flow that introduces nonlinearity while remaining invertible."""

    size: int
    split: int
    scale1: Optional[RandomMLP]
    shift1: Optional[RandomMLP]
    scale2: Optional[RandomMLP]
    shift2: Optional[RandomMLP]
    scale_clip: float = 1.0

    @classmethod
    def random(cls, size: int, *, scale_clip: float = 1.0) -> "RealNVPFlow":
        if size < 2:
            return cls(size=size, split=0, scale1=None, shift1=None, scale2=None, shift2=None, scale_clip=scale_clip)
        split = size // 2
        dim_a = split
        dim_b = size - split
        scale1 = RandomMLP.random(dim_a, dim_b)
        shift1 = RandomMLP.random(dim_a, dim_b)
        scale2 = RandomMLP.random(dim_b, dim_a)
        shift2 = RandomMLP.random(dim_b, dim_a)
        return cls(size=size, split=split, scale1=scale1, shift1=shift1, scale2=scale2, shift2=shift2, scale_clip=scale_clip)

    def _enabled(self) -> bool:
        return all(layer is not None for layer in (self.scale1, self.shift1, self.scale2, self.shift2))

    def _forward_layer(self, cond: torch.Tensor, transformed: torch.Tensor, scale_net: RandomMLP, shift_net: RandomMLP) -> torch.Tensor:
        scale = torch.tanh(scale_net(cond)) * self.scale_clip
        shift = shift_net(cond)
        return transformed * torch.exp(scale) + shift

    def _inverse_layer(self, cond: torch.Tensor, transformed: torch.Tensor, scale_net: RandomMLP, shift_net: RandomMLP) -> torch.Tensor:
        scale = torch.tanh(scale_net(cond)) * self.scale_clip
        shift = shift_net(cond)
        return (transformed - shift) * torch.exp(-scale)

    def forward(self, flat: torch.Tensor) -> torch.Tensor:
        if not self._enabled():
            return flat
        a = flat[: self.split]
        b = flat[self.split :]
        b = self._forward_layer(a, b, self.scale1, self.shift1)
        a2 = b
        b2 = a
        b2 = self._forward_layer(a2, b2, self.scale2, self.shift2)
        return torch.cat([b2, a2])

    def inverse(self, flat: torch.Tensor) -> torch.Tensor:
        if not self._enabled():
            return flat
        b2 = flat[: self.split]
        a2 = flat[self.split :]
        a = self._inverse_layer(a2, b2, self.scale2, self.shift2)
        b = self._inverse_layer(a, a2, self.scale1, self.shift1)
        return torch.cat([a, b])


@dataclass
class InvertibleTensorTransform:
    """Composition of a linear transform and nonlinear coupling flow."""

    linear: TensorPermutationTransform
    flow: RealNVPFlow

    @classmethod
    def random(cls, shape: Sequence[int]) -> "InvertibleTensorTransform":
        linear = TensorPermutationTransform.random(shape)
        flow = RealNVPFlow.random(int(math.prod(shape)))
        return cls(linear=linear, flow=flow)

    @property
    def shape(self) -> Sequence[int]:
        return self.linear.shape

    def apply(self, tensor: torch.Tensor) -> torch.Tensor:
        tensor = self.linear.apply(tensor)
        flat = tensor.reshape(-1)
        flat = self.flow.forward(flat)
        return flat.reshape(self.shape)

    def invert(self, tensor: torch.Tensor) -> torch.Tensor:
        flat = tensor.reshape(-1)
        flat = self.flow.inverse(flat)
        tensor = flat.reshape(self.shape)
        return self.linear.invert(tensor)


@dataclass
class MediaTransform:
    """Applies independent invertible transforms to the image and audio tensors."""

    image_transform: Optional[InvertibleTensorTransform]
    audio_transform: Optional[InvertibleTensorTransform]

    @classmethod
    def random(
        cls,
        image_shape: Optional[Sequence[int]] = (3, 32, 32),
        audio_shape: Optional[Sequence[int]] = (1, DEFAULT_AUDIO_NUM_SAMPLES),
    ) -> "MediaTransform":
        image_tf = InvertibleTensorTransform.random(image_shape) if image_shape else None
        audio_tf = InvertibleTensorTransform.random(audio_shape) if audio_shape else None
        return cls(image_transform=image_tf, audio_transform=audio_tf)

    def apply(self, media: MediaTensor) -> MediaTensor:
        image = media.image
        audio = media.audio
        if media.has_image and self.image_transform is not None:
            image = self.image_transform.apply(media.image)
        if media.has_audio and self.audio_transform is not None:
            audio = self.audio_transform.apply(media.audio)
        return MediaTensor(image=image, audio=audio, has_image=media.has_image, has_audio=media.has_audio)

    def invert(self, media: MediaTensor) -> MediaTensor:
        image = media.image
        audio = media.audio
        if media.has_image and self.image_transform is not None:
            image = self.image_transform.invert(media.image)
        if media.has_audio and self.audio_transform is not None:
            audio = self.audio_transform.invert(media.audio)
        return MediaTensor(image=image, audio=audio, has_image=media.has_image, has_audio=media.has_audio)


def random_media_transform(
    image_shape: Optional[Sequence[int]] = (3, 32, 32),
    audio_shape: Optional[Sequence[int]] = (1, DEFAULT_AUDIO_NUM_SAMPLES),
) -> MediaTransform:
    """Convenience wrapper mirroring :meth:`MediaTransform.random`."""

    return MediaTransform.random(image_shape=image_shape, audio_shape=audio_shape)
