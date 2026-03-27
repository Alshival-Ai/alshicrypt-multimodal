from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


RESAMPLE_MODES = {
    "nearest": Image.Resampling.NEAREST,
    "bilinear": Image.Resampling.BILINEAR,
}


@dataclass(frozen=True)
class PairItem:
    name: str
    original_path: Path
    encoded_path: Path


def build_pairs(original_root: Path, encoded_root: Path) -> list[PairItem]:
    originals = {p.stem: p for p in original_root.rglob("*.png")}
    encoded = {p.stem: p for p in encoded_root.glob("*.png")}
    names = sorted(set(originals) & set(encoded))
    return [PairItem(n, originals[n], encoded[n]) for n in names]


def _load_rgba_tensor(path: Path, image_size: int, resize_mode: str) -> torch.Tensor:
    image = Image.open(path).convert("RGBA").resize((image_size, image_size), RESAMPLE_MODES[resize_mode])
    data = np.asarray(image, dtype=np.uint8)
    float_data = torch.from_numpy(data.astype(np.float32) / 255.0).permute(2, 0, 1).contiguous()
    return float_data


class PokemonPairDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        items: list[PairItem],
        mode: Literal["encoder", "decoder"],
        image_size: int = 120,
        resize_mode: Literal["nearest", "bilinear"] = "nearest",
    ) -> None:
        self.items = items
        self.mode = mode
        self.image_size = image_size
        self.resize_mode = resize_mode

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        item = self.items[index]
        original = _load_rgba_tensor(item.original_path, self.image_size, self.resize_mode)
        encoded = _load_rgba_tensor(item.encoded_path, self.image_size, self.resize_mode)

        if self.mode == "encoder":
            source = original
            target = encoded
        else:
            source = encoded
            target = original

        target_rgb = (target[:3] * 255.0).round().to(torch.long)
        source_alpha = source[3:].clone()
        return source, target_rgb, source_alpha, target
