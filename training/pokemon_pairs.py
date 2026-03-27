from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


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


def _load_rgba_tensor(path: Path, image_size: int) -> torch.Tensor:
    image = Image.open(path).convert("RGBA").resize((image_size, image_size), Image.BILINEAR)
    data = np.asarray(image, dtype=np.float32) / 255.0  # HWC [0,1]
    data = torch.from_numpy(data).permute(2, 0, 1).contiguous()
    return data


class PokemonPairDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        items: list[PairItem],
        mode: Literal["encoder", "decoder"],
        image_size: int = 128,
    ) -> None:
        self.items = items
        self.mode = mode
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        item = self.items[index]
        original = _load_rgba_tensor(item.original_path, self.image_size)
        encoded = _load_rgba_tensor(item.encoded_path, self.image_size)
        if self.mode == "encoder":
            return original, encoded
        return encoded, original
