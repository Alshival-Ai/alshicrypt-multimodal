from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from training.models import DiscreteRGBUNet, compose_rgba, rgb_logits_to_uint8, rgb_uint8_to_float
from training.pokemon_pairs import RESAMPLE_MODES


def load_rgba_tensor(path: Path, image_size: int, resize_mode: str) -> torch.Tensor:
    image = Image.open(path).convert("RGBA").resize((image_size, image_size), RESAMPLE_MODES[resize_mode])
    data = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(data).permute(2, 0, 1).contiguous()


def tensor_to_rgba_image(tensor: torch.Tensor) -> Image.Image:
    array = tensor.detach().cpu().permute(1, 2, 0).numpy()
    array = np.clip(array, 0.0, 1.0)
    return Image.fromarray((array * 255.0).round().astype(np.uint8), mode="RGBA")


def main() -> None:
    parser = argparse.ArgumentParser(description="Restore distorted Pokemon images with a trained decoder.")
    parser.add_argument("--input-dir", default="pokemon_distorted")
    parser.add_argument("--output-dir", default="pokemon_restored")
    parser.add_argument("--checkpoint", default="models/paper_eval/decoder_best.pt")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location=args.device, weights_only=False)
    image_size = int(checkpoint.get("image_size", 120))
    resize_mode = checkpoint.get("resize_mode", "nearest")
    base_channels = int(checkpoint.get("base_channels", 48))

    device = torch.device(args.device)
    model = DiscreteRGBUNet(in_channels=4, base=base_channels).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = sorted(input_dir.glob("*.png"))
    if not paths:
        raise RuntimeError(f"No PNG files found in {input_dir}")

    with torch.no_grad():
        for path in paths:
            out_path = output_dir / path.name
            if out_path.exists() and not args.overwrite:
                continue

            x = load_rgba_tensor(path, image_size=image_size, resize_mode=resize_mode).unsqueeze(0).to(device)
            logits = model(x)
            rgb = rgb_uint8_to_float(rgb_logits_to_uint8(logits))
            pred = compose_rgba(rgb, x[:, 3:])
            tensor_to_rgba_image(pred[0]).save(out_path)
            print(f"saved {out_path}")


if __name__ == "__main__":
    main()
