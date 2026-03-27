from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm


def beta_schedule(steps: int, beta_start: float, beta_end: float) -> np.ndarray:
    return np.linspace(beta_start, beta_end, steps, dtype=np.float32)


def distort_rgb_stochastic(x0: np.ndarray, betas: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.RandomState(seed)
    xt = x0.astype(np.float32).copy()
    for beta in betas:
        a = float(np.sqrt(1.0 - beta))
        s = float(np.sqrt(beta))
        eps = rng.normal(0.0, 1.0, size=xt.shape).astype(np.float32)
        xt = a * xt + s * eps
    return xt


def save_rgba(rgb: np.ndarray, alpha: np.ndarray | None, out_path: Path) -> None:
    rgb8 = (np.clip(rgb, 0.0, 1.0) * 255.0).round().astype(np.uint8)
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    if alpha is None:
        Image.fromarray(rgb8, mode="RGB").save(tmp_path, format="PNG")
        tmp_path.replace(out_path)
        return

    a8 = (np.clip(alpha, 0.0, 1.0) * 255.0).round().astype(np.uint8)
    rgba8 = np.dstack((rgb8, a8))
    Image.fromarray(rgba8, mode="RGBA").save(tmp_path, format="PNG")
    tmp_path.replace(out_path)


def process_one(in_path: Path, out_path: Path, betas: np.ndarray, base_seed: int) -> None:
    image = Image.open(in_path)
    has_alpha = "A" in image.getbands()

    if has_alpha:
        rgba = np.asarray(image.convert("RGBA"), dtype=np.float32) / 255.0
        rgb = rgba[..., :3]
        alpha = rgba[..., 3]
    else:
        rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
        alpha = None

    distorted_rgb = distort_rgb_stochastic(rgb, betas, base_seed)
    save_rgba(distorted_rgb, alpha, out_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stochastically distort all Pokemon PNG images."
    )
    parser.add_argument("--input-dir", default="pokemon", help="Root input directory")
    parser.add_argument(
        "--output-dir", default="pokemon_distorted", help="Output directory"
    )
    parser.add_argument("--steps", type=int, default=1000, help="Diffusion steps")
    parser.add_argument("--beta-start", type=float, default=5e-4, help="Beta start")
    parser.add_argument("--beta-end", type=float, default=1e-1, help="Beta end")
    parser.add_argument(
        "--seed", type=int, default=17, help="Base seed (per-file seeds are derived)"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recreate outputs even if target files already exist",
    )
    args = parser.parse_args()

    in_root = Path(args.input_dir)
    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    files = sorted(in_root.rglob("*.png"))
    if not files:
        raise FileNotFoundError(f"No PNG files found under: {in_root}")

    betas = beta_schedule(args.steps, args.beta_start, args.beta_end)

    processed = 0
    skipped = 0
    for in_file in tqdm(files, desc="Distorting Pokemon", unit="img"):
        out_file = out_root / in_file.name
        if out_file.exists() and not args.overwrite:
            skipped += 1
            continue
        process_one(in_file, out_file, betas, args.seed)
        processed += 1

    print(
        f"Processed {processed} file(s), skipped {skipped} existing file(s) -> {out_root}"
    )


if __name__ == "__main__":
    main()
