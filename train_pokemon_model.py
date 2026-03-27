from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from training.models import DiscreteRGBUNet, compose_rgba, rgb_logits_to_uint8, rgb_uint8_to_float
from training.pokemon_pairs import PokemonPairDataset, build_pairs


def logits_to_rgba(logits: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    rgb = rgb_uint8_to_float(rgb_logits_to_uint8(logits))
    return compose_rgba(rgb, alpha)


def evaluate_mae(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, float]:
    model.eval()
    total_abs_all = 0.0
    total_abs_rgb = 0.0
    total_count_all = 0
    total_count_rgb = 0
    with torch.no_grad():
        for x, y_rgb, source_alpha, target in loader:
            x = x.to(device, non_blocking=True)
            y_rgb = y_rgb.to(device, non_blocking=True)
            source_alpha = source_alpha.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            logits = model(x)
            pred = logits_to_rgba(logits, source_alpha)

            diff = pred - target
            total_abs_all += diff.abs().sum().item()
            total_abs_rgb += diff[:, :3].abs().sum().item()
            total_count_all += target.numel()
            total_count_rgb += target[:, :3].numel()
    return total_abs_all / max(total_count_all, 1), total_abs_rgb / max(total_count_rgb, 1)


def categorical_rgb_loss(logits: torch.Tensor, target_rgb: torch.Tensor) -> torch.Tensor:
    losses = []
    for channel in range(3):
        losses.append(nn.functional.cross_entropy(logits[:, channel], target_rgb[:, channel]))
    return sum(losses) / len(losses)


def make_loader(dataset: PokemonPairDataset, batch_size: int, shuffle: bool, device: torch.device) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=4,
        pin_memory=device.type == "cuda",
        persistent_workers=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Pokemon encoder or decoder model.")
    parser.add_argument("--stage", choices=["encoder", "decoder"], required=True)
    parser.add_argument("--original-root", default="pokemon")
    parser.add_argument("--encoded-root", default="pokemon_distorted")
    parser.add_argument("--out-dir", default="models")
    parser.add_argument("--image-size", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--target-mae", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--base-channels", type=int, default=64)
    parser.add_argument("--resize-mode", choices=["nearest", "bilinear"], default="nearest")
    parser.add_argument("--resume", default="", help="Optional checkpoint path to resume from.")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("WARNING: CUDA not found. Training will run on CPU.")

    pairs = build_pairs(Path(args.original_root), Path(args.encoded_root))
    if not pairs:
        raise RuntimeError("No pairs found. Run distortion script first.")

    dataset = PokemonPairDataset(
        pairs,
        mode=args.stage,
        image_size=args.image_size,
        resize_mode=args.resize_mode,
    )
    train_loader = make_loader(dataset, args.batch_size, True, device)
    eval_loader = make_loader(dataset, args.batch_size, False, device)

    model = DiscreteRGBUNet(in_channels=4, base=args.base_channels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = GradScaler("cuda", enabled=device.type == "cuda")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    best_mae = float("inf")
    best_path = out_dir / f"{args.stage}_best.pt"
    last_path = out_dir / f"{args.stage}_last.pt"
    script_path = out_dir / f"{args.stage}_best.ts"
    start_epoch = 1

    if best_path.exists():
        best_checkpoint = torch.load(best_path, map_location="cpu", weights_only=False)
        best_mae = float(best_checkpoint.get("dataset_mae", best_mae))

    if args.resume:
        resume_path = Path(args.resume)
        checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = int(checkpoint["epoch"]) + 1
        print(f"Resuming from {resume_path} at epoch {checkpoint['epoch']}.")

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        running = 0.0
        batches = 0
        for x, y_rgb, _source_alpha, _target in train_loader:
            x = x.to(device, non_blocking=True)
            y_rgb = y_rgb.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with autocast("cuda", enabled=device.type == "cuda"):
                logits = model(x)
                loss = categorical_rgb_loss(logits, y_rgb)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running += loss.item()
            batches += 1

        epoch_mae_all, epoch_mae_rgb = evaluate_mae(model, eval_loader, device)
        avg_train = running / max(batches, 1)
        print(
            f"epoch={epoch:04d} train_ce={avg_train:.8f} dataset_mae={epoch_mae_all:.8f} dataset_rgb_mae={epoch_mae_rgb:.8f}"
        )

        checkpoint = {
            "stage": args.stage,
            "epoch": epoch,
            "model_family": "discrete_rgb",
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "dataset_mae": epoch_mae_all,
            "dataset_rgb_mae": epoch_mae_rgb,
            "image_size": args.image_size,
            "base_channels": args.base_channels,
            "resize_mode": args.resize_mode,
            "preserve_alpha": True,
            "rgb_bins": 256,
        }
        torch.save(checkpoint, last_path)

        if epoch_mae_all < best_mae:
            best_mae = epoch_mae_all
            torch.save(checkpoint, best_path)
            model.eval()
            with torch.no_grad():
                example = torch.rand(1, 4, args.image_size, args.image_size, device=device)
                traced = torch.jit.trace(model, example)
            traced.save(str(script_path))

        if epoch_mae_all <= args.target_mae:
            print(f"Reached target MAE <= {args.target_mae} at epoch {epoch}.")
            break

    print(f"Best dataset MAE: {best_mae:.8f}")
    print(f"Saved: {best_path}")
    print(f"Saved: {script_path}")


if __name__ == "__main__":
    main()
