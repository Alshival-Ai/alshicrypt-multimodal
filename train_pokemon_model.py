from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from training.models import UNetLike
from training.pokemon_pairs import PokemonPairDataset, build_pairs


def evaluate_mae(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    total_abs = 0.0
    total_count = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            pred = model(x)
            total_abs += torch.abs(pred - y).sum().item()
            total_count += y.numel()
    return total_abs / max(total_count, 1)


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
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--target-mae", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=17)
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

    dataset = PokemonPairDataset(pairs, mode=args.stage, image_size=args.image_size)
    train_loader = make_loader(dataset, args.batch_size, True, device)
    eval_loader = make_loader(dataset, args.batch_size, False, device)

    model = UNetLike(in_channels=4, out_channels=4).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.L1Loss()
    scaler = GradScaler("cuda", enabled=device.type == "cuda")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    best_mae = float("inf")
    best_path = out_dir / f"{args.stage}_best.pt"
    last_path = out_dir / f"{args.stage}_last.pt"
    script_path = out_dir / f"{args.stage}_best.ts"

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        batches = 0
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with autocast("cuda", enabled=device.type == "cuda"):
                pred = model(x)
                loss = loss_fn(pred, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running += loss.item()
            batches += 1

        epoch_mae = evaluate_mae(model, eval_loader, device)
        avg_train = running / max(batches, 1)
        print(f"epoch={epoch:04d} train_l1={avg_train:.8f} dataset_mae={epoch_mae:.8f}")

        checkpoint = {
            "stage": args.stage,
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "dataset_mae": epoch_mae,
            "image_size": args.image_size,
        }
        torch.save(checkpoint, last_path)

        if epoch_mae < best_mae:
            best_mae = epoch_mae
            torch.save(checkpoint, best_path)
            model.eval()
            with torch.no_grad():
                example = torch.rand(1, 4, args.image_size, args.image_size, device=device)
                traced = torch.jit.trace(model, example)
            traced.save(str(script_path))

        if epoch_mae <= args.target_mae:
            print(f"Reached target MAE <= {args.target_mae} at epoch {epoch}.")
            break

    print(f"Best dataset MAE: {best_mae:.8f}")
    print(f"Saved: {best_path}")
    print(f"Saved: {script_path}")


if __name__ == "__main__":
    main()
