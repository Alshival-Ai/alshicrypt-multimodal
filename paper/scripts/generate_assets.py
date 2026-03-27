from __future__ import annotations

import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[2]
PAPER_ROOT = REPO_ROOT / "paper"
FIGURE_DIR = PAPER_ROOT / "figures"
GENERATED_DIR = PAPER_ROOT / "generated"
MODEL_DIR = REPO_ROOT / "models" / "paper_eval"

sys.path.insert(0, str(REPO_ROOT))

from training.models import DiscreteRGBUNet, compose_rgba, rgb_logits_to_uint8, rgb_uint8_to_float
from training.pokemon_pairs import PokemonPairDataset, build_pairs


@dataclass(frozen=True)
class CurvePoint:
    epoch: int
    train_error: float
    dataset_mae: float


def beta_schedule(steps: int, beta_start: float, beta_end: float) -> np.ndarray:
    return np.linspace(beta_start, beta_end, steps, dtype=np.float64)


def parse_log(path: Path) -> list[CurvePoint]:
    pattern = re.compile(
        r"epoch=(?P<epoch>\d+)\s+train_(?:l1|ce)=(?P<train>[0-9.]+)\s+dataset_mae=(?P<mae>[0-9.]+)"
    )
    raw = path.read_bytes()
    for encoding in ("utf-8", "utf-8-sig", "utf-16", "utf-16-le"):
        try:
            text = raw.decode(encoding)
            break
        except UnicodeDecodeError:
            text = ""
    if not text:
        raise UnicodeDecodeError("decode", raw, 0, 1, f"Unsupported log encoding for {path}")
    points: list[CurvePoint] = []
    for line in text.splitlines():
        match = pattern.search(line)
        if not match:
            continue
        points.append(
            CurvePoint(
                epoch=int(match.group("epoch")),
                train_error=float(match.group("train")),
                dataset_mae=float(match.group("mae")),
            )
        )
    return points


def composite_rgba_to_rgb(array: np.ndarray) -> np.ndarray:
    rgb = np.clip(array[..., :3], 0.0, 1.0)
    alpha = np.clip(array[..., 3:4], 0.0, 1.0)
    white = np.ones_like(rgb)
    return rgb * alpha + white * (1.0 - alpha)


def tensor_to_display_image(tensor: torch.Tensor) -> np.ndarray:
    array = tensor.detach().cpu().permute(1, 2, 0).numpy()
    return composite_rgba_to_rgb(array)


def load_checkpoint(stage: str, device: torch.device) -> dict[str, object]:
    return torch.load(MODEL_DIR / f"{stage}_best.pt", map_location=device, weights_only=False)


def load_model(stage: str, device: torch.device) -> tuple[DiscreteRGBUNet, dict[str, object]]:
    checkpoint = load_checkpoint(stage, device)
    model = DiscreteRGBUNet(
        in_channels=4,
        base=int(checkpoint.get("base_channels", 64)),
        rgb_bins=int(checkpoint.get("rgb_bins", 256)),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, checkpoint


def make_loader(items, mode: str, image_size: int, resize_mode: str, batch_size: int = 16) -> DataLoader:
    dataset = PokemonPairDataset(items, mode=mode, image_size=image_size, resize_mode=resize_mode)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)


def predict_rgba(model: DiscreteRGBUNet, x: torch.Tensor) -> torch.Tensor:
    logits = model(x)
    rgb = rgb_uint8_to_float(rgb_logits_to_uint8(logits))
    return compose_rgba(rgb, x[:, 3:])


def evaluate_stage(
    stage: str,
    model: DiscreteRGBUNet,
    items,
    image_size: int,
    resize_mode: str,
    device: torch.device,
) -> dict[str, float]:
    loader = make_loader(items, stage, image_size=image_size, resize_mode=resize_mode)

    total_abs = 0.0
    total_abs_rgb = 0.0
    total_abs_alpha = 0.0
    total_mse = 0.0
    total_mse_rgb = 0.0
    baseline_abs = 0.0
    baseline_abs_rgb = 0.0
    baseline_mse = 0.0
    baseline_mse_rgb = 0.0
    total_count = 0
    rgb_count = 0
    alpha_count = 0

    with torch.no_grad():
        for x, _y_rgb, _source_alpha, target in loader:
            x = x.to(device)
            target = target.to(device)
            pred = predict_rgba(model, x)

            diff = pred - target
            base_diff = x - target

            total_abs += diff.abs().sum().item()
            total_abs_rgb += diff[:, :3].abs().sum().item()
            total_abs_alpha += diff[:, 3:].abs().sum().item()
            total_mse += diff.square().sum().item()
            total_mse_rgb += diff[:, :3].square().sum().item()

            baseline_abs += base_diff.abs().sum().item()
            baseline_abs_rgb += base_diff[:, :3].abs().sum().item()
            baseline_mse += base_diff.square().sum().item()
            baseline_mse_rgb += base_diff[:, :3].square().sum().item()

            total_count += target.numel()
            rgb_count += target[:, :3].numel()
            alpha_count += target[:, 3:].numel()

    mae_all = total_abs / total_count
    mae_rgb = total_abs_rgb / rgb_count
    mae_alpha = total_abs_alpha / alpha_count
    mse_all = total_mse / total_count
    mse_rgb = total_mse_rgb / rgb_count

    baseline_mae_all = baseline_abs / total_count
    baseline_mae_rgb = baseline_abs_rgb / rgb_count
    baseline_mse_all = baseline_mse / total_count
    baseline_mse_rgb = baseline_mse_rgb / rgb_count

    return {
        "mae_all": mae_all,
        "mae_rgb": mae_rgb,
        "mae_alpha": mae_alpha,
        "psnr_all": 10.0 * math.log10(1.0 / max(mse_all, 1e-12)),
        "psnr_rgb": 10.0 * math.log10(1.0 / max(mse_rgb, 1e-12)),
        "baseline_mae_all": baseline_mae_all,
        "baseline_mae_rgb": baseline_mae_rgb,
        "baseline_psnr_all": 10.0 * math.log10(1.0 / max(baseline_mse_all, 1e-12)),
        "baseline_psnr_rgb": 10.0 * math.log10(1.0 / max(baseline_mse_rgb, 1e-12)),
    }


def collect_dataset_stats(items) -> dict[str, float]:
    rgb_value_count = 0
    encoded_saturated = 0
    alpha_abs_sum = 0.0
    alpha_count = 0
    sum_x = 0.0
    sum_y = 0.0
    sum_x2 = 0.0
    sum_y2 = 0.0
    sum_xy = 0.0
    pixel_count = 0

    for item in items:
        original = np.asarray(Image.open(item.original_path).convert("RGBA"), dtype=np.uint8)
        encoded = np.asarray(Image.open(item.encoded_path).convert("RGBA"), dtype=np.uint8)

        original_rgb = original[..., :3].astype(np.float64).reshape(-1)
        encoded_rgb = encoded[..., :3].astype(np.float64).reshape(-1)
        original_alpha = original[..., 3].astype(np.float64).reshape(-1)
        encoded_alpha = encoded[..., 3].astype(np.float64).reshape(-1)

        sum_x += original_rgb.sum()
        sum_y += encoded_rgb.sum()
        sum_x2 += np.square(original_rgb).sum()
        sum_y2 += np.square(encoded_rgb).sum()
        sum_xy += (original_rgb * encoded_rgb).sum()
        pixel_count += original_rgb.size

        encoded_saturated += np.count_nonzero((encoded_rgb == 0.0) | (encoded_rgb == 255.0))
        rgb_value_count += encoded_rgb.size

        alpha_abs_sum += np.abs(original_alpha - encoded_alpha).sum()
        alpha_count += original_alpha.size

    numerator = pixel_count * sum_xy - sum_x * sum_y
    denominator = math.sqrt(
        max(pixel_count * sum_x2 - sum_x * sum_x, 1e-12)
        * max(pixel_count * sum_y2 - sum_y * sum_y, 1e-12)
    )
    rgb_correlation = numerator / denominator if denominator else 0.0

    return {
        "encoded_saturation_rate": encoded_saturated / max(rgb_value_count, 1),
        "rgb_correlation": rgb_correlation,
        "alpha_channel_mae_raw": (alpha_abs_sum / max(alpha_count, 1)) / 255.0,
    }


def generate_forward_process_figure(betas: np.ndarray, alpha: np.ndarray) -> None:
    steps = np.arange(1, betas.size + 1)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))

    axes[0].plot(steps, betas, color="#0b6e4f", linewidth=2.0)
    axes[0].set_title("Noise schedule")
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel(r"$\beta_t$")
    axes[0].grid(alpha=0.25)

    axes[1].plot(steps, np.log10(alpha), color="#a23b72", linewidth=2.0)
    axes[1].set_title("Log signal attenuation")
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel(r"$\log_{10}\alpha_t$")
    axes[1].grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "forward_process.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def generate_training_curve_figure(encoder_curve: list[CurvePoint], decoder_curve: list[CurvePoint]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))

    for ax, curve, title, color in [
        (axes[0], encoder_curve, "Encoder training", "#1d4e89"),
        (axes[1], decoder_curve, "Decoder training", "#7a1f1f"),
    ]:
        epochs = [point.epoch for point in curve]
        train_error = [point.train_error for point in curve]
        dataset_mae = [point.dataset_mae for point in curve]
        ax.plot(epochs, train_error, label="train objective", linewidth=2.0, color=color, alpha=0.75)
        ax.plot(epochs, dataset_mae, label="dataset MAE", linewidth=2.0, color="#111111")
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Error")
        ax.grid(alpha=0.25)
        ax.legend(frameon=False)

    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "training_curves.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def generate_reconstruction_figure(
    items,
    encoder_model: DiscreteRGBUNet,
    decoder_model: DiscreteRGBUNet,
    image_size: int,
    resize_mode: str,
    device: torch.device,
) -> list[str]:
    preferred = ["bulbasaur", "charizard", "pikachu", "mewtwo"]
    items_by_name = {item.name: item for item in items}
    selected_names = [name for name in preferred if name in items_by_name]
    if len(selected_names) < 4:
        selected_names.extend(item.name for item in items if item.name not in selected_names)
    selected_names = selected_names[:4]

    fig, axes = plt.subplots(len(selected_names), 4, figsize=(10, 2.5 * len(selected_names)))
    if len(selected_names) == 1:
        axes = np.expand_dims(axes, axis=0)

    column_titles = ["Original", "Encrypted target", "Encrypter output", "Decrypter reconstruction"]
    for col, title in enumerate(column_titles):
        axes[0, col].set_title(title)

    with torch.no_grad():
        for row, name in enumerate(selected_names):
            item = items_by_name[name]
            original = PokemonPairDataset([item], mode="encoder", image_size=image_size, resize_mode=resize_mode)[0][0].unsqueeze(0).to(device)
            encoded = PokemonPairDataset([item], mode="decoder", image_size=image_size, resize_mode=resize_mode)[0][0].unsqueeze(0).to(device)

            pred_encoded = predict_rgba(encoder_model, original)
            pred_decoded = predict_rgba(decoder_model, encoded)

            row_images = [
                tensor_to_display_image(original[0]),
                tensor_to_display_image(encoded[0]),
                tensor_to_display_image(pred_encoded[0]),
                tensor_to_display_image(pred_decoded[0]),
            ]

            for col, image in enumerate(row_images):
                axes[row, col].imshow(image)
                axes[row, col].axis("off")
            axes[row, 0].set_ylabel(name, rotation=90, fontsize=10)

    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "reconstruction_grid.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    return selected_names


def generate_mew_reconstruction_figure(
    items,
    decoder_model: DiscreteRGBUNet,
    image_size: int,
    resize_mode: str,
    device: torch.device,
) -> None:
    items_by_name = {item.name: item for item in items}
    if "mew" not in items_by_name:
        raise RuntimeError("Expected mew in dataset.")
    encoded = PokemonPairDataset([items_by_name["mew"]], mode="decoder", image_size=image_size, resize_mode=resize_mode)[0][0].unsqueeze(0).to(device)
    with torch.no_grad():
        pred = predict_rgba(decoder_model, encoded)
    array = (pred[0].detach().cpu().permute(1, 2, 0).numpy() * 255.0).round().clip(0, 255).astype(np.uint8)
    Image.fromarray(array, mode="RGBA").save(FIGURE_DIR / "mew_decrypter_reconstruction.png")


def generate_pixel_statistics_figure(
    items,
    image_size: int,
    resize_mode: str,
    encoder_metrics: dict[str, float],
    decoder_metrics: dict[str, float],
) -> None:
    loader = make_loader(items, mode="encoder", image_size=image_size, resize_mode=resize_mode)
    hist_original = np.zeros(256, dtype=np.int64)
    hist_encoded = np.zeros(256, dtype=np.int64)

    for original, _encoded_rgb_target, _source_alpha, encoded_target in loader:
        original_rgb = (original[:, :3].numpy() * 255.0).round().astype(np.uint8).reshape(-1)
        encoded_rgb = (encoded_target[:, :3].numpy() * 255.0).round().astype(np.uint8).reshape(-1)
        hist_original += np.bincount(original_rgb, minlength=256)
        hist_encoded += np.bincount(encoded_rgb, minlength=256)

    centers = np.arange(256)

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    axes[0].plot(centers, hist_original / hist_original.sum(), label="Original RGB", linewidth=2.0, color="#1d4e89")
    axes[0].plot(centers, hist_encoded / hist_encoded.sum(), label="Encoded RGB", linewidth=2.0, color="#a23b72")
    axes[0].set_title("Discrete RGB value distribution")
    axes[0].set_xlabel("8-bit RGB value")
    axes[0].set_ylabel("Density")
    axes[0].grid(alpha=0.25)
    axes[0].legend(frameon=False)

    labels = ["Baseline", "Encoder", "Decoder"]
    mae_full = [
        encoder_metrics["baseline_mae_all"],
        encoder_metrics["mae_all"],
        decoder_metrics["mae_all"],
    ]
    mae_rgb = [
        encoder_metrics["baseline_mae_rgb"],
        encoder_metrics["mae_rgb"],
        decoder_metrics["mae_rgb"],
    ]
    x = np.arange(len(labels))
    width = 0.34
    axes[1].bar(x - width / 2, mae_full, width=width, label="All-channel MAE", color="#0b6e4f")
    axes[1].bar(x + width / 2, mae_rgb, width=width, label="RGB-only MAE", color="#d17b0f")
    axes[1].set_xticks(x, labels)
    axes[1].set_title("Error reduction against direct baseline")
    axes[1].set_ylabel("MAE")
    axes[1].grid(alpha=0.25, axis="y")
    axes[1].legend(frameon=False)

    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "pixel_statistics.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def latex_escape(text: str) -> str:
    return text.replace("_", r"\_")


def write_metric_macros(metrics: dict[str, object]) -> None:
    lines = [
        rf"\newcommand{{\NumPairs}}{{{metrics['num_pairs']}}}",
        rf"\newcommand{{\ImageSize}}{{{metrics['image_size']}}}",
        rf"\newcommand{{\BetaStart}}{{{metrics['beta_start']:.4g}}}",
        rf"\newcommand{{\BetaEnd}}{{{metrics['beta_end']:.4g}}}",
        rf"\newcommand{{\NumSteps}}{{{metrics['num_steps']}}}",
        rf"\newcommand{{\FinalAlpha}}{{{metrics['final_alpha']:.3e}}}",
        rf"\newcommand{{\FinalAlphaSq}}{{{metrics['final_alpha_sq']:.3e}}}",
        rf"\newcommand{{\EncodedSaturationPct}}{{{metrics['encoded_saturation_rate'] * 100.0:.1f}}}",
        rf"\newcommand{{\EncodedRGBCorrelation}}{{{metrics['rgb_correlation']:.4f}}}",
        rf"\newcommand{{\RawAlphaMAE}}{{{metrics['alpha_channel_mae_raw']:.6f}}}",
        rf"\newcommand{{\ModelParamCount}}{{{metrics['model_parameters']:,}}}",
        rf"\newcommand{{\EncoderDatasetMAE}}{{{metrics['encoder_mae_all']:.8f}}}",
        rf"\newcommand{{\DecoderDatasetMAE}}{{{metrics['decoder_mae_all']:.8f}}}",
        rf"\newcommand{{\EncoderRGBMAE}}{{{metrics['encoder_mae_rgb']:.8f}}}",
        rf"\newcommand{{\DecoderRGBMAE}}{{{metrics['decoder_mae_rgb']:.8f}}}",
        rf"\newcommand{{\EncoderPSNR}}{{{metrics['encoder_psnr_rgb']:.2f}}}",
        rf"\newcommand{{\DecoderPSNR}}{{{metrics['decoder_psnr_rgb']:.2f}}}",
        rf"\newcommand{{\BaselineMAE}}{{{metrics['baseline_mae_all']:.8f}}}",
        rf"\newcommand{{\BaselineRGBMAE}}{{{metrics['baseline_mae_rgb']:.8f}}}",
        rf"\newcommand{{\BaselinePSNR}}{{{metrics['baseline_psnr_rgb']:.2f}}}",
        rf"\newcommand{{\EncoderExampleA}}{{{latex_escape(metrics['example_names'][0])}}}",
        rf"\newcommand{{\EncoderExampleB}}{{{latex_escape(metrics['example_names'][1])}}}",
        rf"\newcommand{{\EncoderExampleC}}{{{latex_escape(metrics['example_names'][2])}}}",
        rf"\newcommand{{\EncoderExampleD}}{{{latex_escape(metrics['example_names'][3])}}}",
    ]
    (GENERATED_DIR / "metrics.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(17)
    np.random.seed(17)

    beta_start = 5e-4
    beta_end = 1e-1
    num_steps = 1000

    betas = beta_schedule(num_steps, beta_start, beta_end)
    alpha = np.cumprod(np.sqrt(1.0 - betas))

    items = build_pairs(REPO_ROOT / "pokemon", REPO_ROOT / "pokemon_distorted")
    if not items:
        raise RuntimeError("No training pairs found. Generate the dataset first.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    encoder_curve = parse_log(MODEL_DIR / "encoder.log")
    decoder_curve = parse_log(MODEL_DIR / "decoder.log")
    encoder_model, encoder_checkpoint = load_model("encoder", device=device)
    decoder_model, decoder_checkpoint = load_model("decoder", device=device)

    image_size = int(encoder_checkpoint.get("image_size", 120))
    resize_mode = str(encoder_checkpoint.get("resize_mode", "nearest"))

    encoder_metrics = evaluate_stage("encoder", encoder_model, items, image_size=image_size, resize_mode=resize_mode, device=device)
    decoder_metrics = evaluate_stage("decoder", decoder_model, items, image_size=image_size, resize_mode=resize_mode, device=device)
    dataset_stats = collect_dataset_stats(items)

    generate_forward_process_figure(betas, alpha)
    generate_training_curve_figure(encoder_curve, decoder_curve)
    example_names = generate_reconstruction_figure(
        items,
        encoder_model,
        decoder_model,
        image_size=image_size,
        resize_mode=resize_mode,
        device=device,
    )
    generate_mew_reconstruction_figure(
        items,
        decoder_model,
        image_size=image_size,
        resize_mode=resize_mode,
        device=device,
    )
    generate_pixel_statistics_figure(
        items,
        image_size=image_size,
        resize_mode=resize_mode,
        encoder_metrics=encoder_metrics,
        decoder_metrics=decoder_metrics,
    )

    metrics = {
        "num_pairs": len(items),
        "image_size": image_size,
        "beta_start": beta_start,
        "beta_end": beta_end,
        "num_steps": num_steps,
        "final_alpha": float(alpha[-1]),
        "final_alpha_sq": float(alpha[-1] ** 2),
        "encoded_saturation_rate": dataset_stats["encoded_saturation_rate"],
        "rgb_correlation": dataset_stats["rgb_correlation"],
        "alpha_channel_mae_raw": dataset_stats["alpha_channel_mae_raw"],
        "model_parameters": int(sum(p.numel() for p in encoder_model.parameters())),
        "encoder_mae_all": encoder_metrics["mae_all"],
        "decoder_mae_all": decoder_metrics["mae_all"],
        "encoder_mae_rgb": encoder_metrics["mae_rgb"],
        "decoder_mae_rgb": decoder_metrics["mae_rgb"],
        "encoder_psnr_rgb": encoder_metrics["psnr_rgb"],
        "decoder_psnr_rgb": decoder_metrics["psnr_rgb"],
        "baseline_mae_all": encoder_metrics["baseline_mae_all"],
        "baseline_mae_rgb": encoder_metrics["baseline_mae_rgb"],
        "baseline_psnr_rgb": encoder_metrics["baseline_psnr_rgb"],
        "example_names": example_names,
    }

    write_metric_macros(metrics)
    (GENERATED_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
