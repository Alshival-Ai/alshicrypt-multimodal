# Alshicrypt Multimodal

<p align="center">
  <img src="https://alshival.ai/static/img/logos/brain1_transparent.png" alt="Alshival.Ai Hero" width="280" />
</p>

<p align="center"><strong>Alshival.Ai</strong></p>

Research prototype for learned image encryption/decryption using a shared stochastic distortion process.

## Project Purpose

This repository demonstrates a machine-learning workflow for image transformation in transit:

1. Apply the same stochastic process to each source image (`Original -> Encrypted`).
2. Train an Encrypter model to learn that transformation.
3. Train a Decrypter model to invert it (`Encrypted -> Original`).
4. Export pretrained forward/reverse checkpoints for downstream applications.

The Pokemon dataset is used as a controlled, reproducible example.  
Target applications include secure image workflows in healthcare and security.

## Research Framing

- The Encrypter and Decrypter operate as a learned pair (conceptually similar to paired keys).
- Sender side: encrypt image before transmission.
- Receiver side: decrypt image to reconstruct the original content.
- Shared stochastic generation setup is applied consistently across the dataset.
- The repository CLI still uses `--stage encoder|decoder`; in docs and the paper, those correspond to `Encrypter` and `Decrypter`.

## Repository Workflow

### 1) Generate Distorted Dataset

This script applies the same stochastic process settings to all PNGs under `pokemon/` and writes outputs to `pokemon_distorted/`.

```powershell
python pokemon_distort.py
```

Resume support is built in:

- Existing outputs are skipped by default.
- Use `--overwrite` to regenerate all images.

```powershell
python pokemon_distort.py --overwrite
```

### 2) Prepare Training Pairs

Builds a CSV manifest of matched `original/distorted` pairs.

```powershell
python prepare_training_pairs.py --original-root pokemon --encoded-root pokemon_distorted --out-csv models/pokemon_pairs.csv
```

### 3) Train Encrypter Model

Learns: `Original -> Encrypted`

```powershell
python train_pokemon_model.py --stage encoder --original-root pokemon --encoded-root pokemon_distorted --epochs 500 --target-mae 0.0
```

### 4) Train Decrypter Model

Learns: `Encrypted -> Original`

```powershell
python train_pokemon_model.py --stage decoder --original-root pokemon --encoded-root pokemon_distorted --epochs 500 --target-mae 0.0
```

## Pretrained Model Outputs

Training writes checkpoints to `models/`:

- `encoder_best.pt`
- `encoder_best.ts` (TorchScript for app integration)
- `encoder_last.pt`
- `decoder_best.pt`
- `decoder_best.ts` (TorchScript for app integration)
- `decoder_last.pt`

## Core Scripts

- `pokemon_distort.py`: dataset distortion pipeline with progress bar and resume behavior
- `prepare_training_pairs.py`: pair manifest generation
- `train_pokemon_model.py`: GPU-first training loop for the forward (`encoder`/Encrypter) and reverse (`decoder`/Decrypter) models
- `training/models.py`: U-Net-like CNN architecture
- `training/pokemon_pairs.py`: pair matching + dataset loader

## Notes

- Recommended: NVIDIA GPU with CUDA-enabled PyTorch.
- `target-mae=0.0` is supported as a stopping criterion, but convergence to exact zero depends on architecture capacity, image preprocessing, numeric precision, and optimization settings.
- This repository is a research prototype and not a replacement for standard, formally analyzed cryptographic protocols.
