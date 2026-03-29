# Alshicrypt Multimodal

<p align="center">
  <img src="https://alshival.ai/static/img/logos/brain1_transparent.png" alt="Alshival.Ai Hero" width="280" />
</p>

<p align="center"><strong>Alshival.Ai</strong></p>

<p align="center">Built by <strong>The Data Team</strong> at <a href="https://alshival.ai">Alshival.Ai</a>.</p>

<p align="center">
  <a href="paper/main.pdf">
    <img alt="Read the Paper" src="https://img.shields.io/badge/Read%20the%20Paper-PDF-0A66C2?style=for-the-badge" />
  </a>
</p>

Research prototype for multimodal learned encryption/decryption using a shared stochastic distortion process.

## Project Purpose

This repository demonstrates a machine-learning workflow for image transformation in transit:

1. Apply the same stochastic process to each source image (`Original -> Encrypted`).
2. Train an Encrypter model to learn that transformation.
3. Train a Decrypter model to invert it (`Encrypted -> Original`).
4. Export pretrained forward/reverse checkpoints for downstream applications.

In the current image implementation, RGB is modeled as discrete `8-bit` channel values and alpha is preserved exactly rather than learned.
That matches the current PNG-based dataset well, but it is also a present limitation. Moving to truly continuous color scales will likely require a different output parameterization and more varied training data than the current sprite corpus provides.

The Pokemon dataset is used as a controlled, reproducible image example.  
The image pipeline is the current working modality in the repository. The broader research goal is multimodal: apply the same core transport, corruption, and learned inversion principles to other media types as they mature, with audio as the next planned target.

Target applications include secure image and audio workflows in healthcare and security.

This repository reflects the broader Alshival.Ai direction as The Data Team: building practical AI and data systems that can extend across modalities, starting with images and then carrying the same framework into audio.

## Research Framing

- The Encrypter and Decrypter operate as a learned pair (conceptually similar to paired keys).
- Sender side: encrypt image before transmission.
- Receiver side: decrypt image to reconstruct the original content.
- Shared stochastic generation setup is applied consistently across the dataset.
- The repository CLI still uses `--stage encoder|decoder`; in docs and the paper, those correspond to `Encrypter` and `Decrypter`.
- The current repository demonstrates the image case first; the same research direction is intended to extend to audio next under the same general framework.

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

The current checkpoints use a discrete-RGB objective:

- three independent `256`-way channel predictions for `R`, `G`, and `B`
- preserved alpha copied through from the input image

### 5) Restore Distorted Sprites with the Decrypter

Runs the trained Decrypter across `pokemon_distorted/` and writes reconstructed
sprites to `pokemon_restored/`.

```powershell
python restore_pokemon.py --input-dir pokemon_distorted --output-dir pokemon_restored --checkpoint models/paper_eval/decoder_best.pt
```

Use `--overwrite` to refresh existing restored outputs after retraining.

## Pretrained Model Outputs

Training writes checkpoints to `models/`:

- `encoder_best.pt`
- `encoder_best.ts` (TorchScript for app integration)
- `encoder_last.pt`
- `decoder_best.pt`
- `decoder_best.ts` (TorchScript for app integration)
- `decoder_last.pt`

Paper-ready pretrained checkpoints are also included under `models/paper_eval/` so others can test the current Encrypter/Decrypter pair directly without retraining first.

The corresponding restored sample outputs can be regenerated locally into
`pokemon_restored/` with `restore_pokemon.py`.

## Paper Alignment

The GitHub repo and the research paper are intended to stay tightly aligned.
The current manuscript source is `paper/main.tex`, and the compiled paper is linked near the top of this README for quick reference.
For the canonical paper-sync workflow and source-of-truth files, see `PAPER_SYNC.md`.

## Core Scripts

- `pokemon_distort.py`: dataset distortion pipeline with progress bar and resume behavior
- `prepare_training_pairs.py`: pair manifest generation
- `train_pokemon_model.py`: GPU-first training loop for the forward (`encoder`/Encrypter) and reverse (`decoder`/Decrypter) models using discrete RGB classification
- `restore_pokemon.py`: batch Decrypter inference that reconstructs `pokemon_distorted/` into `pokemon_restored/`
- `training/models.py`: U-Net-like CNN with a discrete RGB logits head
- `training/pokemon_pairs.py`: pair matching + dataset loader for float inputs, integer RGB targets, and preserved alpha

## Notes

- Recommended: NVIDIA GPU with CUDA-enabled PyTorch.
- `target-mae=0.0` is supported as a stopping criterion, but convergence to exact zero depends on architecture capacity, image preprocessing, numeric precision, and optimization settings.
- This repository is a research prototype and not a replacement for standard, formally analyzed cryptographic protocols.
