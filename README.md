# AlshiCrypt

AlshiCrypt is an experiment in **multi-modal, machine-learning-based encryption**. The idea is to treat visual and audio media as two halves of a single tensor and train a pair of neural networks, but only after picking a strictly invertible transform. We first sample a reversible (bijective) permutation across both modalities, then use that transform to synthesize training data so the learned encryptor/decryptor always has a perfect inverse to mimic.

- an **encryptor** that learns to transform aligned image/audio tensors into apparently unrelated tensors, and
- a **decryptor** that learns to undo the same transform and reconstruct the original signals.

Because both networks are trained on the exact invertible transform that produced the data, the system behaves like a differentiable, key-conditioned cipher where the “key” is the randomly sampled media transform.

## Project workflow

1. **Sample a random transform** – `alshicrypt.random_media_transform` builds invertible permutations and channel shuffles for both modalities, guaranteeing that the same transform object can be inverted exactly.
2. **Create paired training data** – `train.py` or `alshicrypt.generate` load files from `samples/images` and `samples/audio/wav`, apply the transform, and serialize `(original, encrypted)` tensor pairs.
3. **Train encryptor/decryptor models** – `alshicrypt.generate` builds two autoencoders (see `src/alshicrypt/model.py`) and optimizes them until they reach near-perfect reconstruction accuracy on the held-out data, leveraging the fact that every training example shares the same invertible transform.
4. **Evaluate / demo** – `sample.py` demonstrates the tensor pipeline, while the generation script prints per-epoch reconstruction accuracy so you can judge when the model has fully learned the mapping.

## Repository guide

| Path | Purpose |
| --- | --- |
| `src/alshicrypt/data.py` | Media loading pipeline (CIFAR-style images and WAV audio). |
| `src/alshicrypt/transforms.py` | Random invertible transforms applied jointly to images and audio. |
| `src/alshicrypt/model.py` | Autoencoder definitions plus reconstruction-loss helpers. |
| `src/alshicrypt/generate.py` | Orchestrates transform sampling, dataset generation, and encryptor/decryptor training. |
| `train.py` | CLI helper that quickly builds a `.pt` dataset of transformed multimodal pairs. |
| `sample.py` | Small demo that loads an image/audio tensor and optionally exercises a random transform. |

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e .

# Train encryptor/decryptor on the bundled sample data
python -m alshicrypt.generate \
  --image-dir samples/images \
  --audio-dir samples/audio/wav \
  --num-train-images 200 \
  --num-train-audio 200 \
  --num-test-images 10 \
  --num-test-audio 10
```

The command above prints accuracy for both models each epoch, stopping early once reconstruction error falls below the tolerance threshold.

## Custom data

Place your own PNG images under `samples/images` (or another directory referenced via `--image-dir`) and WAV files under `samples/audio/wav`. The CLI flags on `alshicrypt.generate` let you control how many items to draw for training and testing, learning rate, batch size, and the reconstruction tolerance.

If you only need the serialized dataset for external experiments, run `python train.py --images <dir> --audio <dir> --num-images <N> --num-audio <M> --out dataset/train.pt` to save the paired tensors without training models.

## Demo pipeline

Use `python sample.py --demo-transform` to load a single image, convert it into the multimodal tensor representation, apply a fresh random transform, and report the max absolute reconstruction error when inverting the transform. This helps verify the deterministic tensor pipeline before full training.
