# Paper Sync Workflow

This repository treats the checked-in code and paper assets as one aligned system.
Empirical claims in the paper, README, and wiki should be traceable to the current
repository state.

## Canonical Sources of Truth

- `models/paper_eval/`: paper-ready Encrypter/Decrypter checkpoints
- `pokemon_restored/`: derived Decrypter reconstructions generated from the current distorted dataset
- `paper/scripts/generate_assets.py`: canonical generator for paper metrics and figures
- `paper/generated/metrics.tex`: generated LaTeX macros consumed by `paper/main.tex`
- `paper/generated/metrics.json`: generated machine-readable metric summary
- `paper/main.tex`: manuscript source
- `paper/main.pdf`: compiled manuscript output

The current implementation models RGB on the discrete `0..255` lattice and preserves alpha exactly.

## Generated Files

These files should be treated as derived artifacts, not hand-edited sources:

- `paper/generated/metrics.tex`
- `paper/generated/metrics.json`
- `paper/figures/*.png`

Empirical values shown in the manuscript should come from these generated files or
from assets produced by the same workflow.

## Sync Steps

From the repository root:

```powershell
python train_pokemon_model.py --stage encoder --original-root pokemon --encoded-root pokemon_distorted --out-dir models/paper_eval --epochs 100 --target-mae 0.0 *> models/paper_eval/encoder.log
python train_pokemon_model.py --stage decoder --original-root pokemon --encoded-root pokemon_distorted --out-dir models/paper_eval --epochs 100 --target-mae 0.0 *> models/paper_eval/decoder.log
python restore_pokemon.py --input-dir pokemon_distorted --output-dir pokemon_restored --checkpoint models/paper_eval/decoder_best.pt --overwrite
python paper/scripts/generate_assets.py
```

Then compile the manuscript from `paper/`:

```powershell
pdflatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

## Alignment Rules

- Paper figures that claim model output must use actual generated outputs, not placeholders.
- `pokemon_restored/` should be regenerated after decoder checkpoint changes if those restored examples are being inspected or reused.
- README and wiki text should refer to `models/paper_eval/` when describing the paper-ready checkpoints.
- Conceptual or illustrative prose is allowed, but it must be clearly labeled as conceptual when it is not an empirical result.
- The paper should not claim broader validation than the checked-in code and artifacts support.
- Image and audio may be used to build intuition, but the high-level framing should remain modality-agnostic when that is the stated research goal.

## Pre-Push Checklist

- Confirm `paper/scripts/generate_assets.py` still points at the intended paper checkpoint directory.
- Regenerate paper metrics and figures after any checkpoint change.
- Rebuild `paper/main.pdf` after paper edits.
- Verify any showcased examples in the abstract or figures are real outputs from the current paper assets.
- Verify README, paper docs, and wiki wording still match the current checkpoints, metrics, and terminology.
