# Research Paper Draft

This directory contains a LaTeX draft paper for the current stochastic image encoding experiments.

The current checked-in paper checkpoints use discrete `8-bit` RGB prediction with preserved alpha passthrough.

## Contents

- `main.tex`: manuscript source
- `references.bib`: bibliography
- `scripts/generate_assets.py`: regenerates figures and metric macros from the current repo state
- `generated/metrics.tex`: generated LaTeX macros used by `main.tex`
- `figures/`: generated PNG figures used by the paper when assets are regenerated locally
- `../alshicrypt-multimodal.wiki/figures/`: currently tracked figure copies that the manuscript can also read directly

## Source of Truth

The paper is intended to stay aligned with the checked-in code and paper-ready checkpoints.

- canonical checkpoints: `../models/paper_eval/`
- canonical generator: `scripts/generate_assets.py`
- generated artifacts: `generated/metrics.tex`, `generated/metrics.json`, and `figures/*.png`

Do not hand-edit generated metrics or paper figures. Regenerate them from the current repo state.

For the full repo-paper synchronization workflow, see `../PAPER_SYNC.md`.

## Regenerate Assets

From the repository root:

```powershell
python paper/scripts/generate_assets.py
```

If the paper checkpoints change, regenerate assets before rebuilding the manuscript.

## Compile

The paper has been compiled successfully with MiKTeX on this machine. The output PDF is:

- `paper/main.pdf`

From `paper/`, the standard build sequence is:

```powershell
pdflatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

If MiKTeX is installed but not on `PATH`, call the executables directly from:

`C:\Users\SamuelCavazos\AppData\Local\Programs\MiKTeX\miktex\bin\x64\`
