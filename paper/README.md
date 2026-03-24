# Research Paper Draft

This directory contains a LaTeX draft paper for the current stochastic image encoding experiments.

## Contents

- `main.tex`: manuscript source
- `references.bib`: bibliography
- `scripts/generate_assets.py`: regenerates figures and metric macros from the current repo state
- `generated/metrics.tex`: generated LaTeX macros used by `main.tex`
- `figures/`: generated PNG figures used by the paper

## Regenerate Assets

From the repository root:

```powershell
python paper/scripts/generate_assets.py
```

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
