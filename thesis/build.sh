#!/usr/bin/env bash
# Rebuild the thesis from the logged results: figures -> LaTeX + Word -> PDF -> Overleaf zip.
set -euo pipefail
cd "$(dirname "$0")"
PY="${PY:-../.venv/bin/python}"
$PY build_figures.py
$PY render.py
latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build thesis.tex > build/latexmk.log 2>&1 || { tail -30 build/latexmk.log; exit 1; }
cp build/thesis.pdf Thesis_Kanan_Kotwani.pdf
rm -f Thesis_Kanan_Kotwani_LaTeX.zip
zip -q -r Thesis_Kanan_Kotwani_LaTeX.zip thesis.tex figures
echo "Built Thesis_Kanan_Kotwani.pdf, Thesis_Kanan_Kotwani.docx and Thesis_Kanan_Kotwani_LaTeX.zip"
