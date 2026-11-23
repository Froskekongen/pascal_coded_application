#!/bin/sh
set -eu
cd "$(dirname "$0")"
mkdir -p build
export TEXINPUTS=".//:${TEXINPUTS:-}"
# Prefer the submission/Overleaf engine; Tectonic is a portable fallback.
if command -v pdflatex >/dev/null 2>&1; then
    for manuscript in paper/main companion/appendix; do
        pdflatex -interaction=nonstopmode -halt-on-error -output-directory=build "$manuscript.tex"
        pdflatex -interaction=nonstopmode -halt-on-error -output-directory=build "$manuscript.tex"
    done
else
    tex_engine="${TECTONIC:-tectonic}"
    if ! command -v "$tex_engine" >/dev/null 2>&1; then
        echo 'Install pdfLaTeX or Tectonic (or set TECTONIC to its executable path).' >&2
        exit 1
    fi
    for manuscript in paper/main companion/appendix; do
        "$tex_engine" --keep-logs --keep-intermediates --outdir build "$manuscript.tex"
    done
fi
cp build/main.pdf ../pascal_coded_signatures.pdf
cp build/appendix.pdf ../pascal_coded_signatures_companion.pdf
