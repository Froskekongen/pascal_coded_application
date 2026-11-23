# Building Activity Representations with Pascal-Coded Signatures

- [Paper (PDF)](pascal_coded_signatures.pdf) and [LaTeX](source/paper/main.tex)
- [Mathematical and experimental companion (PDF)](pascal_coded_signatures_companion.pdf) and [LaTeX](source/companion/appendix.tex)

## Build

From the repository root, run `sh paper/source/build.sh`. It uses pdfLaTeX or falls
back to Tectonic. Set `TECTONIC` to select a custom executable for the fallback.
The two PDFs are written directly to `paper/` and intermediate files to
`paper/source/build/`. The ESANN manuscript is six A4 pages, including references.

In a GitHub-linked Overleaf project, select pdfLaTeX and set **Main document** to
`paper/source/paper/main.tex` or `paper/source/companion/appendix.tex`. The root `latexmkrc`
resolves the shared sources and figure paths. Recompile from scratch after switching.

## Reproduction and evidence

The [reproduction guide](../reproduction/README.md) covers the three retained UCI
HAR maps and all nine NATOPS maps. The archived [HAR and transfer
records](source/evidence/har/README.md), [NATOPS records](source/evidence/natops/README.md),
and [timing records](source/evidence/timing/README.md) document the conventional
HAR controls and the companion’s additional results, including the DSADS study.
