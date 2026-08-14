#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/manuscript"
latexmk -lualatex -interaction=nonstopmode -halt-on-error main.tex
open main.pdf
