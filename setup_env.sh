#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

export PROJECT_ROOT
export HF_HOME="$PROJECT_ROOT/.cache/huggingface"
export TRANSFORMERS_CACHE="$PROJECT_ROOT/.cache/huggingface/transformers"
export HF_DATASETS_CACHE="$PROJECT_ROOT/.cache/huggingface/datasets"
export TORCH_HOME="$PROJECT_ROOT/.cache/torch"
export PIP_CACHE_DIR="$PROJECT_ROOT/.cache/pip"
export TMPDIR="$PROJECT_ROOT/.tmp"

python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install --index-url https://download.pytorch.org/whl/cu128 torch==2.10.0 torchvision==0.25.0
python -m pip install -r requirements.txt

mkdir -p \
	"$HF_HOME" \
	"$TRANSFORMERS_CACHE" \
	"$HF_DATASETS_CACHE" \
	"$TORCH_HOME" \
	"$PIP_CACHE_DIR" \
	"$TMPDIR" \
	Models Results Datasets/financial_phrasebank

echo "Environment ready."
echo "For every new shell session run:"
echo "  source ./activate_project.sh"
echo "or for csh/tcsh:"
echo "  source ./activate_project.csh"
