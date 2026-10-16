#!/usr/bin/env bash
# scripts/reproduce_all.sh
#
# Single entry point that reproduces every figure/number (CLAUDE.md §2
# "Reproducibility", §11). Assumes data/raw/{adfecgdb,ninfea}/ is already populated
# (scripts/00_download.py) -- this script does not download data.
#
# While real ADFECGDB GA is unavailable, run with the synthetic labels (pipeline check
# only, nothing reportable -- outputs are stamped synthetic_ga: true):
#   EXPERIMENT=synthetic_ga_smoketest bash scripts/reproduce_all.sh
#
# Implements: CLAUDE.md §4 scripts/reproduce_all.sh.
set -euo pipefail

EXPERIMENT="${EXPERIMENT:-}"
EXP_ARGS=()
if [[ -n "$EXPERIMENT" ]]; then EXP_ARGS=(--experiment "$EXPERIMENT"); fi

cd "$(dirname "$0")/.."

for dataset in adfecgdb ninfea; do
  python scripts/01_preprocess.py --dataset "$dataset"
  python scripts/02_memd.py --dataset "$dataset"
done

python scripts/03_train_pinn.py
python scripts/04_extract_fecg.py

for dataset in adfecgdb ninfea; do
  python scripts/05_features.py --dataset "$dataset"
done

python scripts/06_train_cvae.py "${EXP_ARGS[@]}"
python scripts/07_train_transformer.py "${EXP_ARGS[@]}"
python scripts/08_evaluate.py "${EXP_ARGS[@]}"
