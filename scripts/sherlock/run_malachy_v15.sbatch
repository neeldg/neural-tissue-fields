#!/bin/bash
#SBATCH --job-name=malachy_v15
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=logs/malachy_%j.out
#SBATCH --error=logs/malachy_%j.err

# ── MALACHY v1.5 — single Sherlock GPU job ────────────────────────────────
#
# Environment variables (set via --export when calling sbatch):
#   MODEL          mlp | gridfield
#   HOLDOUT        stripe | quadrant | random
#   SEED           integer
#   EPOCHS         integer
#   OUTPUT_PREFIX  string prefix for all output filenames
#
# Standalone example (one run, no sweep):
#   mkdir -p logs
#   sbatch --export=MODEL=mlp,HOLDOUT=stripe,SEED=0,EPOCHS=200,OUTPUT_PREFIX=test_run \
#       scripts/sherlock/run_malachy_v15.sbatch

set -euo pipefail

# ── Defaults ──────────────────────────────────────────────────────────────
: "${MODEL:=mlp}"
: "${HOLDOUT:=stripe}"
: "${SEED:=0}"
: "${EPOCHS:=200}"
: "${OUTPUT_PREFIX:=v15_${HOLDOUT}_${MODEL}_seed${SEED}_e${EPOCHS}}"

# ── Job header ────────────────────────────────────────────────────────────
echo "============================================================"
echo "MALACHY v1.5 — Sherlock GPU run"
echo "  MODEL         : ${MODEL}"
echo "  HOLDOUT       : ${HOLDOUT}"
echo "  SEED          : ${SEED}"
echo "  EPOCHS        : ${EPOCHS}"
echo "  OUTPUT_PREFIX : ${OUTPUT_PREFIX}"
echo "  SLURM job     : ${SLURM_JOB_ID:-local}"
echo "  Node          : $(hostname -s)"
echo "  GPU           : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null \
                          | head -1 || echo 'not detected')"
echo "  Started       : $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"
echo ""

# ── Working directory ─────────────────────────────────────────────────────
# Always run from repo root (where sbatch was submitted from).
cd "${SLURM_SUBMIT_DIR:-.}"

# ── (Optional) Load Sherlock modules ─────────────────────────────────────
# Uncomment and adjust to match what is available on your cluster.
# Check available versions with: ml spider cuda  /  ml spider python
#
# ml load python/3.11.7
# ml load cuda/12.1.1

# ── Activate virtual environment ──────────────────────────────────────────
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
    echo "Activated: $(which python)  ($(python --version))"
else
    echo "WARNING: .venv not found — using system Python: $(which python)"
fi

echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo ""

# ── Ensure output directories exist ──────────────────────────────────────
mkdir -p outputs/predictions outputs/figures outputs/summaries logs

# ── Training ──────────────────────────────────────────────────────────────
echo "--- Training ---"
python scripts/train_2d_holdout.py \
    --input       data/processed/breast_tma_hd_0_square016.parquet \
    --gene-list   outputs/spatial_genes/breast_tma_hd_0_square016_spatial_genes.txt \
    --holdout-mode "${HOLDOUT}" \
    --model       "${MODEL}" \
    --epochs      "${EPOCHS}" \
    --seed        "${SEED}" \
    --device      cuda \
    --plot-genes  CCND1,IFI44,FADS2,CHD2,COMP,NDRG1 \
    --output-prefix "${OUTPUT_PREFIX}"

echo ""
echo "Training complete: $(date '+%Y-%m-%d %H:%M:%S')"

# ── Summary ───────────────────────────────────────────────────────────────
SUMMARY_PATH="outputs/summaries/${OUTPUT_PREFIX}_summary.txt"
echo ""
echo "--- Summary ---"
python scripts/summarize_run.py \
    --metrics "outputs/predictions/${OUTPUT_PREFIX}_metrics.csv" \
    | tee "${SUMMARY_PATH}"

echo ""
echo "Summary saved → ${SUMMARY_PATH}"
echo "Job done: $(date '+%Y-%m-%d %H:%M:%S')"
