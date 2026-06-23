#!/bin/bash
# Submit the full MALACHY v1.5 sweep to Sherlock.
#
# Run from repo root:
#   bash scripts/sherlock/launch_v15_sweep.sh
#
# Optionally override epochs:
#   EPOCHS=100 bash scripts/sherlock/launch_v15_sweep.sh

set -euo pipefail

SCRIPT="scripts/sherlock/run_malachy_v15.sbatch"
EPOCHS="${EPOCHS:-200}"

MODELS=(mlp gridfield)
HOLDOUTS=(stripe quadrant)
SEEDS=(0 1 2)

# ── Preflight checks ──────────────────────────────────────────────────────
if [ ! -f "$SCRIPT" ]; then
    echo "ERROR: sbatch script not found: $SCRIPT"
    echo "Run this from the repo root."
    exit 1
fi

if [ ! -f "data/processed/breast_tma_hd_0_square016.parquet" ]; then
    echo "ERROR: input parquet not found: data/processed/breast_tma_hd_0_square016.parquet"
    echo "Transfer it first — see scripts/sherlock/README_sherlock.md"
    exit 1
fi

if [ ! -f "outputs/spatial_genes/breast_tma_hd_0_square016_spatial_genes.txt" ]; then
    echo "ERROR: gene list not found: outputs/spatial_genes/breast_tma_hd_0_square016_spatial_genes.txt"
    echo "Run select_spatial_genes.py first — see scripts/sherlock/README_sherlock.md"
    exit 1
fi

# ── Create output directories (required before sbatch so log paths resolve) ──
mkdir -p logs outputs/predictions outputs/figures outputs/summaries

# ── Submit jobs ───────────────────────────────────────────────────────────
n_submitted=0
job_ids=()

echo "Submitting MALACHY v1.5 sweep  (epochs=${EPOCHS})"
echo ""

for model in "${MODELS[@]}"; do
    for holdout in "${HOLDOUTS[@]}"; do
        for seed in "${SEEDS[@]}"; do
            prefix="v15_${holdout}_${model}_seed${seed}_e${EPOCHS}"

            job_id=$(sbatch \
                --job-name="malachy_${model}_${holdout}_s${seed}" \
                --output="logs/${prefix}_%j.out" \
                --error="logs/${prefix}_%j.err" \
                --export=ALL,MODEL="${model}",HOLDOUT="${holdout}",SEED="${seed}",EPOCHS="${EPOCHS}",OUTPUT_PREFIX="${prefix}" \
                "$SCRIPT" \
                | awk '{print $NF}')

            echo "  [${n_submitted}] ${prefix}  →  job ${job_id}"
            job_ids+=("$job_id")
            n_submitted=$((n_submitted + 1))
        done
    done
done

# ── Summary ───────────────────────────────────────────────────────────────
echo ""
echo "Submitted ${n_submitted} jobs."
echo ""
echo "Job IDs: ${job_ids[*]}"
echo ""
echo "Monitor:"
echo "  squeue -u \$USER"
echo "  squeue -u \$USER -o '%.18i %.9P %.30j %.8T %.10M %.6D %R'"
echo ""
echo "Watch a log:"
echo "  tail -f logs/v15_stripe_mlp_seed0_e${EPOCHS}_*.out"
echo ""
echo "Cancel all:"
echo "  scancel ${job_ids[*]}"
echo ""
echo "Collect summaries when done:"
echo "  cat outputs/summaries/v15_*_summary.txt"
