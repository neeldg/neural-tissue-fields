# MALACHY v1.5 on Sherlock

This guide walks through running the MALACHY v1.5 GPU sweep on Stanford's Sherlock cluster.

---

## 1. Clone or pull the repo on Sherlock

SSH into Sherlock, then:

```bash
ssh <sunetid>@login.sherlock.stanford.edu

# First time: clone
cd $HOME  # or $GROUP_HOME or $SCRATCH — see note below
git clone <your-repo-url> neural-tissue-fields
cd neural-tissue-fields
```

```bash
# Subsequent times: pull latest changes
cd neural-tissue-fields
git pull
```

> **Storage note**: Sherlock home quotas are limited (15 GB).  For large data files use
> `$SCRATCH` (per-user, not backed up) or `$GROUP_SCRATCH` (shared with PI group).
> The repo code itself is small and fine in `$HOME`.

---

## 2. Load a Python module

Sherlock provides Python via the module system.  Check what is available:

```bash
ml spider python
```

Then load a recent version (3.10+ required):

```bash
ml load python/3.11.7   # adjust version as needed
```

Add this to your `~/.bashrc` or run it before any MALACHY work.

---

## 3. Create and activate a virtual environment

```bash
cd neural-tissue-fields

python -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
```

---

## 4. Install requirements

```bash
# Core scientific stack
pip install numpy pandas scipy matplotlib pyarrow

# PyTorch — match the CUDA version available on Sherlock GPU nodes
# Check: ml spider cuda
# Example for CUDA 12.1:
pip install torch --index-url https://download.pytorch.org/whl/cu121

# Scanpy (only needed if preprocessing new Visium data on Sherlock)
pip install scanpy
```

Verify PyTorch sees CUDA (run on a GPU node or in an interactive session):

```bash
srun --partition=gpu --gres=gpu:1 --pty bash
source .venv/bin/activate
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
exit
```

---

## 5. Transfer the processed data

Transfer from your local machine:

```bash
# From your laptop (in the repo root):

# Processed section parquet (~50–100 MB)
scp data/processed/breast_tma_hd_0_square016.parquet \
    <sunetid>@dtn.sherlock.stanford.edu:~/neural-tissue-fields/data/processed/

# Spatial gene list (tiny text file)
scp outputs/spatial_genes/breast_tma_hd_0_square016_spatial_genes.txt \
    <sunetid>@dtn.sherlock.stanford.edu:~/neural-tissue-fields/outputs/spatial_genes/
```

> Use `dtn.sherlock.stanford.edu` (the data transfer node), not the login node, for file
> transfers.  For large files (>1 GB) use `rsync -avz` or Globus.

If you need to generate the spatial gene list on Sherlock instead:

```bash
# On Sherlock (after installing deps)
python scripts/select_spatial_genes.py \
    --input data/processed/breast_tma_hd_0_square016.parquet \
    --top-n 30
```

---

## 6. Submit the sweep

From the repo root on Sherlock:

```bash
bash scripts/sherlock/launch_v15_sweep.sh
```

This submits 12 jobs (2 models × 2 holdouts × 3 seeds, 200 epochs each):

| Model    | Holdout  | Seeds    |
|----------|----------|----------|
| mlp      | stripe   | 0, 1, 2  |
| mlp      | quadrant | 0, 1, 2  |
| gridfield| stripe   | 0, 1, 2  |
| gridfield| quadrant | 0, 1, 2  |

Output prefixes follow the pattern: `v15_{holdout}_{model}_seed{seed}_e200`

To run a custom epoch count:

```bash
EPOCHS=500 bash scripts/sherlock/launch_v15_sweep.sh
```

To submit a single job manually:

```bash
mkdir -p logs
sbatch \
    --export=MODEL=mlp,HOLDOUT=stripe,SEED=0,EPOCHS=200,OUTPUT_PREFIX=v15_stripe_mlp_seed0_e200 \
    scripts/sherlock/run_malachy_v15.sbatch
```

---

## 7. Monitor jobs

```bash
# All your jobs
squeue -u $USER

# Formatted view with more detail
squeue -u $USER -o "%.18i %.9P %.30j %.8T %.10M %.6D %R"

# Watch a specific job log in real time
tail -f logs/v15_stripe_mlp_seed0_e200_<jobid>.out

# Check GPU utilization (inside an interactive session on the node)
nvidia-smi

# Cancel all your jobs
scancel -u $USER

# Cancel a specific job
scancel <jobid>
```

Expected wall time per job: **5–20 minutes** (200 epochs on a V100/A100 with 100k spots × 30 genes).  The 6-hour limit is very conservative.

---

## 8. Collect results when jobs finish

**Individual summaries** (printed at end of each job log and saved to file):

```bash
cat outputs/summaries/v15_stripe_mlp_seed0_e200_summary.txt
```

**All summaries at once** (quick comparison across all runs):

```bash
for f in outputs/summaries/v15_*_summary.txt; do
    echo "=== $f ==="
    grep -E "Method|Mean|Median|neural_field|knn" "$f" | head -12
    echo ""
done
```

**Per-run metrics CSVs** (gene-level detail):

```bash
ls outputs/predictions/v15_*_metrics.csv
```

**Transfer results back to your laptop**:

```bash
# From your laptop:
rsync -avz \
    <sunetid>@dtn.sherlock.stanford.edu:~/neural-tissue-fields/outputs/ \
    outputs/sherlock_v15/
```

---

## Partition and resource notes

The sbatch script uses `--partition=gpu` by default.  If you have PI ownership allocations, change to `--partition=owners` for lower queue priority (but faster start):

```bash
# Override partition at submission:
sbatch --partition=owners \
    --export=MODEL=mlp,HOLDOUT=stripe,SEED=0,EPOCHS=200,OUTPUT_PREFIX=test \
    scripts/sherlock/run_malachy_v15.sbatch
```

Sherlock GPU nodes have V100 (32 GB) or A100 (40/80 GB) cards.  The 32 GB memory
allocation in the sbatch script is sufficient for all MALACHY v1.5 experiments.

---

## Module line in the sbatch script

Open `scripts/sherlock/run_malachy_v15.sbatch` and uncomment the module lines near the
top of the script body, adjusting versions to match what Sherlock offers:

```bash
ml load python/3.11.7
ml load cuda/12.1.1
```

Check available versions with:

```bash
ml spider python
ml spider cuda
```
