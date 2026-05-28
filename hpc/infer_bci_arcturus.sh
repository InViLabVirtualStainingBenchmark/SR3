#!/bin/bash
#SBATCH --job-name=sr3_infer_bci_arcturus
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=60G
#SBATCH --time=24:00:00
#SBATCH -A ap_invilab_td_thesis
#SBATCH -p arcturus_gpu
#SBATCH --gres=gpu:1
#SBATCH -o /data/antwerpen/212/vsc21211/projects/sr3/logs/%x.%j.out
#SBATCH -e /data/antwerpen/212/vsc21211/projects/sr3/logs/%x.%j.err

# infer_bci_arcturus.sh — BCI test inference for SR3 on arcturus_gpu (AMD/ROCm).
# Loads best.pt automatically from UNet/pnt_BCI/.

set -euo pipefail

export REPO_DIR="$VSC_DATA/projects/sr3/code/SR3"
export LOG_DIR="$VSC_DATA/projects/sr3/logs"

GRP_SCRATCH="/scratch/antwerpen/grp/ap_invilab_td_thesis"
export OUT_DIR="$GRP_SCRATCH/diffusion-predictions/sr3/bci_test"

CONTAINER="$VSC_SCRATCH/containers/sr3_rocm.sif"
RUN_SCRIPT="$REPO_DIR/hpc/run_infer_bci.sh"

# =========================================================
# ENVIRONMENT
# =========================================================

module purge
module load calcua/2025a

# =========================================================
# PRE-FLIGHT CHECKS
# =========================================================

echo "=== Container ==="
if [ ! -f "$CONTAINER" ]; then
    echo "ERROR: Container not found: $CONTAINER"
    exit 1
fi
echo "  $CONTAINER"

echo "=== Checking dataset ==="
if [ ! -f "$VSC_SCRATCH/datasets/BCI.sqsh" ]; then
    echo "ERROR: BCI SquashFS archive not found: $VSC_SCRATCH/datasets/BCI.sqsh"
    exit 1
fi
echo "  BCI.sqsh : $(du -h "$VSC_SCRATCH/datasets/BCI.sqsh" | cut -f1)"

echo "=== Checking checkpoint ==="
BEST_PT="$REPO_DIR/UNet/pnt_BCI/best.pt"
if [ ! -f "$BEST_PT" ]; then
    echo "ERROR: best.pt not found: $BEST_PT"
    exit 1
fi
echo "  best.pt : $(du -h "$BEST_PT" | cut -f1)"

# =========================================================
# RUN
# =========================================================

mkdir -p "$VSC_SCRATCH/datasets/BCI"
mkdir -p "$OUT_DIR"

export MIOPEN_USER_DB_PATH=/tmp/miopen_${SLURM_JOB_ID}
mkdir -p "$MIOPEN_USER_DB_PATH"

srun apptainer exec --rocm \
    --env MIOPEN_USER_DB_PATH="$MIOPEN_USER_DB_PATH" \
    -B "$VSC_SCRATCH/datasets/BCI.sqsh:$VSC_SCRATCH/datasets/BCI:image-src=/" \
    -B "$VSC_DATA:$VSC_DATA" \
    -B "$GRP_SCRATCH:$GRP_SCRATCH" \
    "$CONTAINER" \
    bash "$RUN_SCRIPT"

echo ""
echo "BCI inference complete."