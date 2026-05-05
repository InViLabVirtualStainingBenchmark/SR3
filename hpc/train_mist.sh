#!/bin/bash
#SBATCH --job-name=sr3_mist
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=60G
#SBATCH --time=20:00:00
#SBATCH -A ap_invilab_td_thesis
#SBATCH -p ampere_gpu
#SBATCH --gres=gpu:1
#SBATCH -o /data/antwerpen/<group>/<username>/projects/sr3/logs/%x.%j.out
#SBATCH -e /data/antwerpen/<group>/<username>/projects/sr3/logs/%x.%j.err

set -euo pipefail

# =========================================================
# USER SETTINGS
# =========================================================

export REPO_DIR="$VSC_DATA/projects/sr3/code/SR3"
export LOG_DIR="$VSC_DATA/projects/sr3/logs"

# Total gradient steps.
# At batch=4 and ~4153 train images (1038 steps/epoch): 500000 iters ≈ 482 epochs.
export ITERS=500000

# Stain to train: ER | HER2 | Ki67 | PR
# Override at submission with: sbatch --export=ALL,STAIN=HER2 train_mist.sh
: "${STAIN:=ER}"
export STAIN
export DATASET="MIST_${STAIN}"

# Set to "true" to resume from UNet/pnt_MIST_{STAIN}/last.pt.
# Leave as "false" for a fresh run.
export RESUME="false"

CONTAINER="$VSC_SCRATCH/containers/sr3_nvidia.sif"
RUN_SCRIPT="$REPO_DIR/hpc/run_sr3_mist.sh"

# NOTE: Before submitting full training, update main.py:
#   sample_steps: 10  →  100
#   report_img_per: 1  →  10
#   report_img_idx: [0,1,2,3]  →  [0,50,100,150]

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
if [ ! -f "$VSC_SCRATCH/datasets/MIST.sqsh" ]; then
    echo "ERROR: MIST SquashFS archive not found: $VSC_SCRATCH/datasets/MIST.sqsh"
    exit 1
fi
echo "  MIST.sqsh : $(du -h "$VSC_SCRATCH/datasets/MIST.sqsh" | cut -f1)"

echo "=== Stain: $STAIN ==="

# =========================================================
# RUN
# =========================================================

mkdir -p "$VSC_SCRATCH/datasets/MIST"

srun apptainer exec --nv \
    -B "$VSC_SCRATCH/datasets/MIST.sqsh:$VSC_SCRATCH/datasets/MIST:image-src=/" \
    -B "$VSC_DATA:$VSC_DATA" \
    "$CONTAINER" \
    bash "$RUN_SCRIPT"

echo ""
echo "MIST $STAIN training complete."
