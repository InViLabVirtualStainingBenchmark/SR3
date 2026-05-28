#!/bin/bash
#SBATCH --job-name=sr3_infer_mist
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

# infer_mist_arcturus.sh — MIST inference for SR3 on arcturus_gpu (AMD/ROCm).
# Override stain at submission: sbatch --export=ALL,STAIN=HER2 infer_mist_arcturus.sh
# Loads best.pt automatically from UNet/pnt_MIST_{STAIN}/.

set -euo pipefail

export REPO_DIR="$VSC_DATA/projects/sr3/code/SR3"
export LOG_DIR="$VSC_DATA/projects/sr3/logs"

# Stain to infer: ER | HER2 | Ki67 | PR
: "${STAIN:=ER}"
export STAIN
export DATASET="MIST_${STAIN}"

GRP_SCRATCH="/scratch/antwerpen/grp/ap_invilab_td_thesis"
stain_lower=$(echo "$STAIN" | tr '[:upper:]' '[:lower:]')
export OUT_DIR="$GRP_SCRATCH/diffusion-predictions/sr3/mist_${stain_lower}_test_1024"

CONTAINER="$VSC_SCRATCH/containers/sr3_rocm.sif"
RUN_SCRIPT="$REPO_DIR/hpc/run_infer_mist.sh"

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

echo "=== Checking checkpoint ==="
BEST_PT="$REPO_DIR/UNet/pnt_${DATASET}/best.pt"
if [ ! -f "$BEST_PT" ]; then
    echo "ERROR: best.pt not found: $BEST_PT"
    exit 1
fi
echo "  best.pt : $(du -h "$BEST_PT" | cut -f1)"

# =========================================================
# RUN
# =========================================================

mkdir -p "$VSC_SCRATCH/datasets/MIST"
mkdir -p "$OUT_DIR"

export MIOPEN_USER_DB_PATH=/tmp/miopen_${SLURM_JOB_ID}
mkdir -p "$MIOPEN_USER_DB_PATH"

srun apptainer exec --rocm \
    --env MIOPEN_USER_DB_PATH="$MIOPEN_USER_DB_PATH" \
    -B "$VSC_SCRATCH/datasets/MIST.sqsh:$VSC_SCRATCH/datasets/MIST:image-src=/" \
    -B "$VSC_DATA:$VSC_DATA" \
    -B "$GRP_SCRATCH:$GRP_SCRATCH" \
    "$CONTAINER" \
    bash "$RUN_SCRIPT"

echo ""
echo "MIST $STAIN inference complete."