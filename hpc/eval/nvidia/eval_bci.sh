#!/bin/bash
#SBATCH --job-name=sr3_eval_bci
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=60G
#SBATCH --time=24:00:00
#SBATCH -A ap_invilab_td_thesis
#SBATCH -p ampere_gpu
#SBATCH --gres=gpu:1
#SBATCH -o /data/antwerpen/212/vsc21211/projects/sr3/logs/%x.%j.out
#SBATCH -e /data/antwerpen/212/vsc21211/projects/sr3/logs/%x.%j.err

# eval_bci.sh — Evaluate SR3 predictions on BCI test set.
# Submit after infer_bci.sh completes.

set -euo pipefail

GRP_SCRATCH="/scratch/antwerpen/grp/ap_invilab_td_thesis"
: "${RUN_SUFFIX:=fullimg}"
: "${PRED_DIR:=$GRP_SCRATCH/diffusion-predictions/sr3/bci_${RUN_SUFFIX}}"
GT_DIR="$VSC_SCRATCH/datasets/BCI/IHC/test"
OUTPUT_CSV="$VSC_DATA/benchmark_results.csv"
EVAL_SCRIPT="$VSC_DATA/evaluate/evaluate.py"
CONTAINER="$VSC_SCRATCH/containers/evaluate_nvidia.sif"

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

echo "=== Eval script ==="
if [ ! -f "$EVAL_SCRIPT" ]; then
    echo "ERROR: evaluate.py not found at $EVAL_SCRIPT"
    exit 1
fi

echo "=== Predictions ==="
if [ ! -d "$PRED_DIR" ]; then
    echo "ERROR: Predictions folder not found: $PRED_DIR"
    echo "  Run infer_bci.sh first."
    exit 1
fi
echo "  $(find "$PRED_DIR" -maxdepth 1 -type f \( -name "*.png" -o -name "*.jpg" \) | wc -l) predicted images"

echo "=== Dataset archive ==="
if [ ! -f "$VSC_SCRATCH/datasets/BCI.sqsh" ]; then
    echo "ERROR: BCI SquashFS archive not found: $VSC_SCRATCH/datasets/BCI.sqsh"
    exit 1
fi

# =========================================================
# EVALUATION
# =========================================================

echo ""
echo "=== Starting BCI evaluation ==="

mkdir -p "$VSC_SCRATCH/datasets/BCI"

srun apptainer exec --nv \
    -B "$VSC_SCRATCH/datasets/BCI.sqsh:$VSC_SCRATCH/datasets/BCI:image-src=/" \
    -B "$VSC_DATA:$VSC_DATA" \
    -B "$GRP_SCRATCH:$GRP_SCRATCH" \
    "$CONTAINER" \
    python "$EVAL_SCRIPT" \
        --pred         "$PRED_DIR" \
        --gt           "$GT_DIR" \
        --model_name   SR3 \
        --dataset_name BCI \
        --split_name   test \
        --match_by     sort \
        --output       "$OUTPUT_CSV" \
        --device       cuda
echo ""
echo "BCI evaluation complete. Results appended to: $OUTPUT_CSV"