#!/bin/bash
#SBATCH --job-name=sr3_eval_mist
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

# eval_mist.sh — Evaluate SR3 predictions on all four MIST stains.
# Submit after infer_mist.sh completes for each stain.

set -euo pipefail

GRP_SCRATCH="/scratch/antwerpen/grp/ap_invilab_td_thesis"
OUT_BASE="$GRP_SCRATCH/diffusion-predictions/sr3"
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

echo "=== Dataset archive ==="
if [ ! -f "$VSC_SCRATCH/datasets/MIST.sqsh" ]; then
    echo "ERROR: MIST SquashFS archive not found: $VSC_SCRATCH/datasets/MIST.sqsh"
    exit 1
fi

# =========================================================
# EVALUATION — ALL MIST STAINS
# =========================================================

mkdir -p "$VSC_SCRATCH/datasets/MIST"

for stain in ER HER2 Ki67 PR; do

    stain_lower=$(echo "$stain" | tr '[:upper:]' '[:lower:]')
    PRED_DIR="$OUT_BASE/mist_${stain_lower}_test"
    GT_DIR="$VSC_SCRATCH/datasets/MIST/$stain/TrainValAB/valB"

    echo ""
    echo "========================================="
    echo "  Stain: $stain"
    echo "========================================="

    if [ ! -d "$PRED_DIR" ] || [ -z "$(find "$PRED_DIR" -maxdepth 1 -type f \( -name "*.png" -o -name "*.jpg" \) 2>/dev/null)" ]; then
        echo "  No predictions found in $PRED_DIR — skipping."
        continue
    fi
    echo "  Predictions : $(find "$PRED_DIR" -maxdepth 1 -type f \( -name "*.png" -o -name "*.jpg" \) | wc -l) images"

    srun apptainer exec --nv \
        -B "$VSC_SCRATCH/datasets/MIST.sqsh:$VSC_SCRATCH/datasets/MIST:image-src=/" \
        -B "$VSC_DATA:$VSC_DATA" \
        -B "$GRP_SCRATCH:$GRP_SCRATCH" \
        "$CONTAINER" \
        python "$EVAL_SCRIPT" \
            --pred         "$PRED_DIR" \
            --gt           "$GT_DIR" \
            --model_name   SR3_256 \
            --dataset_name "MIST_${stain}" \
            --split_name   test \
            --match_by     sort \
            --output       "$OUTPUT_CSV" \
            --device       cuda \
            --cellpose
    echo "  $stain done."

done

echo ""
echo "All MIST stains evaluation complete. Results appended to: $OUTPUT_CSV"