#!/bin/bash
# Runs inside the Apptainer container for SR3 MIST inference.
# Called by infer_mist.sh via: apptainer exec ... bash run_infer_mist.sh
# Variables exported from the SLURM script: REPO_DIR, LOG_DIR, STAIN, DATASET, SLURM_JOB_ID

set -euo pipefail

echo "=== Environment ==="
python3 --version
python3 -c "import torch; print('torch:', torch.__version__); print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0))"

nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,memory.total \
           --format=csv -l 5 \
    > "$LOG_DIR/gpu_infer_mist_${STAIN}_${SLURM_JOB_ID}.csv" & GPU_LOG_PID=$!

# =========================================================
# INFERENCE
# =========================================================

echo ""
echo "=== Starting MIST-${STAIN} inference ==="
echo "  dataset    : $DATASET"
echo "  input      : $VSC_SCRATCH/datasets/MIST/$STAIN/TrainValAB/valA"
echo "  output     : $OUT_DIR"
echo "  checkpoint : $REPO_DIR/UNet/pnt_${DATASET}/best.pt"

cd "$REPO_DIR"

python3 inference.py -m UNet -a 56 -d "$DATASET" -o "$OUT_DIR" --chop_size 256 --chop_stride 128

kill $GPU_LOG_PID 2>/dev/null || true

echo ""
echo "=== Output image count ==="
echo "  $STAIN : $(find "$OUT_DIR" -name "*.png" | wc -l) images written."
echo "GPU log : $LOG_DIR/gpu_infer_mist_${STAIN}_${SLURM_JOB_ID}.csv"
