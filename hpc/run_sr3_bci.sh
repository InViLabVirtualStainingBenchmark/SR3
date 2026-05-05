#!/bin/bash
# Runs inside the Apptainer container for SR3 BCI training + testing.
# Called by train_bci.sh via: apptainer exec ... bash run_sr3_bci.sh
# Variables exported from the SLURM script: REPO_DIR, LOG_DIR, ITERS, RESUME, SLURM_JOB_ID

set -euo pipefail

echo "=== Environment ==="
python3 --version
python3 -c "import torch; print('torch:', torch.__version__); print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0))"

# Start GPU logging in background
nvidia-smi --query-gpu=timestamp,index,utilization.gpu,utilization.memory,memory.used,memory.total \
           --format=csv -l 5 > "$LOG_DIR/gpu_bci_${SLURM_JOB_ID}.csv" &
GPU_LOG_PID=$!

echo ""
echo "Starting SR3 BCI training..."
echo "  iters  : $ITERS"
echo "  resume : ${RESUME:-false}"

cd "$REPO_DIR"
RESUME_ARG=""
if [ "${RESUME:-false}" = "true" ]; then
    RESUME_ARG="-r"
fi

python3 main.py \
    -m UNet \
    -a 56 \
    -d BCI \
    -i "$ITERS" \
    $RESUME_ARG

kill $GPU_LOG_PID 2>/dev/null || true

echo ""
echo "BCI training and testing complete."
echo "Checkpoints : $REPO_DIR/UNet/pnt_BCI/"
echo "Predictions : $REPO_DIR/UNet/preds_BCI/"
echo "GPU log     : $LOG_DIR/gpu_bci_${SLURM_JOB_ID}.csv"
