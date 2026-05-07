#!/bin/bash
# setup_project.sh
# Run once manually on the login node. Do NOT sbatch.
# Usage: bash setup_project.sh

set -euo pipefail

BASE_DIR="$VSC_DATA/projects/sr3"

echo "Creating project structure at: $BASE_DIR"

mkdir -p "$BASE_DIR"/{code,logs}
# code → cloned repos
# logs → slurm outputs

# Checkpoints live inside the repo under UNet/pnt_[dataset]/
# Results live inside the repo under UNet/log_[dataset]/

mkdir -p "$VSC_SCRATCH/containers"
# containers → apptainer .sif files

echo "Done. Next: bash clone_repo.sh"
