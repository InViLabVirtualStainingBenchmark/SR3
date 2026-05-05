#!/bin/bash
# Clones or updates the SR3 repository into $VSC_DATA/projects/sr3/code/SR3.
# Run once from the cluster login node.
set -euo pipefail

REPO_URL="https://github.com/InViLabVirtualStainingBenchmark/SR3.git"
REPO_DIR="$VSC_DATA/projects/sr3/code/SR3"

if [ -d "$REPO_DIR/.git" ]; then
    echo "Repo exists at $REPO_DIR, pulling latest..."
    cd "$REPO_DIR"
    git pull
else
    echo "Cloning into $REPO_DIR..."
    git clone "$REPO_URL" "$REPO_DIR"
fi

echo ""
echo "Done: $REPO_DIR"
