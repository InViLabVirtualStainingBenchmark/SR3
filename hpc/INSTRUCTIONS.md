# SR3 — Cluster Execution Plan

Complete reference for running SR3 on VSC Tier 2 Antwerp.
All scripts live in `hpc/` inside this repo.
Run all commands from the cluster login node unless stated otherwise. You can also browse files, check job status, and open a terminal through the portal at https://portal.hpc.uantwerpen.be/ without a local SSH client.

## Script inventory

| Script | Type | What it does |
|---|---|---|
| `setup_project.sh` | bash | Creates folder tree under `$VSC_DATA` |
| `clone_repo.sh` | bash | Clones the SR3 repo; pulls if it already exists |
| `train_bci.sh` | sbatch | Trains on BCI dataset (Apptainer container) |
| `run_sr3_bci.sh` | bash | Runs inside the container — called by `train_bci.sh` |
| `train_mist.sh` | sbatch | Trains on MIST stains (Apptainer container) |
| `run_sr3_mist.sh` | bash | Runs inside the container — called by `train_mist.sh` |

Training and testing run in the same job. After training completes, the script automatically runs the test phase on the held-out test set and logs PSNR, SSIM, IS, and FID to TensorBoard.

---

## Execution order

### 1. One-time setup

**Step 1.1. Add SSH key and connect**

Add your public key to your VSC account via the VSC account page.
Connect:

```bash
ssh <username>@login.hpc.uantwerpen.be
echo $VSC_DATA
echo $VSC_SCRATCH
```

Expected:
- `$VSC_DATA`    = `/data/antwerpen/<group>/<username>`
- `$VSC_SCRATCH` = `/scratch/antwerpen/<group>/<username>`

**Step 1.2. Create the project folder tree**

Upload `hpc/setup_project.sh` from the repo to your home directory on the cluster, then run it:

```bash
bash ~/setup_project.sh
```

**Step 1.3. Clone the repository**

Upload `hpc/clone_repo.sh` from the repo to your home directory on the cluster, then run it:

```bash
bash ~/clone_repo.sh
```

The repo is now available at `$VSC_DATA/projects/sr3/code/SR3/`.

**Step 1.4. Prepare and upload SquashFS archives**

The container mounts datasets as read-only SquashFS images for fast I/O. Only the `.sqsh` files are needed on the cluster — do not upload raw dataset directories.

Pack the datasets locally before uploading:

```bash
mksquashfs /path/to/BCI  BCI.sqsh  -noappend
mksquashfs /path/to/MIST MIST.sqsh -noappend
```

The directory structure inside the archives must be:

```
BCI.sqsh root:
    HE/train/    HE/test/
    IHC/train/   IHC/test/

MIST.sqsh root:
    ER/TrainValAB/{trainA,trainB,valA,valB}
    HER2/TrainValAB/...
    Ki67/TrainValAB/...
    PR/TrainValAB/...
```

Upload the archives using a file transfer tool (Cyberduck, FileZilla, WinSCP, scp, rsync):
- Destination: `$VSC_SCRATCH/datasets/BCI.sqsh` and `$VSC_SCRATCH/datasets/MIST.sqsh`

Verify:

```bash
unsquashfs -l $VSC_SCRATCH/datasets/BCI.sqsh  | head -10
unsquashfs -l $VSC_SCRATCH/datasets/MIST.sqsh | head -10
```

**Step 1.5. Build the Apptainer container**


Build the container locally (requires Apptainer installed), then upload the `.sif` to the cluster:

```bash
# On your local machine
apptainer build sr3_nvidia.sif sr3_nvidia.def
```

Upload to the cluster:
- Destination: `$VSC_SCRATCH/containers/sr3_nvidia.sif`

Verify on the cluster:

```bash
ls -lh $VSC_SCRATCH/containers/sr3_nvidia.sif
```

---

### 2. Smoke test (sbatch)

Run a short job before committing to full training. This confirms the container, dataset mount, and code all work together.

In `hpc/train_bci.sh`, temporarily set:

```bash
export ITERS=4
#SBATCH --time=00:30:00
```

Submit from the project root:

```bash
cd $VSC_DATA/projects/sr3/code/SR3
sbatch hpc/train_bci.sh
```

Pass criteria:
1. Log exits without a Python traceback.
2. Loss values are not NaN.
3. Checkpoint exists under `UNet/pnt_BCI/`.
4. GPU log has non-zero utilization entries.

After the smoke test passes, restore `ITERS=500000` and `--time=24:00:00` before full training.

---

### 3. Full training (sbatch)

Run from the project root on the cluster:

```bash
cd $VSC_DATA/projects/sr3/code/SR3

# BCI
sbatch hpc/train_bci.sh

# MIST — one job per stain, all four can run simultaneously
sbatch --job-name=sr3_mist_er   --export=ALL,STAIN=ER   hpc/train_mist.sh
sbatch --job-name=sr3_mist_her2 --export=ALL,STAIN=HER2 hpc/train_mist.sh
sbatch --job-name=sr3_mist_ki67 --export=ALL,STAIN=Ki67 hpc/train_mist.sh
sbatch --job-name=sr3_mist_pr   --export=ALL,STAIN=PR   hpc/train_mist.sh
```

Checkpoints are saved under `UNet/pnt_[dataset]/`:
- `last.pt` — most recent checkpoint (used for resume)
- `old.pt` — previous checkpoint (backup)
- `best.pt` — best validation loss checkpoint (used for testing)

**Resuming a job** — if the job hits the 24-hour wall before finishing, set `RESUME="true"` in the SLURM script and resubmit. Training resumes from `last.pt`.

```bash
# In train_bci.sh, change:
export RESUME="true"
```

---

### 4. Monitoring

Job status can also be checked from the VSC portal at https://portal.hpc.uantwerpen.be/ without using the command line.

```bash
# Check all running and queued jobs
squeue -u $USER

# Get detailed job info including estimated start time
scontrol show job <jobid>

# Watch a log file live
tail -f $VSC_DATA/projects/sr3/logs/sr3_bci.<jobid>.out

# Check GPU utilization during training
tail -5 $VSC_DATA/projects/sr3/logs/gpu_bci_<jobid>.csv

# Check checkpoints
ls -lh $VSC_DATA/projects/sr3/code/SR3/UNet/pnt_BCI/
```

---

## Issues

| Problem | Cause | Fix |
|---|---|---|
| Job killed before training completes | 24-hour wall time | Set `RESUME="true"` in SLURM script and resubmit |
