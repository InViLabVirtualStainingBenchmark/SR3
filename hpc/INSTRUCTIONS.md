# SR3 — Cluster Instructions

Complete reference for running SR3 on VSC Tier 2 Antwerp.
All scripts live in `hpc/` inside this repo.
Run all commands from the cluster login node unless stated otherwise. You can also browse files, check job status, and open a terminal through the portal at https://portal.hpc.uantwerpen.be/ without a local SSH client.

For code changes, inference modes, argument details, and local usage, see `DOCUMENTATION.md`.

## Script inventory

| Script | Type | What it does |
|---|---|---|
| `setup_project.sh` | bash | Creates folder tree under `$VSC_DATA` |
| `clone_repo.sh` | bash | Clones the SR3 repo; pulls if it already exists |
| `train_bci.sh` | sbatch | Trains on BCI dataset (Apptainer container) |
| `run_sr3_bci.sh` | bash | Runs inside the container — called by `train_bci.sh` |
| `train_mist.sh` | sbatch | Trains on MIST stains (Apptainer container) |
| `run_sr3_mist.sh` | bash | Runs inside the container — called by `train_mist.sh` |
| `infer_bci.sh` | sbatch | Runs inference on the BCI test set (NVIDIA) |
| `infer_bci_arcturus.sh` | sbatch | Runs inference on the BCI test set (AMD) |
| `run_infer_bci.sh` | bash | Runs `inference.py` inside the container — called by `infer_bci.sh` / `infer_bci_arcturus.sh` |
| `infer_mist.sh` | sbatch | Runs inference for one MIST stain; STAIN = ER \| HER2 \| Ki67 \| PR (NVIDIA) |
| `infer_mist_arcturus.sh` | sbatch | Runs inference for one MIST stain (AMD) |
| `run_infer_mist.sh` | bash | Runs `inference.py` inside the container — called by `infer_mist.sh` / `infer_mist_arcturus.sh` |
| `eval_bci.sh` | sbatch | Evaluates BCI predictions on the default GPU partition |
| `eval_mist.sh` | sbatch | Evaluates all four MIST stain predictions on the default GPU partition |
| `eval_bci_arcturus.sh` | sbatch | Evaluates BCI predictions on the alternative GPU partition |
| `eval_mist_arcturus.sh` | sbatch | Evaluates all four MIST stain predictions on the alternative GPU partition |

Training and testing run in the same job. After training completes, the script automatically runs the test phase on the held-out test set and logs PSNR, SSIM, IS, and FID to TensorBoard.

Inference and evaluation are separate jobs submitted after training. The training container uses NVIDIA or AMD depending on which cluster partition you target; the eval scripts follow the same pattern.

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

Two SR3 containers are provided: `sr3_nvidia.def` for NVIDIA GPUs (`ampere_gpu`) and `sr3_rocm.def` for AMD GPUs (`arcturus_gpu`). Build whichever partition you intend to use; both can coexist on the cluster.

Build locally (requires Apptainer installed), then upload the `.sif` to the cluster:

```bash
# On your local machine
apptainer build sr3_nvidia.sif sr3_nvidia.def   # NVIDIA
apptainer build sr3_rocm.sif   sr3_rocm.def     # AMD (optional)
```

Upload to the cluster:
- Destination: `$VSC_SCRATCH/containers/`

Verify on the cluster:

```bash
ls -lh $VSC_SCRATCH/containers/
```

---

### 2. Smoke test (sbatch)

Run a short job before committing to full training. This confirms the container, dataset mount, and code all work together.

In `hpc/train/train_bci.sh`, temporarily set:

```bash
export ITERS=4
#SBATCH --time=00:30:00
```

Submit from the project root:

```bash
cd $VSC_DATA/projects/sr3/code/SR3
sbatch hpc/train/train_bci.sh
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
sbatch hpc/train/train_bci.sh

# MIST — one job per stain, all four can run simultaneously
sbatch --job-name=sr3_mist_er   --export=ALL,STAIN=ER   hpc/train/train_mist.sh
sbatch --job-name=sr3_mist_her2 --export=ALL,STAIN=HER2 hpc/train/train_mist.sh
sbatch --job-name=sr3_mist_ki67 --export=ALL,STAIN=Ki67 hpc/train/train_mist.sh
sbatch --job-name=sr3_mist_pr   --export=ALL,STAIN=PR   hpc/train/train_mist.sh
```

Checkpoints are saved under `UNet/pnt_[dataset]/`. See `DOCUMENTATION.md` Section 8 for checkpoint details.

**Resuming a job** — if the job hits the 24-hour wall before finishing, pass `RESUME=true` at submission time. Training resumes from `last.pt`.

```bash
sbatch --export=ALL,RESUME=true hpc/train/train_bci.sh
sbatch --job-name=sr3_mist_er --export=ALL,STAIN=ER,RESUME=true hpc/train/train_mist.sh
```

Note: `best_v_loss` is not persisted across resumes. After a resume, `best.pt` will be overwritten whenever validation loss on the new run segment improves, even if that loss is worse than the pre-resume best. If the pre-resume `best.pt` is the intended inference checkpoint, copy it before resubmitting.

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

### 5. Inference (sbatch)

Inference must be run after training completes. It saves predicted target images to disk so the evaluation step can compare them against ground truth.

Each inference job runs `inference.py` inside the training container. BCI uses full-image inference (no `--chop_size`); MIST uses `--chop_size 512` with stride 448. The run scripts set all paths automatically.

Two GPU partitions are available. Use the default (`ampere_gpu`) when the queue is short; switch to the alternative (`arcturus_gpu`) when the default queue is long.

```bash
cd $VSC_DATA/projects/sr3/code/SR3

# Default GPU partition (ampere_gpu, NVIDIA)
sbatch hpc/infer/nvidia/infer_bci.sh

sbatch --job-name=sr3_infer_mist_er   --export=ALL,STAIN=ER   hpc/infer/nvidia/infer_mist.sh
sbatch --job-name=sr3_infer_mist_her2 --export=ALL,STAIN=HER2 hpc/infer/nvidia/infer_mist.sh
sbatch --job-name=sr3_infer_mist_ki67 --export=ALL,STAIN=Ki67 hpc/infer/nvidia/infer_mist.sh
sbatch --job-name=sr3_infer_mist_pr   --export=ALL,STAIN=PR   hpc/infer/nvidia/infer_mist.sh

# Alternative GPU partition (arcturus_gpu, AMD) — use when default queue is long
sbatch hpc/infer/amd/infer_bci_arcturus.sh

sbatch --job-name=sr3_infer_mist_er   --export=ALL,STAIN=ER   hpc/infer/amd/infer_mist_arcturus.sh
sbatch --job-name=sr3_infer_mist_her2 --export=ALL,STAIN=HER2 hpc/infer/amd/infer_mist_arcturus.sh
sbatch --job-name=sr3_infer_mist_ki67 --export=ALL,STAIN=Ki67 hpc/infer/amd/infer_mist_arcturus.sh
sbatch --job-name=sr3_infer_mist_pr   --export=ALL,STAIN=PR   hpc/infer/amd/infer_mist_arcturus.sh
```

Predictions are written to the group scratch folder under `diffusion-predictions/sr3/` (e.g. `bci_fullimg/`, `mist_er_fullimg/`). See DOCUMENTATION.md Section 6 for full output paths.

**Using a different output folder**

All inference and eval scripts use a `RUN_SUFFIX` variable to construct the output folder name:
- BCI folders are named `bci_<RUN_SUFFIX>` (default: `bci_fullimg`)
- MIST folders are named `mist_<stain>_<RUN_SUFFIX>` (default: `mist_er_fullimg`, etc.)

To run with a different setting and save to a new folder, pass `RUN_SUFFIX` at submission time. Use the same value for inference and eval so both point to the same folder:

```bash
# BCI — inference and eval with a custom suffix
sbatch --export=ALL,RUN_SUFFIX=chop256           hpc/infer/nvidia/infer_bci.sh
sbatch --export=ALL,RUN_SUFFIX=chop256           hpc/eval/nvidia/eval_bci.sh

# MIST — one inference job per stain; eval covers all four stains in one job
sbatch --export=ALL,STAIN=ER,RUN_SUFFIX=chop256   hpc/infer/nvidia/infer_mist.sh
sbatch --export=ALL,STAIN=HER2,RUN_SUFFIX=chop256 hpc/infer/nvidia/infer_mist.sh
sbatch --export=ALL,STAIN=Ki67,RUN_SUFFIX=chop256 hpc/infer/nvidia/infer_mist.sh
sbatch --export=ALL,STAIN=PR,RUN_SUFFIX=chop256   hpc/infer/nvidia/infer_mist.sh
sbatch --export=ALL,RUN_SUFFIX=chop256            hpc/eval/nvidia/eval_mist.sh
```

To change the inference mode itself (e.g. switch from full-image to `--chop_size 256`), edit `hpc/infer/run_infer_bci.sh` or `hpc/infer/run_infer_mist.sh` before submitting.

---

### 6. Evaluation (sbatch)

Evaluation computes LPIPS and other image quality metrics by comparing the saved predictions against ground truth. It must be run after inference. CellPose cell-level metrics are not enabled in the current scripts for speed; add `--cellpose` to the `evaluate.py` call to include them.

Two GPU partitions are available. Use the default when the queue is short; switch to the alternative when the default queue is long.

**Prerequisites (one-time):**

Build and upload the evaluation containers. The NVIDIA definition is in the evaluate repo's `hpc_jobs/` folder; the ROCm definition is in `hpc/evaluate_rocm.def` in this repo. `cluster_plan_container.md` in the evaluate repo covers the NVIDIA build steps in detail; the ROCm build follows the same pattern:

```bash
# NVIDIA (for ampere_gpu) — build from evaluate repo
cd ~/projects/evaluate/hpc_jobs
apptainer build evaluate_nvidia.sif evaluate_nvidia.def

# AMD/ROCm (for arcturus_gpu) — build from SR3 hpc/ folder
cd ~/projects/sr3/code/SR3/hpc
apptainer build evaluate_rocm.sif evaluate_rocm.def
```

Upload both to `$VSC_SCRATCH/containers/`. Only build the variant(s) you intend to use.

Pre-download LPIPS weights on the login node before the first eval job — see `cluster_plan_container.md` Step B3. CellPose weights are only needed if you re-enable `--cellpose`.

```bash
cd $VSC_DATA/projects/sr3/code/SR3

# Default GPU partition
sbatch hpc/eval/nvidia/eval_bci.sh
sbatch hpc/eval/nvidia/eval_mist.sh

# Alternative GPU partition (use when default queue is long)
sbatch hpc/eval/amd/eval_bci_arcturus.sh
sbatch hpc/eval/amd/eval_mist_arcturus.sh
```

Results are appended to `$VSC_DATA/benchmark_results.csv`.

---

## Issues

| Problem | Cause | Fix |
|---|---|---|
| Job killed before training completes | 24-hour wall time | Pass `RESUME=true` at submission: `sbatch --export=ALL,RESUME=true hpc/train/train_bci.sh` |
| `miopenStatusInternalError` in eval job on `arcturus_gpu` | MIOpen writes its kernel cache to the network home filesystem which does not support SQLite file locking | Already handled in `eval_bci_arcturus.sh` and `eval_mist_arcturus.sh` via `MIOPEN_USER_DB_PATH=/tmp/miopen_${SLURM_JOB_ID}` |
| Eval job cancelled due to time limit; `WARNING: channels deprecated in v4.0.1+` | CellPose v4.0.1+ runs too slowly (four MIST stains sequentially) and exceeds the 24-hour wall time | `--cellpose` removed from all four eval scripts (`eval_bci.sh`, `eval_bci_arcturus.sh`, `eval_mist.sh`, `eval_mist_arcturus.sh`) |
