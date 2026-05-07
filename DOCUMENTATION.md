# SR3 Virtual Staining — Documentation

Adaptation of the SR3 image-to-image diffusion model for paired HE→IHC virtual staining on BCI and MIST datasets. All changes are infrastructure-only — no architectural modifications.

---

## 1. Changes from Original SR3

### `data.py` + `train.py` — `scale_factor = 1` and `nn.Identity()`

The original SR3 was a 4× super-resolution model — the label image is 4× larger than the condition image. This had two effects: crop coordinates on the label were multiplied by 4 (`data.py`), and the condition image was bicubic-upsampled 4× before being fed to the UNet (`train.py`). BCI and MIST are same-resolution paired datasets, so both must be 1×. `scale_factor = 1` fixes the crop, and `nn.Identity()` replaces the upsample so the condition passes through unchanged.

### `data.py` — `val_ratio` parameter for train/val split

`TrainDataset` and `EvalDataset` accept a `val_ratio` parameter. When `val_ratio > 0`, both classes sort filenames deterministically and split on the last `val_ratio` fraction:

- `TrainDataset` uses the first `1 - val_ratio` fraction
- `EvalDataset` uses the last `val_ratio` fraction (when `split='val'`)

Both train and val dataloaders point to the same source folder — the split is handled internally by sorted filename order, so there is no data leakage.

### `train.py` — dataset classes replaced + `val_ratio` wired through

The original code used DF2K-specific dataset classes hardwired to four separate path arguments. Replaced with the generic `TrainDataset` and `EvalDataset` classes which take two paths (condition + target), matching the BCI/MIST folder structure. `val_ratio` is read from settings and passed to both dataset constructors.

### `main.py` — model discovery loop fix

The model discovery loop scans all subdirectories and imports `<name>.model`. If a `data/` folder is present locally, it would try to import `data.model`, causing a `ModuleNotFoundError`. Added a check for the existence of `model.py` before importing.

### `main.py` — `len(device)` → `len(devices)` bug fix

Pre-existing typo in the original repo. The `-mgpu` block declared a list called `devices` but referenced the undefined `device`, causing a `NameError` in multi-GPU mode.

### `main.py` — `-d` flag for dataset selection

Added a `-d` flag that accepts a dataset name (`BCI`, `MIST_ER`, `MIST_HER2`, `MIST_Ki67`, `MIST_PR`) and automatically sets all six path settings plus `val_ratio=0.1`, keeping paths out of the source code.

### `metrics.py` — IS score chunk size guard

The IS score calculation splits test activations into chunks of `N // 10`. With fewer than 10 test images, this produces a chunk size of 0, causing a `RuntimeError`. Added `max(1, N // 10)` to allow the pipeline to run on small test sets.

---

## 2. Train / Val / Test Split

| Split | BCI | MIST |
|---|---|---|
| Train (90%) | `HE/train` + `IHC/train` first 90% | `trainA` + `trainB` first 90% |
| Val (10%) | `HE/train` + `IHC/train` last 10% | `trainA` + `trainB` last 10% |
| Test | `HE/test` + `IHC/test` | `valA` + `valB` (all) |

Both train and val config sections point to the same source folder — the dataset class handles the split internally by sorted filename order. For MIST, `valA/valB` are used as the held-out test set and cannot be further split (paired images of the same tissue).

---

## 3. Key Settings (`main.py`)

| Setting | Value            | Original SR3 | Reason for change |
|---|------------------|---|---|
| `crop_size` | `256`            | `64` | Original used 64×64 LR patches for SR; virtual staining needs full image |
| `val_ratio` | `0.1` (via `-d`) | — | 10% of training data held out for validation |
| `-a` channels | `56`             | — | Full-capacity UNet for full training |

All other parameters are kept at original SR3 values.

`crop_size=256` — the original 64 was sized for 64×64 LR patches in the SR task. For virtual staining, 256×256 crops capture enough tissue context and match SinSR's training resolution.
---

## 4. Training Data Structure

Images must be at least `crop_size × crop_size`. Filenames must match between HE and IHC folders.

**Local (smoke test):**
```
data/
  BCI/
    train/he/   train/ihc/
    val/he/     val/ihc/
    test/he/    test/ihc/
  MIST/
    ER/    train/he/  train/ihc/  val/he/  val/ihc/  test/he/  test/ihc/
    HER2/  train/he/  train/ihc/  val/he/  val/ihc/  test/he/  test/ihc/
    Ki67/  train/he/  train/ihc/  val/he/  val/ihc/  test/he/  test/ihc/
    PR/    train/he/  train/ihc/  val/he/  val/ihc/  test/he/  test/ihc/
```

**Cluster** — datasets are mounted as read-only SquashFS archives. The training scripts mount them automatically; `main.py` sets paths from `$VSC_SCRATCH/datasets/` when `-d` is passed:
```
$VSC_SCRATCH/datasets/
    BCI.sqsh   → mounted at $VSC_SCRATCH/datasets/BCI/
        HE/train/    HE/test/
        IHC/train/   IHC/test/
    MIST.sqsh  → mounted at $VSC_SCRATCH/datasets/MIST/
        ER/TrainValAB/trainA   trainB   valA   valB
        HER2/TrainValAB/...
        Ki67/TrainValAB/...
        PR/TrainValAB/...
```

Use `prepare_data.py` to populate the local smoke test structure from your source dataset:

```bash
# BCI
python prepare_data.py \
  --he_src  /path/to/BCI/HE/train \
  --ihc_src /path/to/BCI/IHC/train \
  --dst     data/BCI \
  --train N --val N --test N  # used: 8 / 4 / 4 for the smoke test

# MIST (repeat for each marker: ER, HER2, Ki67, PR)
python prepare_data.py \
  --he_src  /path/to/MIST/ER/TrainValAB/trainA \
  --ihc_src /path/to/MIST/ER/TrainValAB/trainB \
  --dst     data/MIST/ER \
  --train N --val N --test N  # used: 8 / 4 / 4 for the smoke test
```

---

## 5. HPC Scripts

Scripts in `hpc/` for running on the VSC cluster. Training uses an Apptainer container (`sr3_nvidia.def` in the repo root).

| Script | How to run | Purpose |
|---|---|---|
| `train_bci.sh` | `sbatch train_bci.sh` | Train on BCI dataset |
| `run_sr3_bci.sh` | called by `train_bci.sh` | Runs training inside the container |
| `train_mist.sh` | `sbatch --job-name=sr3_mist_er --export=ALL,STAIN=ER train_mist.sh` | Train one MIST stain per job; STAIN = ER \| HER2 \| Ki67 \| PR |
| `run_sr3_mist.sh` | called by `train_mist.sh` | Runs training inside the container |

Checkpoints are saved under `UNet/pnt_[dataset]/`. SLURM and GPU logs are saved under `$VSC_DATA/projects/sr3/logs/`.

---

## 6. Running

**Local smoke test:**
```bash
python main.py -m UNet -a 32 -d BCI
python main.py -m UNet -a 32 -d MIST_ER
```

**Cluster (full training):**
```bash
cd $VSC_DATA/projects/sinsr/code/SR3

sbatch hpc/train_bci.sh

sbatch --job-name=sr3_mist_er   --export=ALL,STAIN=ER   hpc/train_mist.sh
sbatch --job-name=sr3_mist_her2 --export=ALL,STAIN=HER2 hpc/train_mist.sh
sbatch --job-name=sr3_mist_ki67 --export=ALL,STAIN=Ki67 hpc/train_mist.sh
sbatch --job-name=sr3_mist_pr   --export=ALL,STAIN=PR   hpc/train_mist.sh
```

**Test only (requires existing checkpoint):**
```bash
python main.py -m UNet -a 56 -d BCI -t
```

---

## 7. Output Folders

```
UNet/
  log_[dataset]/      TensorBoard logs
  pnt_[dataset]/      Checkpoints (last.pt, old.pt, best.pt)
  states_[dataset]/   Test metric states
```

`best.pt` is saved when validation loss improves. Testing automatically uses `best.pt` if it exists, otherwise falls back to `last.pt`.

View TensorBoard:
```bash
tensorboard --logdir UNet/log_BCI
```

---

## 8. Environment

- Python 3.9
- PyTorch 2.1.2 + CUDA 12.1

```bash
pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements_frozen.txt
```
