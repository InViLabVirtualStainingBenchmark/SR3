# SR3 - Virtual Staining Adaptation

Adaptation of the SR3 image-to-image diffusion model for paired virtual staining of histology images. Currently validated on BCI and MIST datasets; the infrastructure supports any paired histology dataset in the same format. All changes are infrastructure-only with no architectural modifications.

---

## 1. Changes from Original SR3

### `data.py` + `train.py`: `scale_factor = 1` and `nn.Identity()`

The original SR3 was a 4x super-resolution model. The label image is 4x larger than the condition image, which had two effects: crop coordinates on the label were multiplied by 4 (`data.py`), and the condition image was bicubic-upsampled 4x before being fed to the UNet (`train.py`). For virtual staining, input and target images are the same resolution, so both adjustments must be disabled. `scale_factor = 1` fixes the crop, and `nn.Identity()` replaces the upsample so the condition passes through unchanged.

### `data.py`: `val_ratio` parameter for train/val split

`TrainDataset` and `EvalDataset` accept a `val_ratio` parameter. When `val_ratio > 0`, both classes sort filenames deterministically and split on the last `val_ratio` fraction:

- `TrainDataset` uses the first `1 - val_ratio` fraction
- `EvalDataset` uses the last `val_ratio` fraction

Both train and val dataloaders point to the same source folder. The split is handled internally by sorted filename order, so there is no data leakage.

### `train.py`: dataset classes replaced + `val_ratio` wired through

The original code used DF2K-specific dataset classes hardwired to four separate path arguments. Replaced with the generic `TrainDataset` and `EvalDataset` classes which take two paths (condition + target), matching the BCI/MIST folder structure. `val_ratio` is read from settings and passed to both dataset constructors.

### `main.py`: model discovery loop fix

The model discovery loop scans all subdirectories and imports `<name>.model`. If a `data/` folder is present locally, it would try to import `data.model`, causing a `ModuleNotFoundError`. Added a check for the existence of `model.py` before importing.

### `main.py`: `len(device)` -> `len(devices)` bug fix

Pre-existing typo in the original repo. The `-mgpu` block declared a list called `devices` but referenced the undefined `device`, causing a `NameError` in multi-GPU mode.

### `main.py`: `-d` flag for dataset selection

Added a `-d` flag that accepts a dataset name (`BCI`, `MIST_ER`, `MIST_HER2`, `MIST_Ki67`, `MIST_PR`) and automatically sets all six path settings plus `val_ratio=0.1`, keeping paths out of the source code. Adding support for a new dataset requires extending this flag in `main.py`.

### `train.py`: best checkpoint saving by validation loss

`_store_train_env` now accepts a `v_loss` argument. When validation loss improves, the current state is saved to `best.pt` in addition to the regular `last.pt`. `_load_test_env` loads `best.pt` if it exists, otherwise falls back to `last.pt`. `best_v_loss` is initialised to `inf` at the start of each training run and is not persisted across resumes — if a run is resumed, `best.pt` reflects only the best checkpoint from that particular run segment.

### `metrics.py`: IS score chunk size guard

The IS score calculation splits test activations into chunks of `N // 10`. With fewer than 10 test images, this produces a chunk size of 0, causing a `RuntimeError`. Added `max(1, N // 10)` to allow the pipeline to run on small test sets.

### `data.py`: extension filter for image files

`TrainDataset` and `EvalDataset` now filter filenames by extension when listing images. The original code passed every filename returned by `listdir()` to `Image.open()`, which would crash on non-image files (e.g. `.DS_Store`, thumbnail cache files). Both classes now accept only `{.png, .jpg, .jpeg, .tif, .tiff, .bmp}`.

---

## 2. Train / Val / Test Split

| Split | BCI | MIST |
|---|---|---|
| Train (90%) | `HE/train` + `IHC/train` first 90% | `trainA` + `trainB` first 90% |
| Val (10%) | `HE/train` + `IHC/train` last 10% | `trainA` + `trainB` last 10% |
| Test | `HE/test` + `IHC/test` | `valA` + `valB` (all) |

Both train and val config sections point to the same source folder. The dataset class handles the split internally by sorted filename order.

**BCI** provides explicit train and test folders, so the test split is straightforward.

**MIST** does not provide a dedicated test folder. The dataset only has `trainA/trainB` and `valA/valB`. The `valA/valB` folders are therefore treated as the held-out test set and are not further subdivided.

---

## 3. Key Settings (`main.py`)

| Setting | Value | Original SR3 | Reason for change |
|---|---|---|---|
| `crop_size` | `256` | `64` | Original SR3 used 64x64 low-resolution crops corresponding to 256x256 targets under 4x super-resolution. For virtual staining, condition and target are the same resolution, so training uses 256x256 paired crops directly. |
| `val_ratio` | `0.1` (via `-d`) | N/A | 10% of training data held out for validation |
| `-a` channels | `56` | N/A | Base channel width. Use `32` for local smoke tests to reduce memory and runtime. |

All other parameters are kept at original SR3 values.

---

## 4. Training Data Structure

Images must be at least `crop_size x crop_size`. Filenames must match between condition and target folders. For the required cluster archive structure, see `hpc/INSTRUCTIONS.md` Step 1.4.

---

## 5. HPC Scripts

See `hpc/INSTRUCTIONS.md` for the full script list and execution order.

---

## 6. `inference.py`

Standalone inference script that saves predicted target images to disk. It must be run before evaluation, as the eval scripts compare these saved predictions against ground truth.

**Inference modes**

Three modes are available, controlled by `--chop_size`:

| Mode | Flag | Patch size | Default stride | Overlap |
|---|---|---|---|---|
| Matching crop | `--chop_size 256` | 256 × 256 | 224 px | 32 px |
| Larger patch | `--chop_size 512` | 512 × 512 | 448 px | 64 px |
| Full image | *(no flag)* | full H × W | — | — |

**`--chop_size 256`** — The model sees patches of the same size used during training (256 × 256 crops). No distribution shift between training and inference. Generates the most patches per image because the patch is smallest relative to the full image.

**`--chop_size 512` (stride 448)** — Larger patches provide wider context and result in fewer seam positions. The model was trained on 256-px crops so there is some distribution shift, but there are fewer patch boundaries to average over. Currently used for MIST inference.

**Full image (no `--chop_size`)** — The entire image is passed in a single forward pass. No patch boundaries or averaging at all. Requires sufficient GPU memory for the full image at inference resolution. Has the highest distribution shift relative to the 256-px training crop size. Currently used for BCI inference.

For both sliding window modes the default stride is `chop_size - chop_size // 8`, giving 12.5% overlap (32 px for `--chop_size 256`, 64 px for `--chop_size 512`). Overlapping regions are averaged.

**Usage:**
```bash
# Full image — single forward pass (no --chop_size)
python inference.py -m UNet -a 56 -d BCI

# Sliding window — 256×256 patches, stride 224 (matches training resolution)
python inference.py -m UNet -a 56 -d BCI --chop_size 256

# Sliding window — 512×512 patches, stride 448
python inference.py -m UNet -a 56 -d BCI --chop_size 512

# Custom output directory and checkpoint
python inference.py -m UNet -a 56 -d BCI --chop_size 512 -o /path/to/output --ckpt /path/to/best.pt

# Custom stride (e.g. 50% overlap instead of 12.5%)
python inference.py -m UNet -a 56 -d BCI --chop_size 512 --chop_stride 256

# Fewer sampling steps (trades output quality for speed)
python inference.py -m UNet -a 56 -d BCI --chop_size 512 --sample_steps 50

# Run on a specific GPU
python inference.py -m UNet -a 56 -d BCI --chop_size 512 --device cuda:1
```

**Arguments:**

| Argument | Default | Description |
|---|---|---|
| `-m` / `--model` | required | Model directory name (e.g. `UNet`) |
| `-a` / `--args` | required | Model base channel width (use `56` for full training, `32` for smoke test) |
| `-d` / `--dataset` | required | `BCI` \| `MIST_ER` \| `MIST_HER2` \| `MIST_Ki67` \| `MIST_PR` |
| `-o` / `--output` | `<model>/test_out_<dataset>` | Output directory for predicted images |
| `--ckpt` | `<model>/pnt_<dataset>/best.pt` | Checkpoint path |
| `--base` | `$VSC_SCRATCH/datasets` | Dataset base path (set automatically by cluster run scripts) |
| `--steps` | `2000` | Diffusion training steps (must match the value used during training) |
| `--sample_steps` | `100` | Reverse diffusion steps at inference time |
| `--batch_size` | `4` | Batch size for center-crop mode (ignored when `--chop_size` is set, which forces batch size 1) |
| `--workers` | `4` | DataLoader worker processes |
| `--crop_size` | `None` | Center crop applied to each image before a single forward pass. Only active when `--chop_size` is not set. When both flags are omitted, the full image is passed without any cropping. |
| `--chop_size` | `None` | Sliding window patch size. Accepts `256` (matches training) or `512` (larger context). When set, overrides `--crop_size`. When not set, falls back to single-pass mode: center-cropped if `--crop_size` is given, full image otherwise. |
| `--chop_stride` | `chop_size - chop_size // 8` | Sliding window stride. Default is 224 for `--chop_size 256` and 448 for `--chop_size 512`. Smaller values increase overlap and the number of patches. |
| `--device` | auto | Device override (e.g. `cuda`, `cuda:1`, `cpu`). Defaults to CUDA if available. |

On the cluster, `--base` and `--ckpt` are set automatically by the `run_infer_*.sh` scripts. The scripts also pass `-o` pointing to the group scratch predictions folder.

Predictions are written to `$GRP_SCRATCH/diffusion-predictions/sr3/`. Folder names are controlled by `RUN_SUFFIX`; see `hpc/INSTRUCTIONS.md` Section 5 for usage and default values.

---

## 7. Running

See `hpc/INSTRUCTIONS.md` for all cluster commands (training, inference, evaluation, resume).

---

## 8. Output Folders

```
UNet/
  log_[dataset]/      TensorBoard logs
  pnt_[dataset]/      Checkpoints (last.pt, old.pt, best.pt)
  states_[dataset]/   Test metric states
```

- `best.pt`: saved when validation loss improves
- `last.pt`: overwritten every epoch; used to resume training
- `old.pt`: the previous `last.pt`, kept as a one-step fallback in case `last.pt` is corrupted

Testing automatically uses `best.pt` if it exists, otherwise falls back to `last.pt`.

View TensorBoard:
```bash
tensorboard --logdir UNet/log_BCI
```

---

## 9. Environment

This section is for local development only. On the cluster, the container handles the environment.

- Python 3.9
- PyTorch 2.1.2 + CUDA 12.1

```bash
pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements_frozen.txt
```