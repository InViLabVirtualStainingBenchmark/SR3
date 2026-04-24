# SR3 Virtual Staining — Documentation

Adaptation of the SR3 image-to-image diffusion model for the virtual staining benchmark.
Supports BCI and MIST datasets. All changes are infrastructure-only — no architectural modifications.

---

## Changes from original SR3

### `data.py` + `train.py` — `scale_factor = 1` and `nn.Identity()`
The original SR3 was a 4× super-resolution model — the label image is 4× larger than the condition image. This had two effects: crop coordinates on the label were multiplied by 4 (`data.py`), and the condition image was bicubic-upsampled 4× before being fed to the UNet (`train.py`). BCI and MIST are same-resolution paired datasets, so both must be set to 1×. `scale_factor = 1` fixes the crop, and `nn.Identity()` replaces the upsample so the condition passes through unchanged.

### `train.py` — dataset classes replaced
The original code used DF2K-specific dataset classes (`DF2KTrainDataset`, `DIV2KValDataset`, `Flickr2KTestDataset`) hardwired to four separate path arguments. Replaced with the generic `TrainDataset` and `EvalDataset` classes (already present in `data.py`) which take two paths (condition + target), matching the BCI/MIST folder structure.

### `main.py` — `model.py` check in discovery loop
The model discovery loop scans all subdirectories and imports `<name>.model` from each one. If a `data/` folder is present locally, the loop would try to import `data.model`, causing a `ModuleNotFoundError`. Added a check for the existence of `model.py` before importing so only actual model directories are loaded.

### `main.py` — `len(device)` → `len(devices)` bug fix
Pre-existing typo in the original repo. The `-mgpu` block declared a list called `devices` but then referenced the undefined variable `device`, causing a `NameError` whenever multi-GPU mode was used.

### `main.py` — `-d` flag for dataset selection
The original code had dataset paths hardcoded as empty strings requiring manual edits for each run. Added a `-d` flag that accepts a dataset name (`BCI`, `MIST_ER`, `MIST_HER2`, `MIST_Ki67`, `MIST_PR`) and automatically sets all six path settings, keeping paths out of the source code.

### `metrics.py` — IS score chunk size guard
The IS score calculation splits test activations into chunks of `N // 10`. With fewer than 10 test images, this produces a chunk size of 0, causing a `RuntimeError` in `torch.split`. Added `max(1, N // 10)` to guarantee a minimum chunk size of 1, allowing the pipeline to run on small test sets without affecting behavior on full datasets.

---

## Environment

- Python 3.9
- PyTorch 2.1.2 + CUDA 12.1
- torch==2.1.2+cu121
- torchvision==0.16.2+cu121
- numpy==1.26.4

### Installation

```bash
pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements_frozen.txt
```

---

## Data

### Smoke test

The smoke test requires 8 train / 4 val / 4 test image pairs per dataset, placed under `data/`:

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

Use `prepare_data.py` with the default counts (8/4/4) to populate this structure from your source dataset before running the smoke test. Use `prepare_data.py` to copy images from the source dataset into the expected folder structure:

```bash
# BCI
python prepare_data.py \
  --he_src  /path/to/BCI_dataset/HE/train \
  --ihc_src /path/to/BCI_dataset/IHC/train \
  --dst     data/BCI \
  --train N --val N --test N  # defaults: 8 / 4 / 4 for the smoke test

# MIST (repeat for each marker: ER, HER2, Ki67, PR)
python prepare_data.py \
  --he_src  /path/to/MIST/HER2/TrainValAB/trainA \
  --ihc_src /path/to/MIST/HER2/TrainValAB/trainB \
  --dst     data/MIST/HER2 \
  --train N --val N --test N  # defaults: 8 / 4 / 4 for the smoke test
```

The script renames files to zero-padded integers using the source file extension (e.g. `0000.png`, `0001.png`, ... for BCI; `0000.jpg`, `0001.jpg`, ... for MIST) ensuring sorted pairs always align between `he/` and `ihc/` folders.

### Source dataset layout

**BCI:**
```
BCI_dataset/
  HE/train/    HE/test/
  IHC/train/   IHC/test/
```

**MIST:**
```
MIST/
  ER/TrainValAB/trainA    ER/TrainValAB/trainB
  HER2/TrainValAB/trainA  HER2/TrainValAB/trainB
  Ki67/TrainValAB/trainA  Ki67/TrainValAB/trainB
  PR/TrainValAB/trainA    PR/TrainValAB/trainB
```

### Minimum image counts

| Split | Minimum | Reason |
|---|---|---|
| train | 4 | batch size |
| val | 4 | `report_img_idx` covers indices 0–3 |
| test | 4 | IS score guard (`max(1, N//10)`) |

---

## Training & Testing

### Commands

```bash
# Train + test
python main.py -m UNet -a [channels] -d [dataset]

# Test only (requires existing checkpoint)
python main.py -m UNet -a [channels] -d [dataset] -t

```

### Dataset flag `-d`

| Flag | Dataset |
|---|---|
| `-d BCI` | Breast Cancer IHC |
| `-d MIST_ER` | MIST — ER marker |
| `-d MIST_HER2` | MIST — HER2 marker |
| `-d MIST_Ki67` | MIST — Ki67 marker |
| `-d MIST_PR` | MIST — PR marker |

### Example — smoke test (all modalities)

```bash
python main.py -m UNet -a 32 -d BCI
python main.py -m UNet -a 32 -d MIST_ER
python main.py -m UNet -a 32 -d MIST_HER2
python main.py -m UNet -a 32 -d MIST_Ki67
python main.py -m UNet -a 32 -d MIST_PR
```

### Example — full training (HPC)

```bash
python main.py -m UNet -a 56 -d BCI
```

---

## Output folders

Each run writes to model-namespaced folders:

```
UNet/
  log_[dataset]/      TensorBoard logs
  pnt_[dataset]/      Checkpoints (last.pt, old.pt)
  states_[dataset]/   Test metric states
```

View TensorBoard:
```bash
tensorboard --logdir UNet/log_BCI
```

---

## Key settings (main.py)

Current values are configured for the smoke test. Update these in `main.py` before full training on HPC.

| Setting | Smoke test (current) | Full training |
|---|---|---|
| `iters` | 4 | 500000 |
| `sample_steps` | 10 | 100 |
| `report_img_per` | 1 | 10 |
| `report_img_idx` | `[0,1,2,3]` | e.g. `[0,50,100,150]` |
| `crop_size` | 64 | 64 |
| `-a` channels | 32 | 56 |