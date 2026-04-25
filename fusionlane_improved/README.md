# FusionLane Improved

Binary lane-marking segmentation in PyTorch — runs on TFRecord datasets, your own road images, dashcam video, or fully synthetic dummy data with no setup required.

---

## Files at a glance

| File | What it does |
|---|---|
| `dataset_pt.py` | Loads data from TFRecords / image folder / video / dummy; applies normalization and augmentation |
| `train_pt.py` | Trains the model with hybrid Cross-Entropy + Dice loss; saves `best_model.pth` |
| `infer_pt.py` | Runs inference on the same data the model was trained/tested on |
| `infer_media.py` | **Easy entry point** — runs on any road video or image folder you drop in |
| `requirements.txt` | Python dependencies |

---

## Quick start (no data needed)

```bash
# 1. Create and activate virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Mac / Linux

# 2. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 3. Train for 2 epochs on synthetic dummy data
python train_pt.py --epochs 2 --image_height 64 --image_width 64

# 4. Run inference
python infer_pt.py --image_height 64 --image_width 64

# 5. Check outputs
#    outputs/inference/raw/        - raw predictions
#    outputs/inference/cleaned/    - filtered predictions
#    outputs/inference/heatmap/    - confidence maps
#    outputs/inference/comparison/ - side-by-side panels
```

---

## Running on your own road video or images

Use `infer_media.py` — the dedicated script for real-world input.

### From a dashcam video

```bash
python infer_media.py --input path/to/dashcam.mp4 --model_path ./outputs/best_model.pth
```

Outputs:
- `outputs/inference/raw/`, `cleaned/`, `heatmap/`, `comparison/` — one PNG per frame
- `outputs/inference/lane_overlay.mp4` — original video with green lane overlay drawn on top

### From a folder of images

```bash
python infer_media.py --input path/to/road_images/ --model_path ./outputs/best_model.pth
```

Any mix of `.jpg`, `.png`, `.bmp`, `.tiff` inside the folder works. Images are sorted alphabetically.

### From a single image

```bash
python infer_media.py --input path/to/road.jpg --model_path ./outputs/best_model.pth
```

### Supported video formats
`.mp4`  `.avi`  `.mov`  `.mkv`  `.m4v`  `.wmv`  `.flv`

### Supported image formats
`.jpg`  `.jpeg`  `.png`  `.bmp`  `.tiff`  `.tif`  `.webp`

> **Note:** `infer_media.py` uses the model trained by `train_pt.py`.  
> Run `train_pt.py` at least once to create `outputs/best_model.pth` before running `infer_media.py`.

---

## Easy in-code configuration

Every script has a `CONFIG` block near the top — edit it once and just run `python script.py` without typing flags every time.

### `infer_media.py` — for your own video / images

Open [infer_media.py](infer_media.py) and find this section:

```python
# =============================================================================
# EASY CONFIG — edit these to run without typing command-line flags.
# =============================================================================
CONFIG = {
    # ── Input ────────────────────────────────────────────────────────────────
    "input":        "./my_road_video.mp4",   # <-- change this to your file/folder
    # ── Model ────────────────────────────────────────────────────────────────
    "model_path":   "./outputs/best_model.pth",
    # ── Output ───────────────────────────────────────────────────────────────
    "output_dir":   "./outputs/inference",
    # ── Image size (must match what you trained with) ─────────────────────────
    "image_height": 512,
    "image_width":  512,
    # ── Post-processing ───────────────────────────────────────────────────────
    "threshold":    0.50,   # P(lane) required to keep a pixel (lower = more lane shown)
    "min_blob":     100,    # smallest connected-component kept in pixels
    # ── Performance ───────────────────────────────────────────────────────────
    "batch_size":   4,
}
# =============================================================================
```

Then just run:
```bash
python infer_media.py
```

### `train_pt.py` — training parameters

Open [train_pt.py](train_pt.py) and find:

```python
CONFIG = {
    "data_dir":     "./data",    # TFRecord folder | image folder | video file
    "model_dir":    "./outputs",
    "epochs":       10,
    "batch_size":   4,
    "num_workers":  0,
    "image_height": 512,
    "image_width":  512,
    "loss_alpha":   0.5,         # 0 = pure Dice, 1 = pure CE
    "lane_weight":  2.0,         # class weight for lane pixels
    "bg_weight":    1.0,
    "lr":           1e-3,
}
```

### `infer_pt.py` — inference on TFRecord / dummy data

Open [infer_pt.py](infer_pt.py) and find:

```python
CONFIG = {
    "data_dir":     "./data",
    "model_path":   "./outputs/best_model.pth",
    "output_dir":   "./outputs/inference",
    "image_height": 512,
    "image_width":  512,
    "batch_size":   4,
    "threshold":    0.65,
    "min_blob":     200,
    "num_workers":  0,
}
```

> Command-line flags always take priority over CONFIG values, so you can mix both:
> ```bash
> python infer_media.py --threshold 0.3   # overrides CONFIG["threshold"]
> ```

---

## Project folder structure

```
fusionlane_improved/
├── dataset_pt.py          data loader (normalization, augmentation, all input modes)
├── train_pt.py            training loop + model definition
├── infer_pt.py            inference on TFRecord / dummy data
├── infer_media.py         inference on your own video or image folder
├── requirements.txt
├── README.md
│
├── data/                  place TFRecord files here (optional)
└── outputs/               created automatically
    ├── best_model.pth     saved by train_pt.py
    ├── logs/
    └── inference/
        ├── raw/           argmax predictions
        ├── cleaned/       confidence-filtered + morphologically cleaned
        ├── heatmap/       P(lane) confidence maps
        ├── comparison/    4-panel side-by-side
        └── lane_overlay.mp4   (only for video input via infer_media.py)
```

---

## How data input is auto-detected

You do not need to set an input mode flag. The dataset detects what you have:

| What's in `data_dir` / `--input` | Mode used | Labels available? |
|---|---|---|
| `train-0000X-of-00004.tfrecord` files | TFRecord | Yes (from annotation) |
| Image files (`.jpg` `.png` etc.) | Image folder | No (zeros) |
| Video file (`.mp4` `.avi` etc.) | Video | No (zeros) |
| Anything else / empty folder | Dummy data | Synthetic |

When no real labels are available (image or video mode), training loss is computed against zero labels — useful for fine-tuning on structure, but not for measuring accuracy.

---

## All CLI arguments

### `train_pt.py`

| Flag | Default | Description |
|---|---|---|
| `--data_dir` | `./data` | TFRecord folder, image folder, or video file |
| `--model_dir` | `./outputs` | Where `best_model.pth` is saved |
| `--epochs` | `10` | Training epochs |
| `--batch_size` | `4` | Must be divisible by 4 |
| `--num_workers` | `0` | DataLoader workers |
| `--image_height` | `512` | Resize height |
| `--image_width` | `512` | Resize width |
| `--loss_alpha` | `0.5` | CE fraction in hybrid loss |
| `--lane_weight` | `2.0` | CE class weight for lane |
| `--bg_weight` | `1.0` | CE class weight for background |
| `--lr` | `1e-3` | Adam learning rate |

### `infer_media.py`

| Flag | Default | Description |
|---|---|---|
| `--input` | `./my_road_video.mp4` | Video file, image folder, or single image |
| `--model_path` | `./outputs/best_model.pth` | Checkpoint from `train_pt.py` |
| `--output_dir` | `./outputs/inference` | Where outputs are saved |
| `--image_height` | `512` | Must match training height |
| `--image_width` | `512` | Must match training width |
| `--threshold` | `0.50` | P(lane) minimum to keep a pixel |
| `--min_blob` | `100` | Minimum blob size in pixels |
| `--batch_size` | `4` | Must be divisible by 4 |

### `infer_pt.py`

| Flag | Default | Description |
|---|---|---|
| `--data_dir` | `./data` | Same as training |
| `--model_path` | `./outputs/best_model.pth` | Checkpoint |
| `--output_dir` | `./outputs/inference` | Where outputs are saved |
| `--image_height` | `512` | Must match training height |
| `--image_width` | `512` | Must match training width |
| `--threshold` | `0.65` | P(lane) minimum |
| `--min_blob` | `200` | Minimum blob size |
| `--batch_size` | `4` | Must be divisible by 4 |

---

## Understanding the outputs

| Subfolder | Content | When to look at it |
|---|---|---|
| `raw/` | Pure argmax — every pixel the model thinks is lane | Baseline; shows false positives too |
| `cleaned/` | Only pixels with P(lane) > threshold, small blobs removed | Use this as your primary result |
| `heatmap/` | Jet-coloured confidence map (blue=uncertain, red=certain) | Helps pick the right `--threshold` |
| `comparison/` | 4-panel: original \| raw \| cleaned \| heatmap | Quick visual sanity check |
| `lane_overlay.mp4` | Original video + green lane drawn on top | Video only — final deliverable |

---

## Tuning tips

| Parameter | Too low | Too high |
|---|---|---|
| `--threshold` | Many false-positive pixels (noise shows as lane) | Real lanes disappear from `cleaned/` |
| `--min_blob` | Tiny dots remain in `cleaned/` | Dashed lanes get erased |
| `--loss_alpha` | Model ignores background (over-segments) | Dice loss has no effect |
| `--lane_weight` | Lane pixels consistently missed | Everything predicted as lane |
| `--lr` | Training very slow to converge | Loss becomes NaN |

---

## Connecting your real FusionLane model

The training script uses a lightweight `SimpleFusionLaneNet` by default. It also tries to import the full `FusionLaneModel` from `model_pt.py` if one is present:

```python
# In train_pt.py — build_model() function
def build_model(num_classes=2):
    try:
        from model_pt import FusionLaneModel
        return FusionLaneModel(num_classes=num_classes)
    except Exception:
        return SimpleFusionLaneNet(in_channels=4, num_classes=num_classes)
```

To use the full model, copy `model_pt.py` into `fusionlane_improved/`. Adjust `num_classes` to match your dataset (7 for the original FusionLane classes, 2 for binary lane/background).

---

## Requirements

```
torch
numpy
opencv-python
scipy
tqdm
pillow
```

TensorFlow is optional — only needed for reading TFRecord files. If it is not installed, the code automatically uses dummy data or your own images/video.

```bash
pip install -r requirements.txt
# Optionally, for TFRecord support:
pip install tensorflow
```

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError: tensorflow` | Install it separately or ignore it — code falls back to image/video/dummy input |
| `FileNotFoundError: best_model.pth` | Run `train_pt.py` first |
| `batch_size must be divisible by 4` | Use `--batch_size 4` or `8` or `12` |
| `cleaned/` is all black | Lower `--threshold` (try `0.3`) or lower `--min_blob` (try `10`) |
| `heatmap/` is all blue | Model is not confident — train for more epochs or with a lower learning rate |
| `loss is NaN` | Lower `--lr` (try `1e-4`) |
| Video input only loads 500 frames | Split the video into shorter clips, or increase the limit in `_load_video()` in `dataset_pt.py` |
