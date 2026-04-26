# FusionLane Improved - Training and Inference Guide

## Overview

This guide covers the full pipeline from raw data to a trained camera+LiDAR fusion model.

```
Data sources (any combination)
  KITTI Road (recommended) ---> prepare_kitti.py
  TuSimple                 ---> prepare_data.py  --->  data/train/ + data/val/
  CULane                   ---> prepare_data.py
  CVAT (custom annotation) ---> prepare_data.py

                                                       train_pt.py
                                                            |
                                                   outputs/best_model.pth
                                                            |
                          +-----------------------------+---+---+
                          |                             |       |
                  eval_fusion.py               infer_media.py   infer_baseline.py
               (val set, GT metrics)         (any video/image) (Hough, no training)
```

---

## Files

### Core pipeline

| File | Purpose |
|------|---------|
| `model_pt.py` | ResNet-18 U-Net model with ImageNet pretrained weights |
| `train_pt.py` | Training loop - uses `model_pt.py` automatically |
| `dataset_pt.py` | Dataset loader - TFRecord / paired folder / video / dummy |
| `infer_media.py` | Deep-learning inference on any video or image input |
| `infer_pt.py` | Inference on labelled TFRecord data with GT mIoU evaluation |

### Data preparation

| File | Purpose |
|------|---------|
| `prepare_kitti.py` | Downloads KITTI Road (camera + Velodyne LiDAR), reprojects point clouds, splits 80/20 |
| `prepare_data.py` | Converts TuSimple / CULane / CVAT annotations to paired training format |

### Evaluation, testing, and analysis

| File | Purpose |
|------|---------|
| `eval_fusion.py` | Side-by-side fusion vs camera-only evaluation with GT metrics |
| `run_edge_case_tests.py` | 49-test edge case suite for all pipeline components |
| `infer_baseline.py` | Classical Canny + Hough-line detector (no training required) |
| `analyze_results.py` | Cross-run statistical analysis from metrics CSVs |
| `stats_report.py` | Full statistics table for training log and inference metrics |
| `compare_outputs.py` | Side-by-side comparison grid of reference and improved outputs |

---

## Quick Start - Baseline (no training required)

Run immediately on any dashcam video using classical computer vision:

```bash
python infer_baseline.py --input "path/to/video.mp4" --output_dir ./outputs/baseline
```

What it does:
- Detects edges with Canny in a trapezoidal road region of interest
- Finds line segments with Probabilistic Hough Transform
- Separates segments into left and right lanes by slope sign
- Fits a single extended line per side and draws a coloured road fill

Outputs:
```
outputs/baseline/
  overlay/         frame with blue (left) and yellow (right) lane lines
  edges/           Canny edge map clipped to ROI
  comparison/      original | edges | overlay  (side-by-side)
  lane_overlay.mp4
  metrics.csv      per-frame: hough_lines, left_detected, right_detected
```

Tuning parameters:

| Flag | Default | Effect |
|------|---------|--------|
| `--roi_top` | 0.55 | How far up the ROI trapezoid starts (0=top, 1=bottom). Lower for highways. |
| `--canny_low` / `--canny_high` | 50 / 150 | Edge sensitivity. Raise both if too much noise. |
| `--hough_threshold` | 50 | Minimum votes for a line. Raise to reduce false detections. |
| `--hough_min_length` | 80 | Minimum segment length in pixels. |
| `--alpha` | 0.4 | Overlay transparency (0=invisible, 1=opaque). |

---

## Pretrained Model - ResNet-18 Backbone

`model_pt.py` provides `FusionLaneModel`, a U-Net built on a ResNet-18 backbone pretrained on
ImageNet-1K (1.2 million images, 1000 classes).

The file is auto-detected by `train_pt.py`:

```python
# train_pt.py -- build_model()
try:
    from model_pt import FusionLaneModel   # used if present
    model = FusionLaneModel(num_classes=2)
except Exception:
    model = SimpleFusionLaneNet(...)       # fallback (~400K params)
```

On the first run, approximately 45 MB of ResNet-18 weights download automatically from
https://download.pytorch.org/models/resnet18-f37072fd.pth and are cached in
~/.cache/torch/hub/checkpoints/.

Verify the model loads correctly:

```bash
python model_pt.py
```

Expected output:
```
[FusionLaneModel] Loading ImageNet pretrained ResNet-18 weights
[FusionLaneModel] Pretrained weights loaded.
Total parameters : 14,952,066
Input  shape : [2, 4, 512, 512]
Output shape : [2, 2, 512, 512]
Architecture test passed.
```

---

## Data Preparation

### Option A - KITTI Road (recommended - includes real LiDAR)

KITTI Road provides camera images, Velodyne LiDAR point clouds, and GT lane masks.
It is the recommended training source because it includes real LiDAR data for the
fusion channel. `prepare_kitti.py` handles the full download and conversion automatically.

```bash
python prepare_kitti.py \
  --kitti_root ./kitti_data \
  --output_dir ./data \
  --val_split  0.20 \
  --seed       42
```

Download size: approximately 1.5 GB total (471 MB images + 1.1 GB LiDAR).
Download source: http://www.cvlibs.net/datasets/kitti/eval_road.php (downloaded automatically).

After completion:
```
data/
  train/  images/ (232)  masks/ (232)  depths/ (232)   <- real LiDAR depth maps
  val/    images/  (57)  masks/  (57)  depths/  (57)
```

### Option B - TuSimple (highway, approximately 6 K labelled frames)

Download from: https://github.com/TuSimple/tusimple-benchmark

Expected directory structure:
```
TuSimple/
  label_data_0313.json
  label_data_0531.json
  label_data_0601.json
  test_label.json
  clips/
    0313-1/
      6040/
        20.jpg ...
```

Run:
```bash
python prepare_data.py --tusimple_dir "C:/path/to/TuSimple" --output_dir ./data
```

The script reads each JSON annotation, loads the matching image, and draws thick white
polylines (default `--lane_thickness 12`) on a black binary mask.

### Option C - CULane (city + highway + night + rain, approximately 133 K frames)

Download from: https://xingangpan.github.io/projects/CULane.html

Expected structure:
```
CULane/
  driver_23_30frame/
  driver_161_90frame/
  driver_182_30frame/
  laneseg_label_w16/       <- segmentation masks (pixel values 0-4)
    driver_23_30frame/ ...
  list/
    train_gt.txt
    val.txt
```

Run:
```bash
python prepare_data.py --culane_dir "C:/path/to/CULane" --output_dir ./data
```

`train_gt.txt` format per line: `/img_rel_path /mask_rel_path 0 1 0 1`

Mask pixel values: 0=background, 1-4=lane markings, 255=ignore region. The script
converts these to binary (0 or 255). Falls back to polyline drawing if `laneseg_label_w16/`
is absent.

### Option D - CVAT (custom dashcam footage)

1. Extract frames from your video:
   ```bash
   ffmpeg -i "path/to/video.mp4" -vf fps=5 frames/%04d.jpg
   ```
   Aim for 300-500 frames minimum; 1000+ gives reliably better results.

2. Annotate at https://app.cvat.ai:
   - Create Project -> Add label: `lane` (type: Polygon or Polyline)
   - Create Task -> upload your extracted frames
   - Draw polygons or polylines on every visible lane marking
   - Export -> Segmentation Masks 1.1 (ZIP file)

3. Unzip the export:
   ```
   cvat_export/
     SegmentationClass/   frame_000001.png ...   <- coloured PNG masks
     images/default/      frame_000001.jpg ...   <- original frames
     labelmap.txt                                <- name to RGB colour map
   ```

4. Run:
   ```bash
   python prepare_data.py --cvat_dir "C:/path/to/cvat_export" --output_dir ./data
   ```

### Combining all sources

```bash
python prepare_data.py \
  --tusimple_dir "C:/path/to/TuSimple" \
  --culane_dir   "C:/path/to/CULane" \
  --cvat_dir     "C:/path/to/cvat_export" \
  --output_dir   ./data \
  --val_split    0.1 \
  --lane_thickness 12
```

Output structure (same regardless of which sources are used):
```
data/
  train/
    images/   <source>_000000.jpg ...
    masks/    <source>_000000.png ...
  val/
    images/   ...
    masks/    ...
```

Note: KITTI preparation via `prepare_kitti.py` also creates a `depths/` subfolder
containing real LiDAR depth maps. The other sources do not include LiDAR.

---

## Training

```bash
python train_pt.py \
  --data_dir    ./data \
  --epochs      50 \
  --batch_size  4 \
  --patience    10 \
  --lr          3e-4 \
  --scheduler   plateau \
  --image_height 256 \
  --image_width  256
```

Expected mIoU progression with real labelled data:

| Epoch range | Typical val mIoU | Notes |
|-------------|-----------------|-------|
| 1-3 | 0.30-0.50 | Backbone features activating |
| 5-10 | 0.55-0.70 | Lane shapes forming |
| 15-25 | 0.70-0.85 | Fine-tuned, stable |
| 50 | 0.82-0.92 | Converged (dataset-dependent) |

KITTI Road actual results (232 train / 57 val, 80/20 split):

| Epoch | Val mIoU | LR |
|-------|----------|----|
| 1 | 0.8136 | 3.00e-04 |
| 7 | 0.9086 | 3.00e-04 |
| 26 | **0.9405** | 7.50e-05 - best checkpoint |
| 36 | - | Early stopping (patience=10) |

Training outputs:
```
outputs/
  best_model.pth        checkpoint saved whenever val mIoU improves
  logs/
    training_log.csv    epoch, train_loss, val_loss, val_miou, lr
```

Key hyperparameters:

| Flag | Recommended | Notes |
|------|-------------|-------|
| `--lr` | `3e-4` | Lower than default (1e-3) to avoid disrupting pretrained weights |
| `--epochs` | `50` | Early stopping will cut this short if mIoU stagnates |
| `--patience` | `10` | Stops after 10 epochs with no mIoU improvement |
| `--scheduler` | `plateau` | Halves LR when mIoU plateaus |
| `--loss_alpha` | `0.5` | 50% Cross-Entropy + 50% Dice |
| `--lane_weight` | `4.0` | Upweights minority lane class (raise if recall is low) |
| `--image_height` | `256` | Match this at inference time for comparable mIoU |
| `--image_width` | `256` | Match this at inference time |

---

## Fusion Evaluation (Camera + LiDAR vs Camera-Only)

After training on KITTI, run `eval_fusion.py` to measure how much the LiDAR channel
contributes relative to camera-only mode on the validation split:

```bash
python eval_fusion.py \
  --data_dir     ./data \
  --model_path   ./outputs/best_model.pth \
  --output_dir   ./outputs/eval_fusion \
  --image_height 256 \
  --image_width  256 \
  --threshold    0.50
```

Results on KITTI val (57 samples):

| Metric | Camera-only | Fusion (Camera + LiDAR) | Delta |
|--------|-------------|------------------------|-------|
| mIoU | 0.9313 | 0.9375 | +0.0063 |
| Dice (F1) | 0.9327 | 0.9385 | +0.0058 |
| Precision | 0.9098 | 0.9273 | +0.0175 |
| Recall | 0.9721 | 0.9650 | -0.0071 |

Fusion was better on 43 of 57 samples (75%). The precision gain means fewer false
detections in regions where no lane exists - directly relevant to autonomous lane-keeping.

---

## Inference (Deep Learning)

```bash
python infer_media.py \
  --input      "path/to/video.mp4" \
  --output_dir ./outputs/inference_run1 \
  --image_height 512 \
  --image_width  512 \
  --threshold  0.50 \
  --min_blob   100
```

Note: the model was trained at 256x256. Running inference at 512x512 is valid and
often improves fine detail in the output mask.

With ground-truth masks for accuracy metrics:
```bash
python infer_media.py \
  --input      "path/to/video.mp4" \
  --gt_dir     ./data/val/masks \
  --output_dir ./outputs/inference_run1
```

With all pipeline improvements:
```bash
python infer_media.py \
  --input              "path/to/video.mp4" \
  --output_dir         ./outputs/inference_run1_improved \
  --image_height       512 \
  --image_width        512 \
  --use_clahe \
  --temporal_alpha     0.60 \
  --adaptive_threshold \
  --fit_lanes
```

Outputs:
```
outputs/inference_run1/
  raw/             argmax predictions (white=lane, black=background)
  cleaned/         confidence-filtered and morphologically cleaned mask
  heatmap/         P(lane) confidence map (blue=low, red=high)
  comparison/      4-panel: original | raw | cleaned | heatmap
  kitti_seg/       KITTI-style coloured segmentation (red bg, magenta lane)
  fitted/          polynomial curve fits (when --fit_lanes is set)
  lane_overlay.mp4
  metrics.csv
```

---

## Inference (Baseline - Classical CV)

```bash
python infer_baseline.py \
  --input      "path/to/video.mp4" \
  --output_dir ./outputs/baseline_run1

python infer_baseline.py \
  --input      "path/to/video2.mp4" \
  --output_dir ./outputs/baseline_run2
```

---

## Edge Case Testing

Run the full test suite to verify there are no regressions after any code change:

```bash
python run_edge_case_tests.py
```

Expected result: `EDGE CASE TEST RESULTS: 49/49 passed`

Test categories: preprocessing, EMA smoothing, adaptive Otsu threshold, clean mask,
ROI masking, polynomial fitting, KITTI seg colourmap, dataset loader, fusion metrics,
model forward pass (256x256 and 512x512).

---

## Results Summary

### Baseline (Hough lines, no training)

| Video | Frames | Left detected | Right detected | Avg Hough segments |
|-------|--------|--------------|----------------|--------------------|
| input.mp4 | 311 | 14.8% | 0.0% | 17.9 |
| input (2).mp4 | 222 | 94.6% | 93.2% | 79.4 |

The right lane in input.mp4 is never detected by the Hough baseline (0%), while the
deep learning model still detects lanes in 30.9% of frames with 0.856 mean confidence.

### Deep learning model (KITTI-trained, threshold=0.50)

| Video | Lane presence | Avg coverage | Mean confidence | Temporal IoU |
|-------|--------------|--------------|-----------------|-------------|
| input.mp4 | 30.9% | 1.17% | 0.856 | 0.728 |
| input (2).mp4 | 99.5% | 6.26% | 0.910 | 0.576 |

### Deep learning model with pipeline improvements

| Video | Lane presence | Avg coverage | Mean confidence | Temporal IoU |
|-------|--------------|--------------|-----------------|-------------|
| input.mp4 | 37.3% | 1.67% | 0.791 | **0.901** (+23.8%) |
| input (2).mp4 | 100.0% | 8.79% | 0.763 | **0.718** (+24.6%) |

---

## Troubleshooting

**best_model.pth not found**
Run `train_pt.py` first to produce a checkpoint.

**Baseline detects no lanes or very few**
Try `--roi_top 0.45` (start ROI higher) or `--hough_threshold 30` (lower threshold).
Roads with dashed markings only may need `--hough_max_gap 100`.

**Training mIoU stays near 0.0**
Verify your masks have white pixels - open one in an image viewer and confirm.
Check that mask filenames in `data/train/masks/` match image filenames in `data/train/images/`.
Try `--lane_weight 5.0` if lanes cover less than 5% of pixels.

**Out of memory during training**
Lower `--batch_size` to 2, or reduce `--image_height` and `--image_width` to 128.

**Inference is slow (CPU only)**
Install a CUDA-enabled PyTorch build from https://pytorch.org/get-started/locally/
GPU inference is typically 20-50x faster than CPU for this model size.

**eval_fusion.py reports lower mIoU than training log**
Ensure `--image_height` and `--image_width` match the values used during training.
Using 512x512 for eval when trained at 256x256 will give different (not lower, just
different) results because the model sees the scene at a different spatial scale.

**Edge case tests fail**
Run `pip install -r requirements.txt` and confirm Python 3.9 or newer is in use.
