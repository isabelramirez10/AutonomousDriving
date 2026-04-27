# Multi-sweep BEV accumulation

Densify FusionLane's BEV inputs by stacking N consecutive LiDAR sweeps using ego-pose transforms. Static features (lane markings, curbs) accumulate; moving objects smear across frames and get per-pixel-diluted.

## What's in the box

| File | Purpose |
|:-----|:--------|
| `lidar_bev.py` | Core single-sweep projector: point cloud → `image.png` + `region.png`. |
| `multi_sweep_bev.py` | Accumulator that uses `lidar_bev.py` to rasterize stacks of N frames. |

## Prerequisites

You need **sequences** with per-frame ego poses. Specifically:

1. A folder of point cloud files (`.bin`, `.pcd`, or `.npy`), **ordered by name**. File *i* corresponds to frame *i*.
2. A poses file in KITTI-odometry format — one line per frame, 12 space-separated floats = top 3 rows of the 4×4 `frame → map` transform.

If your sensor stack doesn't give you poses directly, the typical alternatives are:
- IMU/GPS fusion (the usual one).
- Registration against a prior HD map.
- Frame-to-frame ICP as a fallback. Produces drift but works for short windows.

A single point cloud without poses **cannot** be accumulated — by definition, accumulation needs multiple frames.

## Usage

### Single-sweep conversion (for inference against the existing checkpoint)

```bash
python lidar_bev.py your_scan.bin out_dir/
```

Writes `out_dir/your_scan.png` (3-channel: intensity / height / density) and `out_dir/your_scan_region.png` (1-channel ROI mask). Default BEV is 321×321 at 0.1 m/px, vehicle-centered on a 32.1×32.1 m area. Every parameter is overridable — see `--help`.

### Multi-sweep accumulation (to regenerate training data denser)

```bash
python multi_sweep_bev.py \
    --pointclouds_dir /data/sequence_07/pointclouds \
    --poses_file      /data/sequence_07/poses.txt \
    --out_dir         /data/bev_accumulated \
    --num_sweeps 3 \
    --stride 1
```

Writes one `.png` + `_region.png` pair per *target frame*, built from `num_sweeps` consecutive clouds transformed into the target frame. The output feeds directly into the existing `create_my_record.py` pipeline:

```bash
python create_my_record.py \
    --image_folder   /data/bev_accumulated \
    --region_folder  /data/bev_accumulated \
    --semantic_segmentation_folder /data/labels \
    --output_dir     /data/tfrecord_accum
```

Then retrain:

```bash
python train_pt.py --data_dir /data/tfrecord_accum --model_dir /model_accum
```

## Parameter tuning

**`--num_sweeps`**. 3 is the standard starting point and matches what most LiDAR segmentation papers use. Larger K densifies more but also admits more pose error and more smearing on moving objects. 5 is a reasonable upper bound.

**`--stride`**. `1` = consecutive frames. Use `>1` if your frame rate is very high (say 20 Hz) and consecutive frames are essentially redundant; a stride of 2–3 gives more spatial variety across the window.

**`--center last` vs `--center middle`**. With `last`, the target is the newest frame — each accumulated frame incorporates the past. This is the right choice for causal / online use. `middle` makes the target the center frame and accumulates both past and future, giving the most symmetric coverage; only use this for offline batch preprocessing.

**BEV extent flags** (`--x_min` / `--x_max` / `--y_min` / `--y_max` / `--resolution`). These must match whatever the model was trained on. The defaults are a plausible guess; the original paper doesn't publish the exact values. Tune by visual comparison against a known-good output from `output_60epochs/`.

## What the knob actually does, in numbers

On synthetic sequences the test suite in this repo verifies:

- A static feature (dashed lane line) goes from 23 filled pixels in a single frame to 64 in a 3-sweep accumulation — **2.78× coverage**.
- A moving object (vehicle drifting 3 m/frame) with 30 LiDAR hits per frame goes from 20 filled pixels (single frame) to 71 (accumulated) — the point density per pixel drops, effectively de-emphasizing it.

## Caveats

- **Must match training distribution.** If you train the model on single-sweep BEVs and accumulate at inference time only, you'll hurt performance — the model has never seen accumulated BEVs. Retrain on the accumulated data. Conversely, if you train on accumulated data, don't feed single-sweep BEVs at inference.
- **Ego-motion error compounds.** Accumulation is only as good as your poses. Bad poses → ghosted lane markings → worse-than-nothing. If your pose error is >~20 cm over the accumulation window, keep `--num_sweeps` small (2) or reconsider accumulation entirely.
- **Per-frame filtering still matters.** The z-range filter (`--z_min`/`--z_max`) drops above-ground clutter *before* accumulation, which is usually what you want. Leaving it wide will accumulate tree canopies and overhead structures into the BEV.
- **Moving objects don't disappear, they dim.** If your scene has dense traffic, the per-pixel dilution may not be enough. The model will still see faint traces. Class-weighted CE already handles this reasonably, but it's worth inspecting a few outputs visually.
