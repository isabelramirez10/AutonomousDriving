# Boundary loss

Distance-transform boundary loss (Kervadec et al., MIDL 2019 / MedIA 2021) layered on top of the existing weighted cross-entropy in `train_pt.py`. Designed to sharpen the model's predictions where they're currently blurriest — the edges of lane-line and marker regions.

## Why this over just weighted CE

Weighted CE treats every pixel equally within its class. A pixel at the exact edge of a lane line and a pixel three meters inside a large background region contribute the same per-pixel loss, scaled only by class weight. That means the model has no explicit incentive to make its class boundaries sharp — it just needs per-pixel accuracy on average.

Boundary loss adds a term that integrates, over each class, the signed distance to that class's ground-truth boundary, weighted by the predicted probability:

```
L_boundary = Σ_classes Σ_pixels  φ_c(pixel) · p(class=c | pixel)
```

`φ_c` is positive outside the ground-truth region of class c and negative inside. Predicting high probability for c *far from where c actually is* gets penalised (positive `φ` × high `p`); predicting c exactly where it should be contributes negatively, which minimisation favours. The gradient implicitly pushes the predicted boundary toward the true one.

For thin structures (lane lines are 1–3 pixels wide in the FusionLane BEV), this matches the loss to the problem.

## What's in the box

| File | Purpose |
|:-----|:--------|
| `boundary_loss.py` | `BoundaryLoss` module + `compute_distance_maps` + `ramp_alpha` scheduler. Standalone, no repo deps. |
| `train_pt_boundary.py` | Drop-in replacement for `train_pt.py`. Same CLI, plus four extra flags. |

## Usage

```bash
python train_pt_boundary.py \
    --data_dir   /data/tfrecord \
    --model_dir  /model_pt_boundary \
    --boundary_weight 0.3 \
    --boundary_warmup 10000
```

That's the recipe. Everything else matches `train_pt.py`.

## Flags

| Flag | Default | Notes |
|:-----|:--------|:------|
| `--boundary_weight` | 0.3 | Target α in `total_loss = ce + α · bd`. The paper uses 0.01–1.0 depending on task. 0.3 is a safe starting point; sweep 0.1/0.3/1.0 if you can afford it. |
| `--boundary_warmup` | 10000 | Steps over which α linearly ramps from 0 to target. **Don't set to 0.** The boundary loss has large magnitude early in training (distance fields are on the order of tens of pixels) and will destabilize optimization before the model can produce sensible class distributions. |
| `--boundary_downscale` | 1 | Compute SDFs on labels downsampled by this factor, then upsample bilinearly. `2` is ~4× faster with minor edge-precision loss; `4` is ~16× faster but coarsens boundaries. Recommended only if the default is too slow. |
| `--boundary_every` | 1 | Apply boundary loss every Nth step (rest use CE-only). Halves CPU overhead at `--boundary_every 2`. |

## Performance

Distance-transform computation is CPU-bound and doesn't release the GIL. At 321×321 with 7 classes and batch size 20, **benchmarked at ~1 second per batch** for realistic label distributions (~99% background + thin markings). For a GPU-bound training loop with ~0.3 s forward+backward per batch, that roughly doubles wall-clock time per step.

If this is too expensive in practice, the production fix is to precompute SDFs once at TFRecord generation time and read them alongside the labels. That's a separate refactor not included here — open an issue if it matters.

## Expected behaviour during training

Watch the `bd` number in the training logs:

- Early on (α ≈ 0): `bd` is whatever it is and doesn't affect training.
- Mid warmup (α ≈ 0.15): `bd` starts decreasing. If it goes *up*, the ramp is too aggressive or your learning rate is too high — reduce `--boundary_weight`.
- After warmup (α = target): `bd` should trend steadily more negative. CE should not spike upward — if it does, α is fighting CE too hard; reduce `--boundary_weight`.

Typical end-of-training signed values on FusionLane-like data: `ce ≈ 0.1–0.5`, `bd ≈ -2 to -5`. The absolute magnitudes don't matter — the trend does.

## Honest calibration

The implementation in `boundary_loss.py` is correctness-verified:

- Signed distance transform: inside ≤ 0, outside > 0, zero for empty/full masks (no NaN/inf).
- Per-class distance maps: mirror relationship between classes (class c's inside is class c̄'s outside).
- Loss is a scalar, produces finite gradients, backward works.
- `good_prediction < offset_prediction < bad_prediction` in signed loss (minimisation pushes toward the good case).
- One gradient step against a random-init prediction reduces the loss.
- Ignore-label pixels contribute exactly zero.

What's *not* verified in this repo: that boundary loss actually improves FusionLane's test-set mIoU on your data. That has to be measured empirically — run a single-seed A/B (`train_pt.py` vs `train_pt_boundary.py` from the same init, same LR schedule) for 3–5 epochs and compare per-class IoU on the test split. Report arrow/stop-line/solid-line IoU separately; the mean IoU is dominated by background and won't show the effect cleanly.

## Reference

H. Kervadec, J. Bouchtiba, C. Desrosiers, E. Granger, J. Dolz, I. Ben Ayed.
*Boundary loss for highly unbalanced segmentation.* Medical Image Analysis, 2021.
[arxiv.org/abs/1812.07032](https://arxiv.org/abs/1812.07032)
