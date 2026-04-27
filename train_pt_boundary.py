"""
train_pt_boundary.py — FusionLane training with weighted CE + boundary loss.

Drop-in replacement for train_pt.py. Adds the Kervadec boundary loss on top
of the existing weighted cross-entropy, with a linear warmup on the boundary
weight α:

    total_loss = ce_loss  +  α(step) * boundary_loss

α ramps linearly from 0 to --boundary_weight over --boundary_warmup steps.
The warmup exists because boundary loss is unstable when the model has not
yet learned to produce reasonable class distributions — the distance field
is large in magnitude and will dominate early gradients if applied cold.

Performance notes
-----------------
Distance-transform computation is CPU-bound and not GIL-releasing, so it
runs serially per sample on the main thread. At 321×321 with 7 classes the
benchmarked cost is ~1 s per batch of 20. On a typical GPU-bound training
loop this roughly doubles per-batch wall time. Options if that's a problem:

  • --boundary_downscale 2       computes SDF on a half-res label and
                                  bilinearly upsamples; ~4× faster with
                                  minor loss of edge precision.
  • --boundary_every 2           runs boundary loss on every 2nd batch
                                  (the other batches use CE-only). Halves
                                  overhead at the cost of a sparser signal.

For production you'd precompute SDFs at TFRecord generation time and read
them alongside the labels; that's a separate refactor.
"""

from __future__ import annotations

import argparse
import os
import shutil
from random import randint

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from model_pt       import FusionLaneModel
from dataset_pt     import FusionLaneDataset
from boundary_loss  import BoundaryLoss, compute_distance_maps, ramp_alpha
# Reuse existing helpers from train_pt so we don't duplicate them.
from train_pt       import (
    weighted_cross_entropy, compute_metrics, poly_lr,
    save_checkpoint, load_checkpoint, _CLASS_WEIGHTS, _NUM_TRAIN,
)


# ---------------------------------------------------------------------------
# Argparse
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser()
# --- identical to train_pt.py ---
parser.add_argument('--data_dir',        type=str,   default='./tfrecord/')
parser.add_argument('--model_dir',       type=str,   default='./model_pt_boundary/')
parser.add_argument('--train_epochs',    type=int,   default=120)
parser.add_argument('--epochs_per_eval', type=int,   default=1)
parser.add_argument('--batch_size',      type=int,   default=20)
parser.add_argument('--initial_lr',      type=float, default=1e-5)
parser.add_argument('--end_lr',          type=float, default=5e-7)
parser.add_argument('--max_iter',        type=int,   default=100000)
parser.add_argument('--weight_decay',    type=float, default=2e-4)
parser.add_argument('--num_classes',     type=int,   default=7)
parser.add_argument('--ignore_label',    type=int,   default=20)
parser.add_argument('--clean_model_dir', action='store_true')
# --- new boundary-loss flags ---
parser.add_argument('--boundary_weight',    type=float, default=0.3,
                    help='Target α for boundary loss (default: 0.3)')
parser.add_argument('--boundary_warmup',    type=int,   default=10000,
                    help='Steps to ramp α from 0 to target (default: 10000; '
                         '0 disables warmup, full weight from step 0)')
parser.add_argument('--boundary_downscale', type=int,   default=1,
                    help='Compute SDF on labels downsampled by this factor '
                         '(1 = full resolution, 2 = half, etc.)')
parser.add_argument('--boundary_every',     type=int,   default=1,
                    help='Only compute boundary loss every Nth step '
                         '(1 = every step)')


# ---------------------------------------------------------------------------
# Distance-map helper honouring the downscale flag
# ---------------------------------------------------------------------------

def _distance_maps_for_batch(
    labels: torch.Tensor,
    num_classes: int,
    ignore_label: int,
    downscale: int,
) -> torch.Tensor:
    """Compute distance maps, optionally on a downsampled label for speed.

    If downscale > 1:
      1. Nearest-neighbour downsample labels by `downscale`.
      2. Compute per-class SDFs on the smaller label.
      3. Multiply by `downscale` to keep SDF values in the full-res pixel unit.
      4. Bilinearly upsample SDFs back to full resolution.
    """
    if downscale <= 1:
        return compute_distance_maps(labels, num_classes, ignore_label)

    B, H, W = labels.shape
    lbl_small = F.interpolate(
        labels.unsqueeze(1).float(), scale_factor=1.0 / downscale,
        mode='nearest',
    ).long().squeeze(1)
    sdf_small = compute_distance_maps(lbl_small, num_classes, ignore_label)
    # Rescale distances to original pixel units.
    sdf_small = sdf_small * downscale
    # Upsample back to (H, W).
    return F.interpolate(sdf_small, size=(H, W), mode='bilinear',
                         align_corners=False)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    FLAGS = parser.parse_args()
    assert FLAGS.batch_size % 4 == 0, \
        f"batch_size must be divisible by 4, got {FLAGS.batch_size}"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device : {device}")
    if device.type == 'cuda':
        print(f"GPU    : {torch.cuda.get_device_name(0)}")
    print(f"Boundary: target α = {FLAGS.boundary_weight}, "
          f"warmup = {FLAGS.boundary_warmup} steps, "
          f"downscale = {FLAGS.boundary_downscale}, "
          f"every = {FLAGS.boundary_every}")

    if FLAGS.clean_model_dir and os.path.exists(FLAGS.model_dir):
        shutil.rmtree(FLAGS.model_dir)
    os.makedirs(FLAGS.model_dir, exist_ok=True)

    # --- Model + optimizer ---
    model     = FusionLaneModel(num_classes=FLAGS.num_classes).to(device)
    optimizer = optim.Adam(model.parameters(),
                           lr=FLAGS.initial_lr, weight_decay=FLAGS.weight_decay)

    global_step, start_epoch = load_checkpoint(model, optimizer, FLAGS.model_dir)

    class_weights  = torch.tensor(_CLASS_WEIGHTS)
    boundary_fn    = BoundaryLoss(
        num_classes=FLAGS.num_classes,
        ignore_label=FLAGS.ignore_label,
        class_weights=_CLASS_WEIGHTS,   # reuse same per-class weighting
        use_softmax=True,
    ).to(device)

    num_outer = FLAGS.train_epochs // FLAGS.epochs_per_eval

    for outer in range(start_epoch, num_outer):
        cp_x = randint(0, 78)
        cp_y = randint(0, 78)
        print(f"\n{'='*60}")
        print(f"Epoch {outer+1}/{num_outer}   cp=({cp_x},{cp_y})")
        print(f"{'='*60}")

        dataset = FusionLaneDataset(FLAGS.data_dir,
                                    is_training=True, cp_x=cp_x, cp_y=cp_y)
        loader  = DataLoader(dataset, batch_size=FLAGS.batch_size,
                             num_workers=0, drop_last=True)

        model.train()
        epoch_ce = epoch_bd = epoch_acc = epoch_iou = 0.0
        n_steps = n_boundary_steps = 0

        for img, reg, lbl in loader:
            lr = poly_lr(FLAGS.initial_lr, FLAGS.end_lr,
                         global_step, FLAGS.max_iter)
            for pg in optimizer.param_groups:
                pg['lr'] = lr

            img = img.to(device); reg = reg.to(device); lbl = lbl.to(device)

            optimizer.zero_grad()
            logits = model(img, reg, training=True)

            # Always: weighted CE.
            ce_loss = weighted_cross_entropy(
                logits, lbl,
                FLAGS.num_classes, FLAGS.ignore_label,
                class_weights, device,
            )

            # Sometimes: boundary loss. Controlled by --boundary_every.
            do_boundary = (global_step % FLAGS.boundary_every == 0)
            alpha = ramp_alpha(global_step, FLAGS.boundary_warmup,
                               FLAGS.boundary_weight)

            if do_boundary and alpha > 0:
                dist_maps = _distance_maps_for_batch(
                    lbl, FLAGS.num_classes, FLAGS.ignore_label,
                    FLAGS.boundary_downscale,
                )
                bd_loss = boundary_fn(logits, lbl, dist_maps=dist_maps)
                total   = ce_loss + alpha * bd_loss
                n_boundary_steps += 1
            else:
                bd_loss = torch.tensor(0.0, device=device)
                total   = ce_loss

            total.backward()
            optimizer.step()

            global_step += 1
            n_steps     += 1

            px_acc, mean_iou = compute_metrics(
                logits, lbl, FLAGS.num_classes, FLAGS.ignore_label)

            epoch_ce  += ce_loss.item()
            epoch_bd  += bd_loss.item()
            epoch_acc += px_acc
            epoch_iou += mean_iou

            if n_steps % 10 == 0:
                print(f"  step={global_step:7d}  "
                      f"ce={ce_loss.item():.4f}  "
                      f"bd={bd_loss.item():+.3f}  "
                      f"α={alpha:.3f}  "
                      f"px_acc={px_acc:.4f}  "
                      f"mIoU={mean_iou:.4f}  "
                      f"lr={lr:.2e}")

        n = max(n_steps, 1)
        nb = max(n_boundary_steps, 1)
        print(f"\nEpoch {outer+1} summary:  "
              f"avg_ce={epoch_ce/n:.4f}  "
              f"avg_bd={epoch_bd/nb:+.4f}  "
              f"avg_acc={epoch_acc/n:.4f}  "
              f"avg_mIoU={epoch_iou/n:.4f}")

        save_checkpoint(model, optimizer, global_step, outer + 1, FLAGS.model_dir)

    print("\nTraining complete.")


if __name__ == '__main__':
    main()
