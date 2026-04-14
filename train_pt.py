"""
FusionLane — PyTorch training script.

Designed for RTX 5080 (Blackwell / CUDA 12.8+).
Data loading uses TensorFlow CPU-only to read the existing TFRecords.

Usage:
    python train_pt.py \
        --data_dir  ~/FusionLane/tfrecord/tfrecord \
        --model_dir ~/FusionLane/model_pt \
        --train_epochs 3
"""

from __future__ import print_function

import argparse
import os
import math
import shutil
from random import randint

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from model_pt   import FusionLaneModel
from dataset_pt import FusionLaneDataset


# ---------------------------------------------------------------------------
# Hyper-parameters (matching the original TF training script)
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir',       type=str,   default='./tfrecord/')
parser.add_argument('--model_dir',      type=str,   default='./model_pt/')
parser.add_argument('--train_epochs',   type=int,   default=120)
parser.add_argument('--epochs_per_eval',type=int,   default=1)
parser.add_argument('--batch_size',     type=int,   default=20,
                    help='Must be divisible by 4 (ConvLSTM temporal grouping).')
parser.add_argument('--initial_lr',     type=float, default=1e-5)
parser.add_argument('--end_lr',         type=float, default=5e-7)
parser.add_argument('--max_iter',       type=int,   default=100000)
parser.add_argument('--weight_decay',   type=float, default=2e-4)
parser.add_argument('--num_classes',    type=int,   default=7)
parser.add_argument('--ignore_label',   type=int,   default=20)
parser.add_argument('--clean_model_dir',action='store_true')

_NUM_TRAIN = 11440

# Per-class weights (matching original)
#   0=background, 1=solidline, 2=dashline, 3=arrow,
#   4=otherline, 5=stopline, 6=backpoints
_CLASS_WEIGHTS = [1.0, 8.0, 8.0, 8.0, 8.0, 10.0, 2.0]


# ---------------------------------------------------------------------------
# Learning-rate schedule
# ---------------------------------------------------------------------------

def poly_lr(initial_lr, end_lr, step, max_iter, power=0.9):
    """Polynomial decay — identical to tf.train.polynomial_decay."""
    step   = min(step, max_iter)
    factor = (1.0 - step / max_iter) ** power
    return end_lr + (initial_lr - end_lr) * factor


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def weighted_cross_entropy(logits, labels, num_classes, ignore_label, weights, device):
    """
    Sparse softmax cross-entropy with per-class weights, ignoring void pixels.

    logits : [B, C, H, W]
    labels : [B, H, W]   int64, void pixels have value `ignore_label`
    """
    B, C, H, W = logits.shape

    logits_flat = logits.permute(0, 2, 3, 1).reshape(-1, C)   # [N, C]
    labels_flat = labels.reshape(-1)                            # [N]

    valid       = labels_flat < num_classes
    v_logits    = logits_flat[valid]
    v_labels    = labels_flat[valid]

    if v_labels.numel() == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)

    w       = weights.to(device)[v_labels]
    ce      = nn.functional.cross_entropy(v_logits, v_labels, reduction='none')
    return (ce * w).mean()


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(logits, labels, num_classes, ignore_label):
    """Returns (pixel_accuracy, mean_iou) for a batch."""
    with torch.no_grad():
        pred  = logits.argmax(dim=1)                 # [B, H, W]
        valid = labels < num_classes
        pv    = pred[valid]
        lv    = labels[valid]

        px_acc = (pv == lv).float().mean().item() if lv.numel() > 0 else 0.0

        ious = []
        for cls in range(num_classes):
            tp = ((pv == cls) & (lv == cls)).sum().item()
            fp = ((pv == cls) & (lv != cls)).sum().item()
            fn = ((pv != cls) & (lv == cls)).sum().item()
            denom = tp + fp + fn
            if denom > 0:
                ious.append(tp / denom)
        mean_iou = sum(ious) / len(ious) if ious else 0.0

    return px_acc, mean_iou


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def _ckpt_path(model_dir, step):
    return os.path.join(model_dir, f'ckpt_step{step:07d}.pt')


def save_checkpoint(model, optimizer, global_step, epoch, model_dir):
    os.makedirs(model_dir, exist_ok=True)
    state = {
        'global_step':      global_step,
        'epoch':            epoch,
        'model_state':      model.state_dict(),
        'optimizer_state':  optimizer.state_dict(),
    }
    path   = _ckpt_path(model_dir, global_step)
    latest = os.path.join(model_dir, 'latest.pt')
    torch.save(state, path)
    torch.save(state, latest)
    print(f"  ✓ checkpoint saved → {path}")
    return path


def load_checkpoint(model, optimizer, model_dir):
    latest = os.path.join(model_dir, 'latest.pt')
    if not os.path.exists(latest):
        return 0, 0
    ckpt = torch.load(latest, map_location='cpu')
    model.load_state_dict(ckpt['model_state'])
    if ckpt["optimizer_state"]: optimizer.load_state_dict(ckpt["optimizer_state"])
    gs  = ckpt['global_step']
    ep  = ckpt['epoch']
    print(f"Loaded checkpoint: step={gs}, epoch={ep}")
    return gs, ep


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    FLAGS = parser.parse_args()

    assert FLAGS.batch_size % 4 == 0, \
        f"batch_size must be divisible by 4, got {FLAGS.batch_size}"

    # ── Device setup ───────────────────────────────────────────────────────
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device : {device}")
    if device.type == 'cuda':
        print(f"GPU    : {torch.cuda.get_device_name(0)}")
        print(f"CUDA   : {torch.version.cuda}")

    # ── Directories ────────────────────────────────────────────────────────
    if FLAGS.clean_model_dir and os.path.exists(FLAGS.model_dir):
        shutil.rmtree(FLAGS.model_dir)
    os.makedirs(FLAGS.model_dir, exist_ok=True)

    # ── Model + optimizer ──────────────────────────────────────────────────
    model     = FusionLaneModel(num_classes=FLAGS.num_classes).to(device)
    optimizer = optim.Adam(model.parameters(),
                           lr=FLAGS.initial_lr,
                           weight_decay=FLAGS.weight_decay)

    global_step, start_epoch = load_checkpoint(model, optimizer, FLAGS.model_dir)

    class_weights = torch.tensor(_CLASS_WEIGHTS)

    num_outer = FLAGS.train_epochs // FLAGS.epochs_per_eval

    # ── Training loop ──────────────────────────────────────────────────────
    for outer in range(start_epoch, num_outer):

        # Randomise crop offsets each outer epoch (matching original)
        cp_x = randint(0, 78)
        cp_y = randint(0, 78)

        print(f"\n{'='*60}")
        print(f"Epoch {outer+1}/{num_outer}   cp=({cp_x},{cp_y})")
        print(f"{'='*60}")

        # Build a fresh dataset for this epoch's crop parameters
        dataset = FusionLaneDataset(FLAGS.data_dir,
                                    is_training=True,
                                    cp_x=cp_x, cp_y=cp_y)
        loader  = DataLoader(dataset,
                             batch_size=FLAGS.batch_size,
                             num_workers=0,   # required: TF session not fork-safe
                             drop_last=True)  # keep batch size fixed for ConvLSTM

        model.train()

        epoch_loss = epoch_acc = epoch_iou = 0.0
        n_steps = 0

        for img, reg, lbl in loader:
            # ── Update learning rate ──────────────────────────────────────
            lr = poly_lr(FLAGS.initial_lr, FLAGS.end_lr,
                         global_step, FLAGS.max_iter)
            for pg in optimizer.param_groups:
                pg['lr'] = lr

            img = img.to(device)
            reg = reg.to(device)
            lbl = lbl.to(device)

            # ── Forward + loss ────────────────────────────────────────────
            optimizer.zero_grad()
            logits = model(img, reg, training=True)

            loss = weighted_cross_entropy(
                logits, lbl,
                FLAGS.num_classes, FLAGS.ignore_label,
                class_weights, device,
            )
            loss.backward()
            optimizer.step()

            global_step += 1
            n_steps     += 1

            # ── Metrics ───────────────────────────────────────────────────
            px_acc, mean_iou = compute_metrics(
                logits, lbl, FLAGS.num_classes, FLAGS.ignore_label)

            epoch_loss += loss.item()
            epoch_acc  += px_acc
            epoch_iou  += mean_iou

            if n_steps % 10 == 0:
                print(f"  step={global_step:7d}  "
                      f"loss={loss.item():.5f}  "
                      f"px_acc={px_acc:.4f}  "
                      f"mIoU={mean_iou:.4f}  "
                      f"lr={lr:.2e}")

        # ── End-of-epoch summary + checkpoint ────────────────────────────
        n = max(n_steps, 1)
        print(f"\nEpoch {outer+1} summary:  "
              f"avg_loss={epoch_loss/n:.4f}  "
              f"avg_acc={epoch_acc/n:.4f}  "
              f"avg_mIoU={epoch_iou/n:.4f}")

        save_checkpoint(model, optimizer, global_step, outer + 1, FLAGS.model_dir)

    print("\nTraining complete.")


if __name__ == '__main__':
    main()
