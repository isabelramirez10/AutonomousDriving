"""
FusionLane — boosted inference script.

Drop-in replacement for infer_pt.py with two extra knobs:
  --logit_bias     : add per-class bias to logits before argmax
  --dilate_kernel  : morphologically dilate marker classes in the output
  --dilate_classes : which class IDs to dilate (default: 3=arrow, 5=stop_line)

Usage:
    python infer_pt_boosted.py \
        --data_dir    ~/FusionLane/tfrecord/tfrecord \
        --model_dir   ~/FusionLane/model_pt_boundary \
        --output_dir  ~/FusionLane/output_boosted \
        --logit_bias  default \
        --dilate_kernel 5 \
        --dilate_classes 3 5
"""

import argparse
import os
import numpy as np
import torch
from PIL import Image

from model_pt    import FusionLaneModel
from dataset_pt  import FusionLaneDataset
from marker_boost import (
    apply_logit_bias,
    boost_labels,
    decode_labels,
    parse_logit_bias,
    MARKER_CLASSES,
)

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir',        type=str,   default='./tfrecord/')
parser.add_argument('--model_dir',       type=str,   default='./model_pt/')
parser.add_argument('--output_dir',      type=str,   default='./output_boosted/')
parser.add_argument('--num_classes',     type=int,   default=7)
parser.add_argument('--split',           type=str,   default='test',
                    choices=['test', 'train'])
parser.add_argument('--logit_bias',      type=str,   default='default',
                    help="'none' | 'default' | JSON dict e.g. '{\"3\":1.5,\"5\":1.5}'")
parser.add_argument('--dilate_kernel',   type=int,   default=5,
                    help='Dilation kernel size (odd int; 0 or 1 = off)')
parser.add_argument('--dilate_classes',  type=int,   nargs='+',
                    default=list(MARKER_CLASSES),
                    help='Class IDs to dilate (default: 3 5)')


def load_model(model_dir, num_classes, device):
    latest = os.path.join(model_dir, 'latest.pt')
    if not os.path.exists(latest):
        pts = sorted([f for f in os.listdir(model_dir) if f.endswith('.pt')])
        if not pts:
            raise FileNotFoundError(f"No checkpoint found in {model_dir}")
        latest = os.path.join(model_dir, pts[-1])
    print(f"Loading checkpoint: {latest}")
    ckpt  = torch.load(latest, map_location='cpu')
    model = FusionLaneModel(num_classes=num_classes)
    model.load_state_dict(ckpt['model_state'])
    model.to(device)
    model.eval()
    print(f"  step={ckpt.get('global_step', '?')}  epoch={ckpt.get('epoch', '?')}")
    return model


def main():
    FLAGS = parser.parse_args()
    os.makedirs(FLAGS.output_dir, exist_ok=True)

    bias          = parse_logit_bias(FLAGS.logit_bias)
    dilate_kernel = FLAGS.dilate_kernel
    dilate_classes = tuple(FLAGS.dilate_classes)

    print(f"Logit bias    : {bias}")
    print(f"Dilate kernel : {dilate_kernel}")
    print(f"Dilate classes: {dilate_classes}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    model = load_model(FLAGS.model_dir, FLAGS.num_classes, device)

    dataset = FusionLaneDataset(FLAGS.data_dir,
                                is_training=(FLAGS.split == 'train'),
                                cp_x=34, cp_y=34)

    print(f"\nRunning boosted inference on '{FLAGS.split}' split...")
    print(f"Output → {FLAGS.output_dir}\n")

    batch_imgs = []
    batch_regs = []
    batch_idxs = []
    sample_idx = 0

    def run_batch(imgs, regs, idxs):
        with torch.no_grad():
            img_t  = torch.stack(imgs).to(device)
            reg_t  = torch.stack(regs).to(device)
            logits = model(img_t, reg_t, training=False)

            # Knob 1 — logit bias before argmax
            logits = apply_logit_bias(logits, bias)

            preds = logits.argmax(dim=1).cpu().numpy()  # [4, H, W]

        for pred, idx in zip(preds, idxs):
            if idx is None:
                continue
            # Knob 2 — dilation on label map
            if dilate_kernel > 1:
                pred = boost_labels(pred,
                                    marker_classes=dilate_classes,
                                    dilate_kernel=dilate_kernel)
            rgb = decode_labels(pred)
            out_path = os.path.join(FLAGS.output_dir, f'pred_{idx:04d}.png')
            Image.fromarray(rgb).save(out_path)
            print(f"  saved {out_path}")

    for img, reg, lbl in dataset:
        batch_imgs.append(img)
        batch_regs.append(reg)
        batch_idxs.append(sample_idx)
        sample_idx += 1

        if len(batch_imgs) == 4:
            run_batch(batch_imgs, batch_regs, batch_idxs)
            batch_imgs, batch_regs, batch_idxs = [], [], []

    # Pad leftover samples
    if batch_imgs:
        while len(batch_imgs) < 4:
            batch_imgs.append(batch_imgs[-1])
            batch_regs.append(batch_regs[-1])
            batch_idxs.append(None)
        run_batch(batch_imgs, batch_regs, batch_idxs)

    print(f"\nDone — {sample_idx} images saved to {FLAGS.output_dir}")


if __name__ == '__main__':
    main()
