"""
FusionLane — PyTorch inference script.

Reads test TFRecords, runs the model, and saves coloured segmentation images.

Usage:
    python infer_pt.py \
        --data_dir  ~/FusionLane/tfrecord/tfrecord \
        --model_dir ~/FusionLane/model_pt \
        --output_dir ~/FusionLane/output_images
"""

import argparse
import os
import numpy as np
import torch
from PIL import Image

from model_pt   import FusionLaneModel
from dataset_pt import FusionLaneDataset

# Colour map — matches preprocessing.py label_colours
LABEL_COLOURS = [
    (0,   0,   0),    # 0 = background
    (220, 0,   0),    # 1 = solid line
    (0,   220, 0),    # 2 = dash line
    (220, 220, 0),    # 3 = arrow
    (0,   0,   220),  # 4 = other line
    (220, 0,   220),  # 5 = stop line
    (220, 220, 220),  # 6 = back points
]
COLOUR_MAP = np.array(LABEL_COLOURS, dtype=np.uint8)  # [7, 3]

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir',   type=str, default='./tfrecord/')
parser.add_argument('--model_dir',  type=str, default='./model_pt/')
parser.add_argument('--output_dir', type=str, default='./output_images/')
parser.add_argument('--num_classes',type=int, default=7)
parser.add_argument('--split',      type=str, default='test',
                    choices=['test', 'train'],
                    help='Which split to run inference on.')


def decode_labels(pred):
    """pred: [H, W] int64 → RGB image [H, W, 3] uint8"""
    pred_clipped = np.clip(pred, 0, len(LABEL_COLOURS) - 1)
    return COLOUR_MAP[pred_clipped]


def load_model(model_dir, num_classes, device):
    latest = os.path.join(model_dir, 'latest.pt')
    if not os.path.exists(latest):
        # Try to find any checkpoint
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

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    model = load_model(FLAGS.model_dir, FLAGS.num_classes, device)

    is_training = (FLAGS.split == 'train')
    dataset = FusionLaneDataset(FLAGS.data_dir,
                                is_training=is_training,
                                cp_x=34, cp_y=34)   # centre crop for inference

    print(f"\nRunning inference on '{FLAGS.split}' split...")
    print(f"Output → {FLAGS.output_dir}\n")

    # We need batches of 4 (ConvLSTM groups 4 samples as a sequence).
    # Accumulate 4 samples then run forward.
    batch_imgs   = []
    batch_regs   = []
    batch_idxs   = []
    sample_idx   = 0

    def run_batch(imgs, regs, idxs):
        with torch.no_grad():
            img_t = torch.stack(imgs).to(device)   # [4, 3, H, W]
            reg_t = torch.stack(regs).to(device)   # [4, 1, H, W]
            logits = model(img_t, reg_t, training=False)  # [4, C, H, W]
            preds  = logits.argmax(dim=1).cpu().numpy()   # [4, H, W]

        for pred, idx in zip(preds, idxs):
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

    # Handle leftover samples (pad to 4 with last sample)
    if batch_imgs:
        while len(batch_imgs) < 4:
            batch_imgs.append(batch_imgs[-1])
            batch_regs.append(batch_regs[-1])
            batch_idxs.append(None)   # don't save padded ones
        with torch.no_grad():
            img_t = torch.stack(batch_imgs).to(device)
            reg_t = torch.stack(batch_regs).to(device)
            logits = model(img_t, reg_t, training=False)
            preds  = logits.argmax(dim=1).cpu().numpy()
        for pred, idx in zip(preds, batch_idxs):
            if idx is None:
                continue
            rgb = decode_labels(pred)
            out_path = os.path.join(FLAGS.output_dir, f'pred_{idx:04d}.png')
            Image.fromarray(rgb).save(out_path)
            print(f"  saved {out_path}")

    print(f"\nDone — {sample_idx} images saved to {FLAGS.output_dir}")


if __name__ == '__main__':
    main()
