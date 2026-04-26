"""
FusionLane — data preparation script.

Converts annotations from TuSimple, CULane, and/or CVAT into the unified
paired format that FusionLaneDataset expects for supervised training:

  output_dir/
    train/
      images/   <source>_<index>.jpg
      masks/    <source>_<index>.png   (white=lane, black=background)
    val/
      images/
      masks/

Usage
-----
python prepare_data.py \\
    [--tusimple_dir  path/to/TuSimple] \\
    [--culane_dir    path/to/CULane  ] \\
    [--cvat_dir      path/to/export  ] \\
    [--output_dir    ./data          ] \\
    [--val_split     0.1             ] \\
    [--lane_thickness 12             ] \\
    [--seed          42              ]

At least one source must be supplied.

Dataset download links
----------------------
TuSimple : https://github.com/TuSimple/tusimple-benchmark
CULane   : https://xingangpan.github.io/projects/CULane.html
CVAT     : https://app.cvat.ai  (annotate your own dashcam frames)

CVAT export format supported
-----------------------------
A) Segmentation Masks 1.1 export:  SegmentationClass/ + images/ + labelmap.txt
B) Simple paired folders:          images/  +  masks/
"""

import argparse
import json
import os
import random

import cv2
import numpy as np
from tqdm import tqdm


# ─────────────────────────────────────────────────────────────────────────────
# Descriptor types
#
# A "descriptor" is a lightweight dict produced by the loaders.
# It does NOT contain image data — materialise() loads/draws on demand,
# so the entire dataset never lives in RAM at once.
# ─────────────────────────────────────────────────────────────────────────────

def _d_path(img_path, mask_path, mask_fmt='binary'):
    """Pre-existing image + mask files on disk."""
    return {'kind': 'path',
            'img_path': img_path, 'mask_path': mask_path, 'mask_fmt': mask_fmt}


def _d_tusimple(img_path, lanes, h_samples):
    """TuSimple annotation: draw polylines to generate mask."""
    return {'kind': 'tusimple',
            'img_path': img_path, 'lanes': lanes, 'h_samples': h_samples}


def _d_culane_lines(img_path, lines_path):
    """CULane .lines.txt annotation: draw polylines to generate mask."""
    return {'kind': 'culane_lines', 'img_path': img_path, 'lines_path': lines_path}


def _d_array(img_bgr, mask_gray):
    """In-memory arrays (used for small CVAT Mode-A sets)."""
    return {'kind': 'array', 'img': img_bgr, 'mask': mask_gray}


def materialise(desc, lane_thickness):
    """
    Convert any descriptor → (img_bgr uint8, mask_gray uint8) or None on error.
    mask_gray values: 255 = lane, 0 = background.
    """
    kind = desc['kind']

    if kind == 'path':
        img = cv2.imread(desc['img_path'])
        if img is None:
            return None
        raw = cv2.imread(desc['mask_path'], cv2.IMREAD_GRAYSCALE)
        if raw is None:
            return None
        if desc['mask_fmt'] == 'seglabel':        # CULane: 0=bg, 1-4=lanes, 255=ignore
            mask = np.zeros_like(raw)
            mask[(raw >= 1) & (raw <= 4)] = 255
        else:                                      # already binary
            mask = (raw > 127).astype(np.uint8) * 255
        return img, mask

    if kind == 'tusimple':
        img = cv2.imread(desc['img_path'])
        if img is None:
            return None
        h, w = img.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        for lane in desc['lanes']:
            pts = [
                (int(x), int(y))
                for x, y in zip(lane, desc['h_samples'])
                if x >= 0          # -2 marks missing point in TuSimple
            ]
            for j in range(1, len(pts)):
                cv2.line(mask, pts[j - 1], pts[j], 255, lane_thickness)
        return img, mask

    if kind == 'culane_lines':
        img = cv2.imread(desc['img_path'])
        if img is None:
            return None
        h, w = img.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        with open(desc['lines_path']) as f:
            for line in f:
                vals = list(map(float, line.strip().split()))
                if len(vals) < 4:
                    continue
                pts = [(int(vals[i]), int(vals[i + 1]))
                       for i in range(0, len(vals) - 1, 2)]
                for j in range(1, len(pts)):
                    cv2.line(mask, pts[j - 1], pts[j], 255, lane_thickness)
        return img, mask

    if kind == 'array':
        return desc['img'], desc['mask']

    return None


# ─────────────────────────────────────────────────────────────────────────────
# TuSimple
# ─────────────────────────────────────────────────────────────────────────────

_TUSIMPLE_TRAIN_JSON = [
    'label_data_0313.json',
    'label_data_0531.json',
    'label_data_0601.json',
]
_TUSIMPLE_VAL_JSON = ['test_label.json']


def load_tusimple(tusimple_dir):
    """
    Returns (train_descs, val_descs).
    Train: label_data_0313/0531/0601.json
    Val  : test_label.json  (labels included in the benchmark release)
    """
    def _parse(json_files, label):
        descs = []
        for jf in json_files:
            jpath = os.path.join(tusimple_dir, jf)
            if not os.path.exists(jpath):
                print(f'  [TuSimple] {jf} not found — skipping.')
                continue
            with open(jpath) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    rec      = json.loads(line)
                    img_path = os.path.join(tusimple_dir,
                                            rec['raw_file'].lstrip('/\\'))
                    if os.path.exists(img_path):
                        descs.append(_d_tusimple(img_path,
                                                  rec['lanes'],
                                                  rec['h_samples']))
        print(f'  [TuSimple] {label}: {len(descs)} samples')
        return descs

    print('[TuSimple] Scanning ...')
    return _parse(_TUSIMPLE_TRAIN_JSON, 'train'), _parse(_TUSIMPLE_VAL_JSON, 'val')


# ─────────────────────────────────────────────────────────────────────────────
# CULane
# ─────────────────────────────────────────────────────────────────────────────

def load_culane(culane_dir, val_split):
    """
    Returns (train_descs, val_descs).
    Reads list/train_gt.txt and list/val.txt (or list/test.txt).
    Falls back to a random val_split from train if no val list is found.

    Mask priority per sample:
      1. laneseg_label_w16 segmentation PNG (values 0-4, binarised)
      2. .lines.txt polyline annotation (drawn at materialise time)
    """
    list_dir = os.path.join(culane_dir, 'list')
    train_gt = os.path.join(list_dir, 'train_gt.txt')

    if not os.path.exists(train_gt):
        print(f'  [CULane] list/train_gt.txt not found at {culane_dir} — skipping.')
        return [], []

    def _parse_list(list_path):
        descs = []
        with open(list_path) as f:
            for raw_line in f:
                parts = raw_line.strip().split()
                if not parts:
                    continue
                img_path = os.path.join(culane_dir, parts[0].lstrip('/\\'))
                if not os.path.exists(img_path):
                    continue

                # Prefer seg label mask
                if len(parts) > 1:
                    seg_path = os.path.join(culane_dir, parts[1].lstrip('/\\'))
                    if os.path.exists(seg_path):
                        descs.append(_d_path(img_path, seg_path, 'seglabel'))
                        continue

                # Fall back to .lines.txt
                lines_path = os.path.splitext(img_path)[0] + '.lines.txt'
                if os.path.exists(lines_path):
                    descs.append(_d_culane_lines(img_path, lines_path))
        return descs

    print('[CULane] Scanning train_gt.txt ...')
    train_descs = _parse_list(train_gt)
    print(f'  {len(train_descs)} train samples')

    # Locate val list
    val_txt = os.path.join(list_dir, 'val.txt')
    if not os.path.exists(val_txt):
        val_txt = os.path.join(list_dir, 'test.txt')

    if os.path.exists(val_txt):
        print(f'[CULane] Scanning {os.path.basename(val_txt)} ...')
        val_descs = _parse_list(val_txt)
        print(f'  {len(val_descs)} val samples')
    else:
        n_val = max(1, int(len(train_descs) * val_split))
        random.shuffle(train_descs)
        val_descs   = train_descs[:n_val]
        train_descs = train_descs[n_val:]
        print(f'  No val/test list — split {n_val} random samples as val')

    return train_descs, val_descs


# ─────────────────────────────────────────────────────────────────────────────
# CVAT
# ─────────────────────────────────────────────────────────────────────────────

_IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}


def _find_img_dir(base):
    for candidate in [
        os.path.join(base, 'images', 'default'),
        os.path.join(base, 'images'),
        base,
    ]:
        if os.path.isdir(candidate) and any(
            os.path.splitext(f)[1].lower() in _IMG_EXTS
            for f in os.listdir(candidate)
        ):
            return candidate
    return None


def _parse_labelmap(lm_path):
    """CVAT labelmap.txt → {label_name: (R, G, B)}"""
    result = {}
    if not os.path.exists(lm_path):
        return result
    with open(lm_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split(':')
            if len(parts) < 2:
                continue
            name = parts[0].strip()
            try:
                r, g, b = map(int, parts[1].strip().split(','))
                result[name] = (r, g, b)
            except Exception:
                pass
    return result


def load_cvat(cvat_dir, val_split):
    """
    Supports two CVAT export layouts:
      Mode A — Segmentation Masks 1.1 export
               SegmentationClass/  images/  labelmap.txt
      Mode B — simple paired folders
               images/  masks/
    Returns (train_descs, val_descs).
    """
    seg_class_dir = os.path.join(cvat_dir, 'SegmentationClass')
    masks_dir     = os.path.join(cvat_dir, 'masks')
    descs         = []

    # ── Mode A ───────────────────────────────────────────────────────────────
    if os.path.isdir(seg_class_dir):
        labelmap   = _parse_labelmap(os.path.join(cvat_dir, 'labelmap.txt'))
        lane_color = None
        for name, color in labelmap.items():
            if 'lane' in name.lower():
                lane_color = color
                break
        if lane_color is None:
            # Use first non-background colour as fallback
            for name, color in labelmap.items():
                if 'background' not in name.lower() and color != (0, 0, 0):
                    lane_color = color
                    break
        print(f'  [CVAT] Mode A — lane colour from labelmap: {lane_color}')
        img_dir = _find_img_dir(cvat_dir)

        for fname in os.listdir(seg_class_dir):
            if os.path.splitext(fname)[1].lower() not in _IMG_EXTS:
                continue
            stem     = os.path.splitext(fname)[0]
            mask_bgr = cv2.imread(os.path.join(seg_class_dir, fname))
            if mask_bgr is None:
                continue

            if lane_color is not None:
                r, g, b = lane_color
                lane_px = (
                    (mask_bgr[:, :, 2] == r) &
                    (mask_bgr[:, :, 1] == g) &
                    (mask_bgr[:, :, 0] == b)
                )
            else:
                lane_px = mask_bgr.sum(axis=2) > 0       # any non-black = lane

            binary_mask = lane_px.astype(np.uint8) * 255

            img_bgr = None
            if img_dir:
                for ext in ('.jpg', '.jpeg', '.png', '.bmp'):
                    p = os.path.join(img_dir, stem + ext)
                    if os.path.exists(p):
                        img_bgr = cv2.imread(p)
                        break
            if img_bgr is None:
                continue
            descs.append(_d_array(img_bgr, binary_mask))

        print(f'  [CVAT] Mode A — {len(descs)} annotated frames found')

    # ── Mode B ───────────────────────────────────────────────────────────────
    elif os.path.isdir(masks_dir):
        img_dir = _find_img_dir(cvat_dir)
        if not img_dir:
            print(f'  [CVAT] Mode B — no images/ folder found in {cvat_dir}')
            return [], []

        for fname in sorted(os.listdir(img_dir)):
            if os.path.splitext(fname)[1].lower() not in _IMG_EXTS:
                continue
            stem = os.path.splitext(fname)[0]
            mask_path = None
            for ext in ('.png', '.jpg', '.jpeg', '.bmp'):
                p = os.path.join(masks_dir, stem + ext)
                if os.path.exists(p):
                    mask_path = p
                    break
            if mask_path is None:
                continue
            descs.append(_d_path(os.path.join(img_dir, fname),
                                  mask_path, 'binary'))

        print(f'  [CVAT] Mode B — {len(descs)} pairs found')

    else:
        print(
            f'  [CVAT] Could not detect layout in {cvat_dir}\n'
            '  Expected: SegmentationClass/ + images/ + labelmap.txt  (Mode A)\n'
            '         or images/ + masks/                             (Mode B)'
        )
        return [], []

    if not descs:
        return [], []

    random.shuffle(descs)
    n_val       = max(1, int(len(descs) * val_split))
    val_descs   = descs[:n_val]
    train_descs = descs[n_val:]
    print(f'  [CVAT] {len(train_descs)} train / {len(val_descs)} val')
    return train_descs, val_descs


# ─────────────────────────────────────────────────────────────────────────────
# Writer
# ─────────────────────────────────────────────────────────────────────────────

def write_split(descs, split_name, out_dir, prefix, lane_thickness):
    """Materialise each descriptor and write to split_name/images + masks."""
    if not descs:
        return 0
    img_dir  = os.path.join(out_dir, split_name, 'images')
    mask_dir = os.path.join(out_dir, split_name, 'masks')
    os.makedirs(img_dir,  exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)

    written = 0
    for i, desc in enumerate(tqdm(descs, desc=f'  {split_name}/{prefix}')):
        result = materialise(desc, lane_thickness)
        if result is None:
            continue
        img_bgr, mask_gray = result
        stem = f'{prefix}_{i:06d}'
        cv2.imwrite(os.path.join(img_dir,  stem + '.jpg'), img_bgr,
                    [cv2.IMWRITE_JPEG_QUALITY, 95])
        cv2.imwrite(os.path.join(mask_dir, stem + '.png'), mask_gray)
        written += 1
    return written


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description='Prepare FusionLane training data from TuSimple, CULane, and/or CVAT'
    )
    p.add_argument('--tusimple_dir',   type=str, default=None,
                   help='Root of the TuSimple dataset (contains label_data_*.json)')
    p.add_argument('--culane_dir',     type=str, default=None,
                   help='Root of the CULane dataset (contains list/train_gt.txt)')
    p.add_argument('--cvat_dir',       type=str, default=None,
                   help='CVAT export folder (Mode A: SegmentationClass/, Mode B: masks/)')
    p.add_argument('--output_dir',     type=str, default='./data',
                   help='Where to write train/ and val/ (default: ./data)')
    p.add_argument('--val_split',      type=float, default=0.1,
                   help='Fraction of CVAT / CULane-no-vallist data to use as val (default 0.1)')
    p.add_argument('--lane_thickness', type=int, default=12,
                   help='Line thickness in pixels for TuSimple/CULane polyline drawing (default 12)')
    p.add_argument('--seed',           type=int, default=42,
                   help='Random seed for reproducible val splits (default 42)')
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    random.seed(args.seed)

    if not any([args.tusimple_dir, args.culane_dir, args.cvat_dir]):
        print('ERROR: supply at least one of --tusimple_dir, --culane_dir, --cvat_dir')
        return

    os.makedirs(args.output_dir, exist_ok=True)

    counts = {}   # {source: {'train': n, 'val': n}}

    # ── TuSimple ─────────────────────────────────────────────────────────────
    if args.tusimple_dir:
        tr, va = load_tusimple(args.tusimple_dir)
        n_tr = write_split(tr, 'train', args.output_dir, 'tusimple', args.lane_thickness)
        n_va = write_split(va, 'val',   args.output_dir, 'tusimple', args.lane_thickness)
        counts['TuSimple'] = {'train': n_tr, 'val': n_va}

    # ── CULane ───────────────────────────────────────────────────────────────
    if args.culane_dir:
        tr, va = load_culane(args.culane_dir, args.val_split)
        n_tr = write_split(tr, 'train', args.output_dir, 'culane', args.lane_thickness)
        n_va = write_split(va, 'val',   args.output_dir, 'culane', args.lane_thickness)
        counts['CULane'] = {'train': n_tr, 'val': n_va}

    # ── CVAT ─────────────────────────────────────────────────────────────────
    if args.cvat_dir:
        tr, va = load_cvat(args.cvat_dir, args.val_split)
        n_tr = write_split(tr, 'train', args.output_dir, 'cvat', args.lane_thickness)
        n_va = write_split(va, 'val',   args.output_dir, 'cvat', args.lane_thickness)
        counts['CVAT'] = {'train': n_tr, 'val': n_va}

    # ── Summary ──────────────────────────────────────────────────────────────
    total_tr = sum(v['train'] for v in counts.values())
    total_va = sum(v['val']   for v in counts.values())

    print('\n── Data preparation complete ────────────────────────────────────')
    print(f'Output : {os.path.abspath(args.output_dir)}')
    print(f'{"Source":<12}  {"Train":>8}  {"Val":>8}')
    print('─' * 32)
    for src, c in counts.items():
        print(f'{src:<12}  {c["train"]:>8}  {c["val"]:>8}')
    print('─' * 32)
    print(f'{"TOTAL":<12}  {total_tr:>8}  {total_va:>8}')

    if total_tr == 0:
        print('\nWARNING: No training samples were written.')
        print('Check that the dataset paths are correct and contain expected files.')
        return

    print(f"""
Next steps
──────────
1. Train:
   python train_pt.py --data_dir {args.output_dir} --epochs 50 --patience 10

2. Inference (video):
   python infer_media.py --input path/to/video.mp4 --output_dir ./outputs/run1

3. If mIoU plateaus, re-annotate or add more data, then re-run prepare_data.py.
""")


if __name__ == '__main__':
    main()
