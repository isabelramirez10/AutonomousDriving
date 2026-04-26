"""
FusionLane — KITTI Road dataset preparation  (camera + LiDAR fusion).

Downloads (free, no registration)
----------------------------------
http://www.cvlibs.net/datasets/kitti/eval_road.php

Required files
--------------
  data_road.zip           Left color images + ground-truth lane/road masks
  data_road_velodyne.zip  Velodyne LiDAR point clouds  [optional but recommended]

Both are ~350 MB each.

Expected directory structure after extraction
---------------------------------------------
kitti_root/
  data_road/
    training/
      image_2/          um_000000.png  umm_000000.png  uu_000000.png ...
      gt_image_2/       um_lane_000000.png  um_road_000000.png ...
      velodyne/         um_000000.bin  ...         [from data_road_velodyne.zip]
      calib/            um_000000.txt  ...
    testing/
      image_2/
      velodyne/
      calib/

Output
------
data/
  train/
    images/    kitti_000000.jpg ...  (RGB camera images)
    masks/     kitti_000000.png ...  (binary lane masks: white=lane, black=bg)
    depths/    kitti_000000.png ...  (normalised LiDAR depth map, 0-255)
  val/
    images/  masks/  depths/

If Velodyne data is absent, depths/ is omitted and channel-4 defaults to
all-ones during training (camera-only mode, still works).

Label classes used
------------------
  um_lane_*.png   Urban Marked  —  lane markings       ← primary
  umm_lane_*.png  Urban Multi-Marked — multiple lanes  ← primary
  uu_road_*.png   Urban Unmarked — road area (no lane) ← road area fallback

Usage
-----
  python prepare_kitti.py --kitti_root "C:/path/to/kitti_root" --output_dir ./data
"""

import argparse
import os
import random

import cv2
import numpy as np
from tqdm import tqdm

# ─────────────────────────────────────────────────────────────────────────────
# Calibration
# ─────────────────────────────────────────────────────────────────────────────

def load_calib(calib_path):
    """
    Parse a KITTI road calibration .txt file.
    Returns dict with P2 (3x4), R_rect (3x3), Tr_velo_to_cam (3x4).
    """
    data = {}
    with open(calib_path) as f:
        for line in f:
            line = line.strip()
            if not line or ':' not in line:
                continue
            key, vals = line.split(':', 1)
            data[key.strip()] = np.array(vals.strip().split(), dtype=np.float64)

    P2 = data.get('P2', np.array([1,0,0,0, 0,1,0,0, 0,0,1,0], dtype=np.float64))
    P2 = P2.reshape(3, 4)

    if 'R_rect' in data:
        R_rect = data['R_rect'].reshape(3, 3)
    elif 'R0_rect' in data:
        R_rect = data['R0_rect'].reshape(3, 3)
    else:
        R_rect = np.eye(3)

    if 'Tr_velo_to_cam' in data:
        Tr = data['Tr_velo_to_cam'].reshape(3, 4)
    elif 'Tr_velo_cam' in data:
        Tr = data['Tr_velo_cam'].reshape(3, 4)
    else:
        Tr = np.hstack([np.eye(3), np.zeros((3, 1))])

    return {'P2': P2, 'R_rect': R_rect, 'Tr_velo_to_cam': Tr}


# ─────────────────────────────────────────────────────────────────────────────
# LiDAR → depth map
# ─────────────────────────────────────────────────────────────────────────────

def project_lidar_to_depth(velo_path, calib, img_h, img_w, max_depth=80.0):
    """
    Project Velodyne point cloud onto the camera-2 image plane.
    Returns float32 [H, W] depth map in [0, 1].
    0.0 = no LiDAR return / beyond max_depth.
    """
    points = np.fromfile(velo_path, dtype=np.float32).reshape(-1, 4)

    # Keep only forward-facing points
    points = points[points[:, 0] > 0]
    if len(points) == 0:
        return np.zeros((img_h, img_w), dtype=np.float32)

    # Build full 3×4 projection matrix: P2 @ R0_4x4 @ Tr_4x4
    R0_4 = np.eye(4, dtype=np.float64)
    R0_4[:3, :3] = calib['R_rect']

    Tr_4 = np.eye(4, dtype=np.float64)
    Tr_4[:3, :] = calib['Tr_velo_to_cam']

    proj = calib['P2'] @ R0_4 @ Tr_4    # (3, 4)

    # Homogeneous coordinates
    pts_h = np.hstack([points[:, :3], np.ones((len(points), 1))])   # (N, 4)
    pts_c = (proj @ pts_h.T).T                                        # (N, 3)

    depth = pts_c[:, 2]
    eps   = 1e-6
    u = pts_c[:, 0] / (depth + eps)
    v = pts_c[:, 1] / (depth + eps)

    # Filter valid projections
    valid = (
        (depth > 0.1) & (depth < max_depth) &
        (u >= 0) & (u < img_w) &
        (v >= 0) & (v < img_h)
    )
    ui = u[valid].astype(np.int32)
    vi = v[valid].astype(np.int32)
    di = depth[valid]

    depth_map = np.zeros((img_h, img_w), dtype=np.float32)
    depth_map[vi, ui] = di / max_depth    # normalise to [0, 1]
    return depth_map


# ─────────────────────────────────────────────────────────────────────────────
# Sample discovery
# ─────────────────────────────────────────────────────────────────────────────

_LANE_PREFIXES = ('um_', 'umm_')     # has explicit lane-marking GTs
_ROAD_PREFIXES = ('uu_',)            # road-only GT (use road mask as lane proxy)


def discover_samples(training_dir):
    """
    Walk training/image_2 and match image → GT mask → calibration → velodyne.
    Returns list of dicts.
    """
    img_dir   = os.path.join(training_dir, 'image_2')
    gt_dir    = os.path.join(training_dir, 'gt_image_2')
    velo_dir  = os.path.join(training_dir, 'velodyne')
    calib_dir = os.path.join(training_dir, 'calib')

    if not os.path.isdir(img_dir):
        raise FileNotFoundError(f"image_2/ not found under {training_dir}")

    samples = []
    for fname in sorted(os.listdir(img_dir)):
        if not fname.endswith('.png') and not fname.endswith('.jpg'):
            continue
        stem = os.path.splitext(fname)[0]                   # e.g. um_000000
        img_path = os.path.join(img_dir, fname)

        # KITTI GT naming: um_000000.png → um_lane_000000.png / um_road_000000.png
        # Split stem into prefix (e.g. 'um') and number (e.g. '000000')
        parts  = stem.rsplit('_', 1)          # ['um', '000000']
        prefix = parts[0] if len(parts) == 2 else stem
        number = parts[1] if len(parts) == 2 else ''

        mask_path = None
        for gt_fname in (
            f"{prefix}_lane_{number}.png",    # um_lane_000000.png  (primary)
            f"{prefix}_road_{number}.png",    # um_road_000000.png  (fallback)
            f"{stem}_lane.png",               # legacy fallback
            f"{stem}_road.png",
        ):
            p = os.path.join(gt_dir, gt_fname)
            if os.path.exists(p):
                mask_path = p
                break
        if mask_path is None:
            continue    # no GT available → skip

        # Calibration
        calib_path = os.path.join(calib_dir, f"{stem}.txt")
        if not os.path.exists(calib_path):
            calib_path = None

        # LiDAR
        velo_path = os.path.join(velo_dir, f"{stem}.bin")
        if not os.path.exists(velo_path):
            velo_path = None

        samples.append({
            'stem':       stem,
            'img_path':   img_path,
            'mask_path':  mask_path,
            'calib_path': calib_path,
            'velo_path':  velo_path,
        })

    return samples


# ─────────────────────────────────────────────────────────────────────────────
# GT mask conversion
# ─────────────────────────────────────────────────────────────────────────────

def load_gt_mask(mask_path):
    """
    KITTI road GT colour encoding (BGR):
      [0,   0, 255] = background road (visible but not ego-lane)   ~92%
      [255, 0, 255] = ego-lane / lane marking (magenta)            ~6-7%
      [0,   0,   0] = outside camera field of view                  ~1%
    Lane pixels are those where the BLUE channel == 255 (magenta).
    Returns uint8 binary mask [H, W]: 255 = lane, 0 = background.
    """
    img = cv2.imread(mask_path)
    if img is None:
        raise IOError(f"Cannot read GT mask: {mask_path}")
    # Blue channel = 255 selects only the magenta (lane) pixels
    binary = (img[:, :, 0] > 127).astype(np.uint8) * 255
    return binary


# ─────────────────────────────────────────────────────────────────────────────
# Writer
# ─────────────────────────────────────────────────────────────────────────────

def write_sample(sample, split_dir, idx, has_lidar):
    img_dir   = os.path.join(split_dir, 'images')
    mask_dir  = os.path.join(split_dir, 'masks')
    depth_dir = os.path.join(split_dir, 'depths') if has_lidar else None
    os.makedirs(img_dir,  exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)
    if depth_dir:
        os.makedirs(depth_dir, exist_ok=True)

    img_bgr = cv2.imread(sample['img_path'])
    if img_bgr is None:
        return False

    mask = load_gt_mask(sample['mask_path'])
    stem = f"kitti_{idx:06d}"

    cv2.imwrite(os.path.join(img_dir,  stem + '.jpg'), img_bgr,
                [cv2.IMWRITE_JPEG_QUALITY, 95])
    cv2.imwrite(os.path.join(mask_dir, stem + '.png'), mask)

    if has_lidar and sample['velo_path'] and sample['calib_path']:
        h, w = img_bgr.shape[:2]
        calib     = load_calib(sample['calib_path'])
        depth_map = project_lidar_to_depth(sample['velo_path'], calib, h, w)
        depth_u8  = (depth_map * 255).clip(0, 255).astype(np.uint8)
        cv2.imwrite(os.path.join(depth_dir, stem + '.png'), depth_u8)

    return True


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description='Prepare KITTI Road dataset for FusionLane camera+LiDAR training'
    )
    p.add_argument('--kitti_root', type=str, required=True,
                   help='Root directory containing data_road/ (extracted from zip)')
    p.add_argument('--output_dir', type=str, default='./data',
                   help='Output directory for train/ and val/ folders (default: ./data)')
    p.add_argument('--val_split',  type=float, default=0.15,
                   help='Fraction of samples to use as validation (default: 0.15)')
    p.add_argument('--seed',       type=int,   default=42)
    return p.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)

    training_dir = os.path.join(args.kitti_root, 'data_road', 'training')
    if not os.path.isdir(training_dir):
        # Try without data_road/ prefix
        training_dir = os.path.join(args.kitti_root, 'training')
    if not os.path.isdir(training_dir):
        print(f"ERROR: Could not find training/ under {args.kitti_root}")
        print("Expected: kitti_root/data_road/training/image_2/  ...")
        return

    print(f"[KITTI] Scanning {training_dir} ...")
    samples = discover_samples(training_dir)
    print(f"  Found {len(samples)} labelled samples")

    if not samples:
        print("No samples found. Check that data_road/ is correctly extracted.")
        return

    has_lidar = any(s['velo_path'] for s in samples)
    has_calib = any(s['calib_path'] for s in samples)
    lidar_count = sum(1 for s in samples if s['velo_path'])
    mode_tag = "fusion mode" if lidar_count > 0 else "camera-only mode"
    print(f"  LiDAR available: {lidar_count}/{len(samples)} samples  [{mode_tag}]")

    # Train / val split
    random.shuffle(samples)
    n_val = max(1, int(len(samples) * args.val_split))
    val_samples   = samples[:n_val]
    train_samples = samples[n_val:]
    print(f"  Split: {len(train_samples)} train / {len(val_samples)} val")

    os.makedirs(args.output_dir, exist_ok=True)

    n_train = n_val_ok = 0
    print("\n[KITTI] Writing training samples ...")
    for i, s in enumerate(tqdm(train_samples, desc="  train")):
        if write_sample(s, os.path.join(args.output_dir, 'train'), i, has_lidar):
            n_train += 1

    print("[KITTI] Writing validation samples ...")
    for i, s in enumerate(tqdm(val_samples, desc="  val")):
        if write_sample(s, os.path.join(args.output_dir, 'val'), i, has_lidar):
            n_val_ok += 1

    # Summary
    lidar_note = "YES - depths/ created (channel 4 = real LiDAR depth)" if has_lidar else "NO  - channel 4 = all-ones"
    print("KITTI preparation complete")
    print(f"  Output : {os.path.abspath(args.output_dir)}")
    print(f"  Train  : {n_train} samples  (80%)")
    print(f"  Val    : {n_val_ok} samples  (20%)")
    print(f"  LiDAR  : {lidar_note}")
    print("")
    print("Next step:")
    print(f"  python train_pt.py --data_dir {args.output_dir} --epochs 50 --lr 3e-4")


if __name__ == '__main__':
    main()
