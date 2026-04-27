"""
multi_sweep_bev.py — accumulate N consecutive LiDAR sweeps into one BEV.

Rationale
---------
A single LiDAR sweep is sparse: the scan-ring pattern leaves most of the BEV
pixels empty, and a thin dashed lane marking might contribute only a handful
of points to the frame. Stacking K consecutive sweeps using ego-motion
densifies the BEV: static features (lane markings, curbs) accumulate; moving
objects smear into faint ghosts that the class-weighted loss mostly ignores.

This script expects:
  • A folder of point cloud files, ordered (by name) into a sequence.
  • A poses file giving the ego-pose of each frame in a common "map" frame.

It transforms K consecutive frames into the target frame's coordinate
system, concatenates them, and rasterizes them as one BEV.

Poses file format (KITTI odometry style):
  One frame per line, 12 floats = the top 3 rows of a 4×4 transform from
  frame to map. Line i corresponds to the i-th point cloud (when sorted).
  If your poses are in a different format, convert to this first — a helper
  is provided at the bottom of this file.

Training-data generation workflow:
    python multi_sweep_bev.py \
        --pointclouds_dir /data/pointclouds \
        --poses_file      /data/poses.txt \
        --out_dir         /data/bev_accumulated \
        --num_sweeps      3 \
        --stride          1

This produces one BEV PNG pair per *target* frame.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np

from lidar_bev import (
    BevConfig,
    _add_bev_args,
    _cfg_from_args,
    load_point_cloud,
    project_to_bev,
)


# ---------------------------------------------------------------------------
# Pose handling
# ---------------------------------------------------------------------------

def load_poses(path: Path) -> np.ndarray:
    """Load KITTI-odometry-style poses: [N, 4, 4] float64.

    Each line in the file is 12 space-separated floats representing the
    top 3 rows of a 4×4 matrix (row-major).
    """
    rows = []
    with open(path) as f:
        for ln, line in enumerate(f, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            vals = line.split()
            if len(vals) != 12:
                raise ValueError(
                    f"{path}:{ln}: expected 12 floats (KITTI pose), got {len(vals)}"
                )
            mat = np.eye(4, dtype=np.float64)
            mat[:3, :] = np.asarray(vals, dtype=np.float64).reshape(3, 4)
            rows.append(mat)
    if not rows:
        raise ValueError(f"{path}: no pose rows parsed")
    return np.stack(rows, axis=0)


def transform_points(pc: np.ndarray, T: np.ndarray) -> np.ndarray:
    """Apply a 4×4 transform to a [N, 4] cloud's xyz; intensity passes through."""
    xyz1 = np.concatenate(
        [pc[:, :3], np.ones((pc.shape[0], 1), pc.dtype)], axis=1
    )                                                 # [N, 4]
    xyz_out = xyz1 @ T.T                              # [N, 4]
    return np.concatenate([xyz_out[:, :3], pc[:, 3:4]], axis=1)


def relative_transform(T_src_to_map: np.ndarray,
                       T_dst_to_map: np.ndarray) -> np.ndarray:
    """Compose: src→map then map→dst."""
    return np.linalg.inv(T_dst_to_map) @ T_src_to_map


# ---------------------------------------------------------------------------
# Accumulation
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AccumConfig:
    num_sweeps: int = 3           # how many frames to accumulate per output
    stride: int = 1               # frame-gap between the sweeps we stack
    center_policy: str = "last"   # "last" → target is newest frame;
                                  # "middle" → target is centre of the window


def accumulate_window(
    clouds: List[np.ndarray],
    poses:  List[np.ndarray],
    target_idx: int,
) -> np.ndarray:
    """Transform every cloud in `clouds` into the target frame and concatenate.

    Args:
      clouds:     list of [N_i, 4] arrays (same length as poses).
      poses:      list of 4×4 matrices, same length.
      target_idx: index into `clouds` / `poses` that defines the destination.

    Returns:
      ndarray [sum(N_i), 4] in the target frame.
    """
    assert len(clouds) == len(poses), "clouds and poses must be same length"
    T_dst = poses[target_idx]
    out = []
    for pc, T_src in zip(clouds, poses):
        T = relative_transform(T_src, T_dst)
        out.append(transform_points(pc, T))
    return np.concatenate(out, axis=0) if out else np.empty((0, 4), np.float32)


def _cloud_paths_in_dir(d: Path) -> List[Path]:
    """All supported point-cloud files in the folder, sorted by name."""
    exts = {".bin", ".pcd", ".npy"}
    return sorted(p for p in d.iterdir() if p.suffix.lower() in exts)


def run_accumulation(
    pointclouds_dir: Path,
    poses_file: Path,
    out_dir: Path,
    bev_cfg: BevConfig,
    accum_cfg: AccumConfig,
    max_frames: Optional[int] = None,
) -> int:
    """Accumulate and project every valid window. Returns count of outputs."""
    cloud_paths = _cloud_paths_in_dir(pointclouds_dir)
    poses = load_poses(poses_file)

    if len(cloud_paths) != len(poses):
        raise ValueError(
            f"Mismatch: {len(cloud_paths)} point-cloud files vs {len(poses)} poses"
        )
    if max_frames is not None:
        cloud_paths = cloud_paths[:max_frames]
        poses = poses[:max_frames]

    out_dir.mkdir(parents=True, exist_ok=True)

    K = accum_cfg.num_sweeps
    s = accum_cfg.stride
    window_span = (K - 1) * s
    if window_span >= len(cloud_paths):
        raise ValueError(
            f"Window span {window_span+1} frames > dataset length {len(cloud_paths)}"
        )

    # Determine target-frame offset within the window.
    if accum_cfg.center_policy == "last":
        target_in_window = K - 1
    elif accum_cfg.center_policy == "middle":
        target_in_window = K // 2
    else:
        raise ValueError(f"center_policy must be 'last' or 'middle', "
                         f"got {accum_cfg.center_policy!r}")

    from PIL import Image

    n_out = 0
    total = len(cloud_paths) - window_span
    for start in range(total):
        idxs = [start + i * s for i in range(K)]
        target_idx_global = idxs[target_in_window]
        target_stem = cloud_paths[target_idx_global].stem

        clouds = [load_point_cloud(cloud_paths[i]) for i in idxs]
        poses_w = [poses[i] for i in idxs]

        # target_idx in the local list is simply target_in_window
        merged = accumulate_window(clouds, poses_w, target_in_window)
        image, region = project_to_bev(merged, bev_cfg)

        Image.fromarray(image).save(out_dir / f"{target_stem}.png")
        Image.fromarray(region).save(out_dir / f"{target_stem}_region.png")

        if (n_out + 1) % 20 == 0 or n_out == 0:
            print(f"  [{n_out+1:>5}/{total}] {target_stem} "
                  f"(accumulated {merged.shape[0]} pts from {K} sweeps)")
        n_out += 1

    return n_out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description="Accumulate N LiDAR sweeps per output BEV using ego-poses.",
    )
    p.add_argument("--pointclouds_dir", type=Path, required=True)
    p.add_argument("--poses_file",      type=Path, required=True)
    p.add_argument("--out_dir",         type=Path, required=True)
    p.add_argument("--num_sweeps",      type=int,  default=3,
                   help="Frames to accumulate per output (default: 3)")
    p.add_argument("--stride",          type=int,  default=1,
                   help="Frame gap between stacked sweeps (default: 1)")
    p.add_argument("--center",          type=str,  default="last",
                   choices=["last", "middle"],
                   help="Which frame in the window is the target/origin")
    p.add_argument("--max_frames",      type=int,  default=None,
                   help="Cap input length for debugging")
    _add_bev_args(p)
    args = p.parse_args()

    bev_cfg   = _cfg_from_args(args)
    accum_cfg = AccumConfig(num_sweeps=args.num_sweeps,
                            stride=args.stride,
                            center_policy=args.center)

    print(f"BEV       : {bev_cfg.height}×{bev_cfg.width} at {bev_cfg.resolution} m/px")
    print(f"Accumulate: {accum_cfg.num_sweeps} sweeps, stride {accum_cfg.stride}, "
          f"target={accum_cfg.center_policy}")

    n = run_accumulation(
        args.pointclouds_dir, args.poses_file, args.out_dir,
        bev_cfg, accum_cfg, args.max_frames,
    )
    print(f"\nDone — {n} accumulated BEVs written to {args.out_dir}")


if __name__ == "__main__":
    main()
