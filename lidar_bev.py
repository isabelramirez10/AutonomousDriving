"""
lidar_bev.py — project a LiDAR point cloud to a FusionLane-compatible BEV.

Takes a point cloud (x, y, z, intensity) and rasterizes it into:
  • image.png : 3 channels (intensity, height, density) uint8  [H, W, 3]
  • region.png: 1 channel  (ROI mask)                 uint8  [H, W]

Defaults to a 321×321 vehicle-centered BEV at 0.1 m/pixel (≈32×32 m area).
Every parameter is overridable — you will almost certainly want to tune
extent / resolution to match whatever the model was trained on.

The three-channel encoding is one reasonable choice; if your original
training used a different encoding (e.g. only intensity, or RGB from
camera), swap the channel-packing function accordingly.

Supported inputs:
  • .bin  — KITTI float32 [N, 4] (x, y, z, intensity)
  • .pcd  — Point Cloud Data format (via open3d; intensity if present)
  • .npy  — numpy array [N, 4] or [N, 3] (intensity defaults to 0)

Axes convention:
  LiDAR frame:  +x forward, +y left, +z up (right-handed, KITTI-style).
  BEV image:    col increases with +y (left-right flipped on screen),
                row decreases with +x (forward is "up" in the image).
  Vehicle is at image center by default.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Point cloud I/O
# ---------------------------------------------------------------------------

def load_point_cloud(path: Path) -> np.ndarray:
    """Load a point cloud as an ndarray [N, 4]: (x, y, z, intensity).

    Intensity defaults to 0 if the source format doesn't provide it.
    """
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".bin":
        # KITTI convention: flat float32, four values per point.
        arr = np.fromfile(str(path), dtype=np.float32)
        if arr.size % 4 != 0:
            raise ValueError(
                f"{path}: .bin size {arr.size} not divisible by 4 "
                "(expected x,y,z,intensity per point)"
            )
        return arr.reshape(-1, 4)

    if suffix == ".npy":
        arr = np.load(str(path))
        if arr.ndim != 2 or arr.shape[1] not in (3, 4):
            raise ValueError(f"{path}: expected [N, 3] or [N, 4], got {arr.shape}")
        if arr.shape[1] == 3:
            arr = np.concatenate([arr, np.zeros((arr.shape[0], 1), arr.dtype)], axis=1)
        return arr.astype(np.float32, copy=False)

    if suffix == ".pcd":
        try:
            import open3d as o3d
        except ImportError as e:
            raise ImportError(
                "open3d is required for .pcd files. `pip install open3d`."
            ) from e
        # Use tensor API so we can access custom fields like `intensity`.
        pc = o3d.t.io.read_point_cloud(str(path))
        xyz = pc.point["positions"].numpy().astype(np.float32)
        if "intensity" in pc.point:
            inten = pc.point["intensity"].numpy().astype(np.float32).reshape(-1, 1)
        else:
            inten = np.zeros((xyz.shape[0], 1), dtype=np.float32)
        return np.concatenate([xyz, inten], axis=1)

    raise ValueError(f"{path}: unsupported extension {suffix}")


# ---------------------------------------------------------------------------
# BEV configuration and projection
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BevConfig:
    """BEV rasterization parameters.

    All distances in meters. Default is a 32.1 × 32.1 m area centered on the
    vehicle at 0.1 m/pixel → 321 × 321 image.
    """
    x_min: float = -16.05
    x_max: float =  16.05
    y_min: float = -16.05
    y_max: float =  16.05
    z_min: float =  -2.5   # exclude far-below-ground noise
    z_max: float =   1.0   # exclude tall structures (trees, bridges) that swamp intensity
    resolution: float = 0.1  # meters per pixel

    @property
    def height(self) -> int:
        return int(round((self.x_max - self.x_min) / self.resolution))

    @property
    def width(self) -> int:
        return int(round((self.y_max - self.y_min) / self.resolution))


def _filter_and_index(pc: np.ndarray, cfg: BevConfig) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Crop the cloud to the BEV frustum and return point rows, cols, and the
    filtered point cloud itself."""
    x, y, z = pc[:, 0], pc[:, 1], pc[:, 2]
    mask = (
        (x >= cfg.x_min) & (x < cfg.x_max) &
        (y >= cfg.y_min) & (y < cfg.y_max) &
        (z >= cfg.z_min) & (z < cfg.z_max)
    )
    pc_f = pc[mask]
    if pc_f.size == 0:
        return (np.empty(0, np.int64), np.empty(0, np.int64), pc_f)

    # x-forward is "up" in the image → row = H - 1 - (x - x_min)/res
    rows = (cfg.height - 1) - ((pc_f[:, 0] - cfg.x_min) / cfg.resolution).astype(np.int64)
    # +y is "right" in image frame
    cols = ((pc_f[:, 1] - cfg.y_min) / cfg.resolution).astype(np.int64)

    rows = np.clip(rows, 0, cfg.height - 1)
    cols = np.clip(cols, 0, cfg.width  - 1)
    return rows, cols, pc_f


def _normalize_to_uint8(channel: np.ndarray, clip_percentile: float = 99.0) -> np.ndarray:
    """Rescale a float channel to uint8 [0, 255] using a percentile-based max.

    Using the 99th percentile (rather than the true max) prevents a handful of
    outlier points with huge intensity from flattening everything else to 0.
    """
    if channel.size == 0:
        return np.zeros_like(channel, dtype=np.uint8)
    hi = np.percentile(channel[channel > 0], clip_percentile) if np.any(channel > 0) else 1.0
    hi = max(hi, 1e-6)
    out = np.clip(channel / hi, 0.0, 1.0) * 255.0
    return out.astype(np.uint8)


def project_to_bev(
    pc: np.ndarray,
    cfg: BevConfig = BevConfig(),
) -> Tuple[np.ndarray, np.ndarray]:
    """Project a point cloud to BEV.

    Args:
      pc:  ndarray [N, 4] — (x, y, z, intensity).
      cfg: BevConfig.

    Returns:
      image:  uint8 [H, W, 3] — (intensity_max, height_max, density) channels.
      region: uint8 [H, W]    — binary-like ROI mask (density > 0), 0 or 255.
    """
    H, W = cfg.height, cfg.width
    rows, cols, pc_f = _filter_and_index(pc, cfg)

    intensity_max = np.zeros((H, W), dtype=np.float32)
    height_max    = np.full ((H, W), cfg.z_min, dtype=np.float32)
    density       = np.zeros((H, W), dtype=np.int32)

    if pc_f.size > 0:
        flat = rows * W + cols
        # np.maximum.at is a scatter-max; slower than add.at but correct for max-reduce.
        np.maximum.at(intensity_max.ravel(), flat, pc_f[:, 3])
        np.maximum.at(height_max.ravel(),    flat, pc_f[:, 2])
        np.add.at    (density.ravel(),       flat, 1)

    # Channel 0: intensity, percentile-clipped to uint8.
    ch_int = _normalize_to_uint8(intensity_max, clip_percentile=99.0)
    # Channel 1: height above ground, linear z_min..z_max → 0..255.
    ch_hgt = np.clip(
        (height_max - cfg.z_min) / max(cfg.z_max - cfg.z_min, 1e-6), 0.0, 1.0
    )
    ch_hgt = (ch_hgt * 255.0).astype(np.uint8)
    ch_hgt[density == 0] = 0  # empty pixels shouldn't be "at ground level"
    # Channel 2: log-density, for the dense-ring texture the model was trained on.
    ch_den = _normalize_to_uint8(np.log1p(density.astype(np.float32)), clip_percentile=99.0)

    image = np.stack([ch_int, ch_hgt, ch_den], axis=2)

    # Region: any pixel that got at least one return. Matches a "coverage" mask,
    # which is a reasonable proxy for road-ROI when combined with the z-filter.
    region = np.where(density > 0, 255, 0).astype(np.uint8)
    return image, region


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _add_bev_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--x_min", type=float, default=-16.05)
    p.add_argument("--x_max", type=float, default= 16.05)
    p.add_argument("--y_min", type=float, default=-16.05)
    p.add_argument("--y_max", type=float, default= 16.05)
    p.add_argument("--z_min", type=float, default= -2.5)
    p.add_argument("--z_max", type=float, default=  1.0)
    p.add_argument("--resolution", type=float, default=0.1,
                   help="Meters per pixel (default: 0.1)")


def _cfg_from_args(args) -> BevConfig:
    return BevConfig(
        x_min=args.x_min, x_max=args.x_max,
        y_min=args.y_min, y_max=args.y_max,
        z_min=args.z_min, z_max=args.z_max,
        resolution=args.resolution,
    )


def main():
    p = argparse.ArgumentParser(
        description="Project a single LiDAR point cloud to BEV (image.png + region.png)."
    )
    p.add_argument("input",  type=Path, help="Path to .bin / .pcd / .npy point cloud")
    p.add_argument("out_dir", type=Path, help="Output directory")
    p.add_argument("--stem",  type=str, default=None,
                   help="Output filename stem (default: input filename stem)")
    _add_bev_args(p)
    args = p.parse_args()

    cfg = _cfg_from_args(args)
    pc  = load_point_cloud(args.input)
    image, region = project_to_bev(pc, cfg)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = args.stem or args.input.stem

    from PIL import Image
    Image.fromarray(image).save(args.out_dir / f"{stem}.png")
    Image.fromarray(region).save(args.out_dir / f"{stem}_region.png")

    kept = int((region > 0).sum())
    print(f"Input : {args.input}  ({pc.shape[0]} points)")
    print(f"Output: {args.out_dir}/{stem}.png  (+ _region.png)")
    print(f"BEV   : {cfg.height}×{cfg.width} at {cfg.resolution} m/px, "
          f"extent x[{cfg.x_min:.1f}, {cfg.x_max:.1f}] y[{cfg.y_min:.1f}, {cfg.y_max:.1f}]")
    print(f"Filled pixels: {kept} / {cfg.height * cfg.width} "
          f"({100*kept/(cfg.height*cfg.width):.2f}%)")


if __name__ == "__main__":
    main()
