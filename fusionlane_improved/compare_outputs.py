"""
FusionLane — output comparison tool.

Generates side-by-side comparison panels that place reference outputs from
the original FusionLane model alongside outputs from FusionLane Improved.

Reference outputs (outputs/reference/):
  output_60epochs          LiDAR BEV + lane predictions, 60-epoch model
  output_boosted           LiDAR BEV + lane predictions, boosted/post-processed
  output_images            Camera-only mode, low-confidence output
  output_images_pretrained LiDAR BEV, camera-pretrained weights
  output_pretrained        KITTI-style red/magenta segmentation, pretrained model

Our outputs (outputs/inference_videoX_improved/):
  cleaned/                 Binary lane mask, morphologically cleaned
  kitti_seg/               KITTI-style segmentation (red bg, magenta lane)
  fitted/                  Polynomial curve fits

Usage
-----
  # Compare specific frames
  python compare_outputs.py --frames 0 10 20 30 40 50

  # Compare frames from a specific video run
  python compare_outputs.py --our_dir ./outputs/inference_video2_improved --frames 0 20 40

  # Auto-select evenly spaced frames
  python compare_outputs.py --n_frames 8 --our_dir ./outputs/inference_video2_improved

  # Save comparison grid to a specific location
  python compare_outputs.py --output_file ./outputs/comparison_grid.png
"""

import argparse
import os
import sys

import cv2
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Source definitions
# ─────────────────────────────────────────────────────────────────────────────

REFERENCE_SOURCES = {
    "ref_60epoch":        ("outputs/reference/output_60epochs",        "60-Epoch (LiDAR BEV)"),
    "ref_boosted":        ("outputs/reference/output_boosted",          "Boosted (LiDAR BEV)"),
    "ref_img_pretrained": ("outputs/reference/output_images_pretrained","Img-Pretrained (BEV)"),
    "ref_pretrained":     ("outputs/reference/output_pretrained",       "Pretrained (KITTI seg)"),
    "ref_images":         ("outputs/reference/output_images",           "Cam-Only (sparse)"),
}

OUR_SUBFOLDERS = {
    "our_cleaned":   ("cleaned",   "Improved: Cleaned"),
    "our_kitti":     ("kitti_seg", "Improved: KITTI seg"),
    "our_fitted":    ("fitted",    "Improved: Poly fit"),
}


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

TARGET_H = 321    # match the reference output resolution
TARGET_W = 321

LABEL_FONT       = cv2.FONT_HERSHEY_SIMPLEX
LABEL_SCALE      = 0.38
LABEL_THICKNESS  = 1
LABEL_COLOR      = (255, 255, 255)
LABEL_BG_COLOR   = (40, 40, 40)
LABEL_PAD        = 18   # px of header per column


def _load_and_resize(path):
    img = cv2.imread(path)
    if img is None:
        # Return a dark placeholder
        img = np.zeros((TARGET_H, TARGET_W, 3), dtype=np.uint8)
        cv2.putText(img, "NOT FOUND", (10, TARGET_H // 2),
                    LABEL_FONT, 0.4, (80, 80, 80), 1)
        return img
    return cv2.resize(img, (TARGET_W, TARGET_H), interpolation=cv2.INTER_LINEAR)


def _add_column_header(img, label):
    """Add a dark header bar with label text above the image."""
    header = np.full((LABEL_PAD, TARGET_W, 3), LABEL_BG_COLOR, dtype=np.uint8)
    (tw, th), _ = cv2.getTextSize(label, LABEL_FONT, LABEL_SCALE, LABEL_THICKNESS)
    tx = max(0, (TARGET_W - tw) // 2)
    ty = (LABEL_PAD + th) // 2
    cv2.putText(header, label, (tx, ty), LABEL_FONT, LABEL_SCALE,
                LABEL_COLOR, LABEL_THICKNESS, cv2.LINE_AA)
    return np.vstack([header, img])


def _add_row_header(height, label, width=50):
    """Add a narrow dark sidebar with rotated frame label."""
    panel = np.full((height, width, 3), (30, 30, 30), dtype=np.uint8)
    # Write text rotated 90°
    tmp = np.full((width, height, 3), (30, 30, 30), dtype=np.uint8)
    (tw, th), _ = cv2.getTextSize(label, LABEL_FONT, LABEL_SCALE, LABEL_THICKNESS)
    tx = max(0, (height - tw) // 2)
    ty = (width + th) // 2
    cv2.putText(tmp, label, (tx, ty), LABEL_FONT, LABEL_SCALE,
                LABEL_COLOR, LABEL_THICKNESS, cv2.LINE_AA)
    panel = cv2.rotate(tmp, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return panel


def _separator(height, color=(60, 60, 60), width=2):
    return np.full((height, width, 3), color, dtype=np.uint8)


def find_available_frames(ref_dir, our_dir, our_subdir, n_frames):
    """Find frame indices present in both reference and our output directories."""
    ref_files  = sorted(f for f in os.listdir(ref_dir)  if f.endswith(".png"))
    our_path   = os.path.join(our_dir, our_subdir)
    our_files  = sorted(f for f in os.listdir(our_path) if f.endswith(".png")) if os.path.isdir(our_path) else []

    # Reference uses pred_XXXX.png; ours uses pred_XXXX.png too
    ref_idxs = set(int(f.split("_")[1].split(".")[0]) for f in ref_files)
    max_ref   = max(ref_idxs) if ref_idxs else 0

    step = max(1, max_ref // (n_frames - 1)) if n_frames > 1 else 1
    return [min(i * step, max_ref) for i in range(n_frames)]


# ─────────────────────────────────────────────────────────────────────────────
# Main comparison grid builder
# ─────────────────────────────────────────────────────────────────────────────

def build_grid(frame_indices, our_dir, output_path, show_ref, show_our):
    """
    Build a comparison grid:
      Rows    = selected frames
      Columns = selected reference sources + our output types
    """
    # Determine which sources to include
    col_sources = []
    if show_ref:
        for key, (path, label) in REFERENCE_SOURCES.items():
            full_path = os.path.join(os.path.dirname(__file__), path) \
                        if not os.path.isabs(path) else path
            if os.path.isdir(full_path):
                col_sources.append((full_path, label, "ref"))
            else:
                print(f"  [warn] Reference folder not found, skipping: {full_path}")

    if show_our and our_dir:
        for key, (subfolder, label) in OUR_SUBFOLDERS.items():
            full_path = os.path.join(our_dir, subfolder)
            if os.path.isdir(full_path):
                col_sources.append((full_path, label, "our"))
            else:
                print(f"  [warn] Our subfolder not found, skipping: {full_path}")

    if not col_sources:
        print("ERROR: No valid output directories found.")
        sys.exit(1)

    n_cols = len(col_sources)
    row_header_w = 55
    sep_w        = 3
    cell_h       = TARGET_H + LABEL_PAD

    print(f"\nBuilding comparison grid:")
    print(f"  Frames  : {frame_indices}")
    print(f"  Columns : {[label for _, label, _ in col_sources]}")

    rows = []
    for frame_idx in frame_indices:
        stem = f"pred_{frame_idx:04d}.png"
        cells = []
        for col_dir, label, kind in col_sources:
            img_path = os.path.join(col_dir, stem)
            img      = _load_and_resize(img_path)
            img      = _add_column_header(img, label)
            cells.append(img)

        # Assemble row with separators between reference and our sections
        row_parts = []
        prev_kind = None
        for i, ((_, _, kind), cell) in enumerate(zip(col_sources, cells)):
            if prev_kind == "ref" and kind == "our":
                # Insert a wider separator between reference and our columns
                row_parts.append(_separator(cell_h, color=(100, 180, 100), width=4))
            elif i > 0:
                row_parts.append(_separator(cell_h))
            row_parts.append(cell)
            prev_kind = kind

        row_img = np.hstack(row_parts)

        # Add row label (frame number)
        rh = _add_row_header(cell_h, f"f{frame_idx:03d}", width=row_header_w)
        row_img = np.hstack([rh, row_img])
        rows.append(row_img)

    grid = np.vstack(rows)

    # Add title bar at top
    title      = "FusionLane — Output Comparison"
    subtitle   = "Reference (original model)  |  Green divider  |  Improved model"
    total_w    = grid.shape[1]
    title_h    = 36
    title_bar  = np.full((title_h, total_w, 3), (20, 20, 40), dtype=np.uint8)
    cv2.putText(title_bar, title,    (10, 14),       LABEL_FONT, 0.55, (200,220,255), 1, cv2.LINE_AA)
    cv2.putText(title_bar, subtitle, (10, title_h - 6), LABEL_FONT, 0.34, (140,160,180), 1, cv2.LINE_AA)
    grid = np.vstack([title_bar, grid])

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    cv2.imwrite(output_path, grid)
    print(f"\nSaved comparison grid: {output_path}")
    print(f"  Size: {grid.shape[1]}×{grid.shape[0]} px")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="FusionLane output comparison grid")
    p.add_argument("--our_dir",     type=str,
                   default="./outputs/inference_video2_improved",
                   help="Directory containing our improved inference outputs "
                        "(must have cleaned/, kitti_seg/, fitted/ subfolders).")
    p.add_argument("--frames",      type=int, nargs="+", default=None,
                   help="Specific frame indices to include (e.g. --frames 0 10 20 30).")
    p.add_argument("--n_frames",    type=int, default=6,
                   help="Number of evenly-spaced frames to auto-select (default 6). "
                        "Ignored when --frames is provided.")
    p.add_argument("--output_file", type=str,
                   default="./outputs/comparison_grid.png",
                   help="Path to save the output comparison PNG.")
    p.add_argument("--no_ref",      action="store_true",
                   help="Exclude reference model columns.")
    p.add_argument("--no_our",      action="store_true",
                   help="Exclude our improved model columns.")
    return p.parse_args()


def main():
    args = parse_args()

    # Find frame indices
    if args.frames:
        frame_indices = args.frames
    else:
        # Auto-select from the first reference folder
        ref_dir = "./outputs/reference/output_60epochs"
        if os.path.isdir(ref_dir):
            files = sorted(f for f in os.listdir(ref_dir) if f.endswith(".png"))
            max_idx = int(files[-1].split("_")[1].split(".")[0]) if files else 67
        else:
            max_idx = 67
        step = max(1, max_idx // (args.n_frames - 1))
        frame_indices = [min(i * step, max_idx) for i in range(args.n_frames)]

    build_grid(
        frame_indices  = frame_indices,
        our_dir        = args.our_dir,
        output_path    = args.output_file,
        show_ref       = not args.no_ref,
        show_our       = not args.no_our,
    )


if __name__ == "__main__":
    main()
