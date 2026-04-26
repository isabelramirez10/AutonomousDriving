"""
FusionLane Baseline — classical Hough-line lane detector.

No training data required.  Works immediately on any road video.
Use this to:
  1. Verify your pipeline produces output before training the DL model.
  2. Compare DL model output against a classical CV baseline.
  3. Generate a reference overlay when no ground-truth masks exist.

Algorithm
---------
1. Convert frame to grayscale and apply Gaussian blur.
2. Mask a trapezoidal region of interest (road area, lower portion).
3. Canny edge detection.
4. Probabilistic Hough line transform.
5. Separate lines into left / right lanes by slope sign.
6. Fit a single line through each side and extend to ROI boundaries.
7. Draw semi-transparent coloured overlays.

Output
------
  outputs/baseline/
    overlay/        frame with left (blue) and right (yellow) lane lines
    edges/          Canny edge map inside ROI
    comparison/     side-by-side: original | edges | overlay
  outputs/baseline/lane_overlay.mp4   (if input is a video)
  outputs/baseline/metrics.csv        per-frame detection stats

Usage
-----
  python infer_baseline.py --input path/to/video.mp4
  python infer_baseline.py --input path/to/frames/  --output_dir ./out
"""

import argparse
import csv
import os

import cv2
import numpy as np
from tqdm import tqdm

# ─────────────────────────────────────────────────────────────────────────────
# Config defaults
# ─────────────────────────────────────────────────────────────────────────────

CONFIG = {
    "input":       "./my_road_video.mp4",
    "output_dir":  "./outputs/baseline",
    # ROI: fraction of frame height where the road trapezoid starts
    "roi_top":     0.55,
    # Hough parameters
    "hough_threshold":     50,
    "hough_min_length":    80,
    "hough_max_gap":       50,
    # Canny thresholds
    "canny_low":   50,
    "canny_high":  150,
    # Overlay alpha blend
    "alpha":       0.4,
}

_IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
_VID_EXTS = {'.mp4', '.avi', '.mov', '.mkv', '.m4v', '.wmv', '.flv'}


# ─────────────────────────────────────────────────────────────────────────────
# Frame reader (same interface as infer_media.py)
# ─────────────────────────────────────────────────────────────────────────────

def read_frames(input_path):
    ext = os.path.splitext(input_path)[1].lower()

    if ext in _IMG_EXTS and os.path.isfile(input_path):
        frame = cv2.imread(input_path)
        if frame is None:
            raise IOError(f"Cannot read image: {input_path}")
        yield frame
    elif os.path.isdir(input_path):
        files = sorted(
            os.path.join(input_path, f)
            for f in os.listdir(input_path)
            if os.path.splitext(f)[1].lower() in _IMG_EXTS
        )
        if not files:
            raise ValueError(f"No images found in {input_path}")
        print(f"Found {len(files)} images.")
        for p in files:
            frame = cv2.imread(p)
            if frame is not None:
                yield frame
    elif ext in _VID_EXTS and os.path.isfile(input_path):
        cap   = cv2.VideoCapture(input_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps   = cap.get(cv2.CAP_PROP_FPS) or 30
        print(f"Video: {total} frames @ {fps:.1f} fps — {os.path.basename(input_path)}")
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            yield frame
        cap.release()
    else:
        raise ValueError(f"Unsupported input: {input_path}")


# ─────────────────────────────────────────────────────────────────────────────
# ROI
# ─────────────────────────────────────────────────────────────────────────────

def roi_mask(h, w, roi_top):
    """Trapezoidal mask covering the road region."""
    pts = np.array([[
        (0,          h),
        (w,          h),
        (int(w * 0.62), int(h * roi_top)),
        (int(w * 0.38), int(h * roi_top)),
    ]], dtype=np.int32)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, pts, 255)
    return mask, pts[0]


# ─────────────────────────────────────────────────────────────────────────────
# Hough detection
# ─────────────────────────────────────────────────────────────────────────────

def detect_edges(frame_bgr, canny_low, canny_high, roi_top):
    h, w  = frame_bgr.shape[:2]
    gray  = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    blur  = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, canny_low, canny_high)
    mask, roi_pts = roi_mask(h, w, roi_top)
    return cv2.bitwise_and(edges, mask), roi_pts


def hough_lines(edges, threshold, min_length, max_gap):
    return cv2.HoughLinesP(
        edges, rho=1, theta=np.pi / 180,
        threshold=threshold,
        minLineLength=min_length,
        maxLineGap=max_gap,
    )


def fit_lane(line_group, h, roi_top):
    """Fit a single extended line through a group of raw Hough segments."""
    if not line_group:
        return None
    pts = np.array(
        [[x, y]
         for x1, y1, x2, y2 in line_group
         for x, y in [(x1, y1), (x2, y2)]],
        dtype=np.float32,
    )
    if len(pts) < 2:
        return None
    line = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01).flatten()
    vx, vy, x0, y0 = float(line[0]), float(line[1]), float(line[2]), float(line[3])
    if abs(vy) < 1e-6:
        return None
    y_bot = h
    y_top = int(h * roi_top)
    x_bot = int(x0 + (y_bot - y0) / vy * vx)
    x_top = int(x0 + (y_top - y0) / vy * vx)
    return (x_bot, y_bot), (x_top, y_top)


def separate_lanes(lines, h, roi_top, slope_thresh=0.3):
    """Split Hough segments into left / right lanes, fit each."""
    left_raw, right_raw = [], []
    if lines is not None:
        for seg in lines:
            x1, y1, x2, y2 = seg[0]
            if x2 == x1:
                continue
            slope = (y2 - y1) / (x2 - x1)
            if abs(slope) < slope_thresh:
                continue                         # near-horizontal → noise
            if slope < 0:
                left_raw.append((x1, y1, x2, y2))
            else:
                right_raw.append((x1, y1, x2, y2))
    left  = fit_lane(left_raw,  h, roi_top)
    right = fit_lane(right_raw, h, roi_top)
    return left, right


# ─────────────────────────────────────────────────────────────────────────────
# Drawing
# ─────────────────────────────────────────────────────────────────────────────

_LEFT_COLOR  = (255, 100,   0)   # blue
_RIGHT_COLOR = (  0, 200, 255)   # yellow
_LINE_THICK  = 6


def draw_overlay(frame_bgr, left_lane, right_lane, roi_pts, alpha):
    """Draw semi-transparent lane lines and ROI boundary."""
    overlay = frame_bgr.copy()
    h, w    = frame_bgr.shape[:2]

    # ROI boundary (thin white)
    cv2.polylines(overlay, [roi_pts], isClosed=True,
                  color=(200, 200, 200), thickness=1)

    # Lane lines
    if left_lane:
        cv2.line(overlay, left_lane[0], left_lane[1], _LEFT_COLOR,  _LINE_THICK)
    if right_lane:
        cv2.line(overlay, right_lane[0], right_lane[1], _RIGHT_COLOR, _LINE_THICK)

    # Fill between lanes (if both detected)
    if left_lane and right_lane:
        poly = np.array([
            left_lane[0], left_lane[1],
            right_lane[1], right_lane[0],
        ], dtype=np.int32)
        road_layer = frame_bgr.copy()
        cv2.fillPoly(road_layer, [poly], (0, 255, 100))
        overlay = cv2.addWeighted(overlay, 1.0, road_layer, 0.18, 0)

    return cv2.addWeighted(frame_bgr, 1 - alpha, overlay, alpha, 0)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="FusionLane classical baseline")
    p.add_argument("--input",             type=str,   default=CONFIG["input"])
    p.add_argument("--output_dir",        type=str,   default=CONFIG["output_dir"])
    p.add_argument("--roi_top",           type=float, default=CONFIG["roi_top"])
    p.add_argument("--hough_threshold",   type=int,   default=CONFIG["hough_threshold"])
    p.add_argument("--hough_min_length",  type=int,   default=CONFIG["hough_min_length"])
    p.add_argument("--hough_max_gap",     type=int,   default=CONFIG["hough_max_gap"])
    p.add_argument("--canny_low",         type=int,   default=CONFIG["canny_low"])
    p.add_argument("--canny_high",        type=int,   default=CONFIG["canny_high"])
    p.add_argument("--alpha",             type=float, default=CONFIG["alpha"])
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    for sub in ("overlay", "edges", "comparison"):
        os.makedirs(os.path.join(args.output_dir, sub), exist_ok=True)

    ext      = os.path.splitext(args.input)[1].lower()
    is_video = ext in _VID_EXTS and os.path.isfile(args.input)

    all_frames   = list(read_frames(args.input))
    n_total      = len(all_frames)
    video_writer = None
    overlay_path = os.path.join(args.output_dir, "lane_overlay.mp4")
    frames_out   = []
    metrics_rows = []

    print(f"Processing {n_total} frame(s) with Hough baseline ...")

    for idx, frame in enumerate(tqdm(all_frames, desc="baseline")):
        h, w = frame.shape[:2]

        masked_edges, roi_pts = detect_edges(
            frame, args.canny_low, args.canny_high, args.roi_top
        )
        lines = hough_lines(
            masked_edges,
            args.hough_threshold,
            args.hough_min_length,
            args.hough_max_gap,
        )
        left, right = separate_lanes(lines, h, args.roi_top)
        ov = draw_overlay(frame, left, right, roi_pts, args.alpha)

        # Comparison panel: original | edges (3ch) | overlay
        edges_bgr = cv2.cvtColor(masked_edges, cv2.COLOR_GRAY2BGR)
        panel     = np.concatenate([frame, edges_bgr, ov], axis=1)

        stem = f"frame_{idx:04d}"
        cv2.imwrite(os.path.join(args.output_dir, "overlay",     stem + ".jpg"), ov)
        cv2.imwrite(os.path.join(args.output_dir, "edges",       stem + ".jpg"), edges_bgr)
        cv2.imwrite(os.path.join(args.output_dir, "comparison",  stem + ".jpg"), panel)

        # Per-frame metrics
        n_hough = len(lines) if lines is not None else 0
        metrics_rows.append({
            "frame":        idx,
            "hough_lines":  n_hough,
            "left_detected":  int(left  is not None),
            "right_detected": int(right is not None),
        })

        if is_video:
            frames_out.append(ov)

    # Write overlay video
    if is_video and frames_out:
        oh, ow = frames_out[0].shape[:2]
        cap = cv2.VideoCapture(args.input)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        cap.release()
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        vw = cv2.VideoWriter(overlay_path, fourcc, fps, (ow, oh))
        for fr in frames_out:
            vw.write(fr)
        vw.release()
        print(f"Overlay video saved: {overlay_path}")

    # Write metrics CSV
    metrics_path = os.path.join(args.output_dir, "metrics.csv")
    with open(metrics_path, "w", newline="") as mf:
        writer = csv.DictWriter(
            mf, fieldnames=["frame", "hough_lines", "left_detected", "right_detected"]
        )
        writer.writeheader()
        writer.writerows(metrics_rows)

    # Summary
    left_pct  = sum(r["left_detected"]  for r in metrics_rows) / n_total * 100
    right_pct = sum(r["right_detected"] for r in metrics_rows) / n_total * 100
    avg_lines = sum(r["hough_lines"]    for r in metrics_rows) / n_total

    print(f"\nBaseline summary ({n_total} frames):")
    print(f"  Left  lane detected : {left_pct:.1f}% of frames")
    print(f"  Right lane detected : {right_pct:.1f}% of frames")
    print(f"  Avg Hough segments  : {avg_lines:.1f} per frame")
    print(f"\nOutputs in {args.output_dir}/")
    print("  overlay/     — coloured lane line overlay")
    print("  edges/       — Canny edge map inside ROI")
    print("  comparison/  — original | edges | overlay (side-by-side)")
    if is_video:
        print(f"  lane_overlay.mp4")
    print(f"  metrics.csv")


if __name__ == "__main__":
    main()
