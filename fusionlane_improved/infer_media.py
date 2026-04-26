"""
FusionLane Improved — media inference script.

PURPOSE
-------
Run lane detection on YOUR OWN road images or dashcam video — no TFRecords needed.
Just point the script at a file or folder and it does the rest.

SUPPORTED INPUT
---------------
  Single video file    : --input path/to/dashcam.mp4   (also .avi .mov .mkv)
  Folder of images     : --input path/to/frames/        (JPG / PNG / BMP)
  Single image file    : --input path/to/road.jpg

OUTPUT
------
  outputs/inference/
    raw/          argmax predictions (white = lane, black = background)
    cleaned/      confidence-filtered + blob-cleaned predictions
    heatmap/      P(lane) confidence map (blue=low, red=high)
    comparison/   4-panel side-by-side (original | raw | cleaned | heatmap)
  If --input is a video: outputs/lane_overlay.mp4 is also written
    (original video with lane overlay drawn in green)
"""

import argparse
import csv
import os

import cv2
import numpy as np
import torch
from scipy import ndimage
from tqdm import tqdm

from train_pt import SimpleFusionLaneNet, compute_miou


# ─────────────────────────────────────────────────────────────────────────────
# Optional DPT depth estimator (pseudo-LiDAR for camera-only videos)
# Activated by --use_depth flag.  Requires `transformers` package.
# ─────────────────────────────────────────────────────────────────────────────

_DPT_MODEL_ID = "Intel/dpt-swin2-tiny-256"   # ~100 MB, fast on CPU

def load_depth_estimator():
    """Download and return (DPT model, image processor)."""
    try:
        from transformers import DPTForDepthEstimation, DPTImageProcessor
    except ImportError:
        raise ImportError("pip install transformers  — required for --use_depth")
    from PIL import Image as _PIL
    print(f"[depth] Loading DPT model: {_DPT_MODEL_ID} (downloads ~100 MB on first run)")
    processor = DPTImageProcessor.from_pretrained(_DPT_MODEL_ID)
    model     = DPTForDepthEstimation.from_pretrained(_DPT_MODEL_ID)
    model.eval()
    print("[depth] DPT model ready.")
    return model, processor


@torch.no_grad()
def estimate_depth(frame_bgr, dpt_model, dpt_processor, out_h, out_w):
    """
    Run DPT on a single BGR frame.
    Returns float32 [out_h, out_w] depth map normalised to [0, 1].
    Higher values = closer to camera (inverse depth convention).
    """
    from PIL import Image as _PIL
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil = _PIL.fromarray(rgb)
    inputs = dpt_processor(images=pil, return_tensors="pt")
    out    = dpt_model(**inputs).predicted_depth.squeeze().numpy()
    mn, mx = out.min(), out.max()
    norm   = (out - mn) / (mx - mn + 1e-6)
    return cv2.resize(norm.astype(np.float32), (out_w, out_h),
                      interpolation=cv2.INTER_LINEAR)

# =============================================================================
# EASY CONFIG — edit these to run without typing command-line flags.
# =============================================================================
CONFIG = {
    # ── Input ────────────────────────────────────────────────────────────────
    "input":        "./my_road_video.mp4",   # video file | image folder | image file
    # ── Model ────────────────────────────────────────────────────────────────
    "model_path":   "./outputs/best_model.pth",
    # ── Output ───────────────────────────────────────────────────────────────
    "output_dir":   "./outputs/inference",
    # ── Image size (must match what you trained with) ─────────────────────────
    "image_height": 256,
    "image_width":  256,
    # ── Post-processing ───────────────────────────────────────────────────────
    "threshold":    0.25,   # lowered: captures more lane pixels on dashcam footage
    "min_blob":     50,     # lowered: keeps smaller lane detections
    # ── Road ROI mask ─────────────────────────────────────────────────────────
    "roi_top":      0.40,   # ignore top 40% of frame (sky/horizon) during inference
    # ── Performance ───────────────────────────────────────────────────────────
    "batch_size":   4,      # frames processed at once (must be divisible by 4)
}
# =============================================================================


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(model_path, device):
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {model_path}\n"
            "Run train_pt.py first to generate best_model.pth."
        )
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    # Use build_model so the architecture always matches the saved checkpoint
    # (handles both SimpleFusionLaneNet and FusionLaneModel / ResNet-18 UNet)
    from train_pt import build_model
    model = build_model(num_classes=2).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded model from {model_path}")
    return model


# ---------------------------------------------------------------------------
# Input readers
# ---------------------------------------------------------------------------

_IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
_VID_EXTS = {'.mp4', '.avi', '.mov', '.mkv', '.m4v', '.wmv', '.flv'}


def read_frames(input_path):
    """
    Yield (frame_bgr, original_hw) tuples from any supported source.
    frame_bgr : uint8 numpy [H, W, 3]  in BGR colour space
    original_hw: (height, width) before any resizing
    """
    ext = os.path.splitext(input_path)[1].lower()

    # Single image file
    if ext in _IMG_EXTS and os.path.isfile(input_path):
        frame = cv2.imread(input_path)
        if frame is None:
            raise IOError(f"Cannot read image: {input_path}")
        yield frame, frame.shape[:2]

    # Image folder
    elif os.path.isdir(input_path):
        files = sorted(
            os.path.join(input_path, f)
            for f in os.listdir(input_path)
            if os.path.splitext(f)[1].lower() in _IMG_EXTS
        )
        if not files:
            raise ValueError(f"No images found in {input_path}")
        print(f"Found {len(files)} images in {input_path}")
        for path in files:
            frame = cv2.imread(path)
            if frame is not None:
                yield frame, frame.shape[:2]

    # Video file
    elif ext in _VID_EXTS and os.path.isfile(input_path):
        cap   = cv2.VideoCapture(input_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps   = cap.get(cv2.CAP_PROP_FPS) or 30
        print(f"Video: {total} frames at {fps:.1f} fps "
              f"({total/fps:.1f}s) — {os.path.basename(input_path)}")
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            yield frame, frame.shape[:2]
        cap.release()

    else:
        raise ValueError(
            f"Unsupported input: {input_path}\n"
            "Expected: video file (.mp4 .avi .mov .mkv), "
            "image file (.jpg .png .bmp), or image folder."
        )


# ---------------------------------------------------------------------------
# Pre-processing
# ---------------------------------------------------------------------------

_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)


# ---------------------------------------------------------------------------
# Preprocessing enhancements
# ---------------------------------------------------------------------------

_clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

def apply_clahe(frame_bgr):
    """
    Adaptive contrast enhancement via CLAHE on the L* channel of LAB color space.
    Improves local contrast in under- or over-exposed frames before model input.
    Useful when inference footage has different brightness/exposure than training data.
    """
    lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
    lab[:, :, 0] = _clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def smooth_confidence(curr_conf, prev_conf, alpha):
    """
    Exponential moving average of the confidence map across consecutive frames.
      smooth_t = alpha * smooth_{t-1} + (1 - alpha) * raw_t

    Reduces frame-to-frame flicker in the binary output mask by smoothing
    the continuous confidence values before thresholding.

    alpha=0.0  → no smoothing (each frame independent)
    alpha=0.6  → moderate smoothing (recommended for dashcam video)
    alpha=0.8  → heavy smoothing (very slow-moving or stable camera)
    """
    if prev_conf is None or alpha <= 0.0:
        return curr_conf
    return alpha * prev_conf + (1.0 - alpha) * curr_conf


def adaptive_otsu_threshold(conf_np, floor=0.10, ceiling=0.70):
    """
    Per-frame automatic threshold computed via Otsu's method on the confidence map.

    Instead of using a fixed threshold value, this finds the optimal binary split
    in the confidence histogram for each individual frame. Useful when:
      - Inference footage has different lighting or dynamic range than training data
      - Scene brightness shifts between frames (tunnels, sunrise/sunset, shadows)
      - You want to avoid manually tuning --threshold for different environments

    The result is clamped to [floor, ceiling] to prevent extreme values.
    """
    conf_u8 = np.clip(conf_np * 255, 0, 255).astype(np.uint8)
    otsu_val, _ = cv2.threshold(conf_u8, 0, 255,
                                cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return float(np.clip(otsu_val / 255.0, floor, ceiling))


def preprocess(frame_bgr, H, W, depth_map=None):
    """
    BGR uint8 frame → normalised float32 tensor [4, H, W].
    depth_map: optional float32 [H, W] in [0,1] (DPT pseudo-LiDAR).
               When None, channel 4 = all-ones (camera-only mode).
    """
    rgb = cv2.cvtColor(
        cv2.resize(frame_bgr, (W, H), interpolation=cv2.INTER_LINEAR),
        cv2.COLOR_BGR2RGB,
    ).astype(np.float32)                               # [H, W, 3]
    rgb = rgb.transpose(2, 0, 1) / 255.0              # [3, H, W]
    rgb = (rgb - _MEAN) / _STD                        # normalise

    if depth_map is not None:
        reg = depth_map[np.newaxis].astype(np.float32)
    else:
        reg = np.ones((1, H, W), dtype=np.float32)

    return torch.from_numpy(np.concatenate([rgb, reg], axis=0))  # [4, H, W]


def apply_roi_mask(mask_hw, roi_top):
    """Zero out predicted lane pixels in the top roi_top fraction (sky region)."""
    if roi_top <= 0.0:
        return mask_hw
    cut = int(mask_hw.shape[0] * roi_top)
    out = mask_hw.copy()
    out[:cut, :] = 0
    return out


# ---------------------------------------------------------------------------
# Post-processing
# ---------------------------------------------------------------------------

def confidence_filter(logits, threshold):
    """logits [B,2,H,W] → (binary mask [B,H,W], confidence [B,H,W])"""
    probs = torch.softmax(logits, dim=1)
    conf  = probs[:, 1]
    return (conf >= threshold).to(torch.uint8), conf


def clean_mask(mask_np, min_blob):
    """Remove connected components smaller than min_blob pixels."""
    m = mask_np.astype(bool)
    m = ndimage.binary_opening(m, structure=np.ones((3, 3), dtype=bool))
    m = ndimage.binary_closing(m, structure=np.ones((5, 5), dtype=bool))
    labeled, n = ndimage.label(m)
    if n == 0:
        return np.zeros_like(mask_np, dtype=np.uint8)
    counts = np.bincount(labeled.ravel())
    keep   = np.zeros_like(m, dtype=bool)
    for i in range(1, len(counts)):
        if counts[i] >= min_blob:
            keep |= (labeled == i)
    return keep.astype(np.uint8)


def fit_lane_polynomials(clean_mask, degree=2, min_pixels=30, thickness=6):
    """
    Fit a 2nd-order polynomial to each connected lane component and rasterize
    smooth curves back onto the mask.

    For each blob:
      1. Collect pixel coordinates (x, y) of the component
      2. Fit  x = a·y² + b·y + c  via least-squares (column as function of row)
      3. Draw the fitted curve at the given thickness
      4. Fall back to the original blob if fitting fails (too few points, etc.)

    Produces thin, geometrically smooth lane lines rather than irregular blobs.
    Output goes to kitti_seg/ and fitted/ folders; the blob clean mask is kept
    for metrics and the overlay video.
    """
    H, W = clean_mask.shape
    fitted = np.zeros_like(clean_mask, dtype=np.uint8)
    labeled, n = ndimage.label(clean_mask.astype(bool))
    if n == 0:
        return fitted

    for comp_id in range(1, n + 1):
        ys, xs = np.where(labeled == comp_id)
        if len(ys) < min_pixels:
            continue
        y_span = int(ys.max()) - int(ys.min())
        if y_span < 5:                             # near-horizontal blob — skip
            fitted[labeled == comp_id] = 1
            continue
        try:
            # Fit x = a*y^2 + b*y + c  (column as function of row)
            coeffs = np.polyfit(ys, xs, degree)
            y_range = np.arange(int(ys.min()), min(int(ys.max()) + 1, H))
            x_fitted = np.polyval(coeffs, y_range).astype(int)
            half = thickness // 2
            for y, x in zip(y_range, x_fitted):
                x1, x2 = max(0, x - half), min(W, x + half + 1)
                fitted[y, x1:x2] = 1
        except (np.linalg.LinAlgError, ValueError):
            fitted[labeled == comp_id] = 1        # fallback: keep original blob
    return fitted


def make_heatmap(conf_np):
    """conf_np [H,W] float → BGR uint8 jet heatmap."""
    u8 = np.clip(conf_np * 255, 0, 255).astype(np.uint8)
    return cv2.applyColorMap(u8, cv2.COLORMAP_JET)


def overlay_lane(frame_bgr, clean_mask_np, original_hw, H, W, alpha=0.5):
    """
    Draw a semi-transparent green lane overlay on the original-resolution frame.
    Returns a copy of frame_bgr (original resolution) with the overlay.
    """
    # Upscale mask back to original resolution
    oh, ow = original_hw
    mask_up = cv2.resize(clean_mask_np, (ow, oh),
                         interpolation=cv2.INTER_NEAREST)
    out = frame_bgr.copy()
    green = np.zeros_like(out)
    green[:, :, 1] = 255                      # green channel
    out[mask_up == 1] = cv2.addWeighted(
        out, 1 - alpha, green, alpha, 0
    )[mask_up == 1]
    return out


def denorm_bgr(tensor_4hw):
    """Denormalise a [4,H,W] tensor back to uint8 BGR for display."""
    rgb = tensor_4hw[:3].numpy() * _STD + _MEAN
    rgb = np.clip(rgb * 255, 0, 255).astype(np.uint8).transpose(1, 2, 0)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


# KITTI-style colour map (matches original FusionLane output_pretrained / output_60epochs)
# background=red, lane=magenta — mirrors the reference output colour scheme
_KITTI_COLOURS = np.array([
    (0,   0,   220),    # 0 = background  → red   (BGR)
    (220, 0,   220),    # 1 = lane        → magenta (BGR)
], dtype=np.uint8)


def make_kitti_seg(clean_mask_np):
    """
    Convert binary clean mask [H, W] → KITTI-style coloured segmentation [H, W, 3] BGR.
    Background = red, lane = magenta — same palette as output_pretrained reference.
    """
    return _KITTI_COLOURS[clean_mask_np.astype(np.uint8)]


def overlay_lane_kitti(frame_bgr, clean_mask_np, original_hw, H, W, alpha=0.55):
    """
    Blend original frame with KITTI-coloured segmentation.
    Lane pixels → magenta tint; background → slight red tint.
    Only the lane pixels are visibly coloured (alpha-blended onto original).
    """
    oh, ow = original_hw
    mask_up = cv2.resize(clean_mask_np.astype(np.uint8), (ow, oh),
                         interpolation=cv2.INTER_NEAREST)
    out = frame_bgr.copy()
    magenta = np.zeros_like(out)
    magenta[:, :, 0] = 220   # B
    magenta[:, :, 2] = 220   # R  → magenta in BGR
    out[mask_up == 1] = cv2.addWeighted(
        out, 1 - alpha, magenta, alpha, 0
    )[mask_up == 1]
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="FusionLane media inference")
    p.add_argument("--input",        type=str,   default=CONFIG["input"],
                   help="Video file, image folder, or single image")
    p.add_argument("--model_path",   type=str,   default=CONFIG["model_path"])
    p.add_argument("--output_dir",   type=str,   default=CONFIG["output_dir"])
    p.add_argument("--image_height", type=int,   default=CONFIG["image_height"])
    p.add_argument("--image_width",  type=int,   default=CONFIG["image_width"])
    p.add_argument("--threshold",    type=float, default=CONFIG["threshold"])
    p.add_argument("--min_blob",     type=int,   default=CONFIG["min_blob"])
    p.add_argument("--batch_size",   type=int,   default=CONFIG["batch_size"])
    p.add_argument("--gt_dir",       type=str,   default=None,
                   help="Optional folder of ground-truth binary masks named "
                        "gt_0000.png, gt_0001.png … (white=lane). "
                        "When provided, mIoU and pixel accuracy are computed.")
    p.add_argument("--use_depth",    action="store_true",
                   help="Estimate monocular depth (DPT) and use as channel 4 "
                        "(pseudo-LiDAR).  Requires `pip install transformers`. "
                        "Downloads ~100 MB on first run.")
    p.add_argument("--roi_top",           type=float, default=CONFIG["roi_top"],
                   help="Blank the top fraction of each frame before inference "
                        "to suppress sky/horizon false positives (default 0.40).")
    p.add_argument("--use_clahe",         action="store_true",
                   help="Apply CLAHE adaptive contrast enhancement before inference. "
                        "Reduces domain gap from different cameras/lighting.")
    p.add_argument("--temporal_alpha",    type=float, default=0.0,
                   help="EMA smoothing of confidence maps across frames "
                        "(0.0=off, 0.6=moderate, 0.8=heavy). "
                        "Eliminates frame-to-frame jitter without a temporal model.")
    p.add_argument("--adaptive_threshold",action="store_true",
                   help="Per-frame automatic threshold via Otsu's method. "
                        "Replaces the fixed --threshold. Self-adjusts to each frame's "
                        "confidence distribution — handles different cameras and lighting "
                        "without manual tuning.")
    p.add_argument("--fit_lanes",         action="store_true",
                   help="Fit polynomial curves to detected lane blobs. "
                        "Outputs smooth geometric lanes instead of irregular blobs. "
                        "Improves temporal consistency and visual quality.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    H, W   = args.image_height, args.image_width

    assert args.batch_size % 4 == 0, "batch_size must be divisible by 4"

    model = load_model(args.model_path, device)

    # Optional DPT pseudo-LiDAR depth estimator
    dpt_model = dpt_processor = None
    if args.use_depth:
        dpt_model, dpt_processor = load_depth_estimator()
        print("[depth] Channel 4 = DPT monocular depth estimate (pseudo-LiDAR)")
    else:
        print("[depth] Channel 4 = all-ones (camera-only mode). "
              "Pass --use_depth to enable pseudo-LiDAR.")

    print(f"[roi]    Top {args.roi_top*100:.0f}% of each frame blanked (road ROI masking)")
    if args.adaptive_threshold:
        print(f"[thresh] Adaptive (Otsu per-frame)  floor={0.10}  ceiling={0.70}")
    else:
        print(f"[thresh] Fixed = {args.threshold}  min_blob = {args.min_blob}")
    if args.use_clahe:
        print("[clahe]  CLAHE contrast enhancement  ON")
    if args.temporal_alpha > 0:
        print(f"[ema]    Temporal EMA smoothing  alpha={args.temporal_alpha}")
    if args.fit_lanes:
        print("[poly]   Polynomial lane fitting  ON")

    # Create output directories
    subdirs = ["raw", "cleaned", "heatmap", "comparison", "kitti_seg", "fitted"]
    for sub in subdirs:
        os.makedirs(os.path.join(args.output_dir, sub), exist_ok=True)

    # Detect if input is video → also produce overlay video
    ext         = os.path.splitext(args.input)[1].lower()
    is_video    = ext in _VID_EXTS and os.path.isfile(args.input)
    video_writer = None
    overlay_path = os.path.join(args.output_dir, "lane_overlay.mp4")

    # Collect all frames first to know total count
    all_frames   = list(read_frames(args.input))   # (bgr, hw) pairs
    n_total      = len(all_frames)
    print(f"Processing {n_total} frame(s)...")

    # Pad so total is divisible by batch_size (pad with last frame)
    while len(all_frames) % args.batch_size != 0:
        all_frames.append(all_frames[-1])
    n_padded = len(all_frames)

    saved        = 0
    frames_out   = []    # overlay frames for video writer
    metrics_rows = []    # per-frame metric records
    prev_clean   = None  # for temporal IoU
    prev_smooth_conf = None  # for EMA temporal smoothing

    metrics_path = os.path.join(args.output_dir, "metrics.csv")

    with torch.no_grad():
        for b_start in tqdm(range(0, n_padded, args.batch_size), desc="infer"):
            batch_frames = all_frames[b_start: b_start + args.batch_size]
            # Optional CLAHE contrast enhancement before model inference
            if args.use_clahe:
                proc_frames = [(apply_clahe(f), hw) for f, hw in batch_frames]
            else:
                proc_frames = batch_frames

            if dpt_model is not None:
                depth_maps = [estimate_depth(f, dpt_model, dpt_processor, H, W)
                              for f, _ in proc_frames]
            else:
                depth_maps = [None] * len(proc_frames)
            tensors = torch.stack([
                preprocess(f, H, W, dm)
                for (f, _), dm in zip(proc_frames, depth_maps)
            ])
            logits   = model(tensors.to(device))
            raw_pred = logits.argmax(dim=1).cpu().numpy().astype(np.uint8)
            filt, conf_t = confidence_filter(logits.cpu(), args.threshold)

            for i, (frame_bgr, orig_hw) in enumerate(batch_frames):
                idx = b_start + i
                if idx >= n_total:   # skip padded frames
                    break

                raw_mask = raw_pred[i]
                conf_np  = conf_t[i].numpy()

                # ── Temporal EMA smoothing on continuous confidence map ────────
                # Smooths jitter before thresholding — novel vs fixed-threshold
                # pipelines.  Works across batch boundaries via prev_smooth_conf.
                conf_np_smooth = smooth_confidence(conf_np, prev_smooth_conf,
                                                   args.temporal_alpha)
                prev_smooth_conf = conf_np_smooth

                # ── Per-frame adaptive threshold (Otsu) ───────────────────────
                if args.adaptive_threshold:
                    frame_thresh = adaptive_otsu_threshold(conf_np_smooth)
                else:
                    frame_thresh = args.threshold

                filt_mask = (conf_np_smooth >= frame_thresh).astype(np.uint8)
                clean     = clean_mask(filt_mask, args.min_blob)

                # ── ROI: suppress sky/horizon false positives ─────────────────
                clean    = apply_roi_mask(clean,    args.roi_top)
                raw_mask = apply_roi_mask(raw_mask, args.roi_top)

                # ── Polynomial lane fitting (separate from blob mask) ─────────
                # Fitted curves go to kitti_seg/ and fitted/ for visualization.
                # The blob clean mask is kept for metrics and the overlay video
                # so pixel-level coverage numbers remain comparable.
                if args.fit_lanes and clean.any():
                    fitted_mask = fit_lane_polynomials(clean)
                else:
                    fitted_mask = clean

                # Save outputs
                stem = f"pred_{saved:04d}"
                cv2.imwrite(os.path.join(args.output_dir, "raw",     f"{stem}.png"),
                            raw_mask * 255)
                cv2.imwrite(os.path.join(args.output_dir, "cleaned", f"{stem}.png"),
                            clean    * 255)
                cv2.imwrite(os.path.join(args.output_dir, "heatmap", f"{stem}.png"),
                            make_heatmap(conf_np))

                orig_bgr  = denorm_bgr(tensors[i].cpu())
                raw_bgr   = cv2.cvtColor(raw_mask * 255, cv2.COLOR_GRAY2BGR)
                clean_bgr = cv2.cvtColor(clean    * 255, cv2.COLOR_GRAY2BGR)
                panel     = np.concatenate(
                    [orig_bgr, raw_bgr, clean_bgr, make_heatmap(conf_np)], axis=1
                )

                # ── per-frame metrics ─────────────────────────────────────────
                total_pixels  = clean.size
                lane_pixels   = int(clean.sum())
                raw_pixels    = int(raw_mask.sum())
                lane_pct      = lane_pixels / total_pixels * 100.0
                raw_pct       = raw_pixels  / total_pixels * 100.0
                noise_removed = max(0.0, raw_pct - lane_pct)

                # Confidence stats over all pixels
                conf_flat = conf_np.ravel()
                conf_sorted = np.sort(conf_flat)
                n_pix = len(conf_sorted)
                mean_conf = (float(conf_np[clean == 1].mean())
                             if lane_pixels > 0 else 0.0)
                conf_p50  = float(conf_sorted[int(n_pix * 0.50)])
                conf_p75  = float(conf_sorted[int(n_pix * 0.75)])
                conf_p95  = float(conf_sorted[int(n_pix * 0.95)])

                # Temporal IoU vs previous frame's cleaned mask
                if prev_clean is not None:
                    inter = int((clean & prev_clean).sum())
                    union = int((clean | prev_clean).sum())
                    temp_iou = inter / union if union > 0 else 1.0
                else:
                    temp_iou = 1.0   # first frame — perfect by convention
                prev_clean = clean.copy()

                # Lane horizontal centre (0=left edge, 1=right edge)
                if lane_pixels > 0:
                    cols = np.where(clean.any(axis=0))[0]
                    lane_center_x = float(cols.mean()) / (W - 1)
                else:
                    lane_center_x = 0.5   # default to centre when nothing detected

                # GT accuracy (optional)
                miou_val = None
                pix_acc  = None
                if args.gt_dir:
                    gt_path = os.path.join(args.gt_dir, f"gt_{saved:04d}.png")
                    if os.path.exists(gt_path):
                        gt_img = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
                        if gt_img is not None:
                            gt_r     = cv2.resize(gt_img, (W, H),
                                                  interpolation=cv2.INTER_NEAREST)
                            gt_label = torch.from_numpy(
                                (gt_r > 127).astype(np.int64)
                            ).unsqueeze(0)
                            miou_val = compute_miou(logits[i:i+1].cpu(), gt_label)
                            pred_bin = logits[i].cpu().argmax(0).numpy()
                            gt_bin   = (gt_r > 127).astype(np.uint8)
                            pix_acc  = float((pred_bin == gt_bin).mean())

                metrics_rows.append({
                    "frame":          saved,
                    "lane_pct":       f"{lane_pct:.2f}",
                    "raw_pct":        f"{raw_pct:.2f}",
                    "noise_removed":  f"{noise_removed:.2f}",
                    "mean_conf":      f"{mean_conf:.4f}",
                    "conf_p50":       f"{conf_p50:.4f}",
                    "conf_p75":       f"{conf_p75:.4f}",
                    "conf_p95":       f"{conf_p95:.4f}",
                    "threshold_used": f"{frame_thresh:.4f}",
                    "temporal_iou":   f"{temp_iou:.4f}",
                    "lane_center_x":  f"{lane_center_x:.4f}",
                    "miou":      f"{miou_val:.4f}" if miou_val is not None else "N/A",
                    "pixel_acc": f"{pix_acc:.4f}"  if pix_acc  is not None else "N/A",
                })

                # Overlay metric text on comparison panel
                line1 = (f"frame:{saved:04d}  cov:{lane_pct:.1f}%"
                         f"  conf:{mean_conf:.3f}  t-iou:{temp_iou:.3f}")
                line2 = (f"raw:{raw_pct:.1f}%  noise_rm:{noise_removed:.1f}%"
                         f"  cx:{lane_center_x:.2f}"
                         + (f"  mIoU:{miou_val:.3f}" if miou_val is not None else ""))
                ph = panel.shape[0]
                cv2.putText(panel, line1, (8, ph - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255,255,255), 1, cv2.LINE_AA)
                cv2.putText(panel, line2, (8, ph - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200,200,200), 1, cv2.LINE_AA)

                # KITTI-style and fitted polynomial outputs (use fitted_mask)
                kitti_seg_img = make_kitti_seg(fitted_mask)
                cv2.imwrite(os.path.join(args.output_dir, "kitti_seg", f"{stem}.png"),
                            kitti_seg_img)
                cv2.imwrite(os.path.join(args.output_dir, "fitted",    f"{stem}.png"),
                            fitted_mask * 255)

                cv2.imwrite(os.path.join(args.output_dir, "comparison", f"{stem}.png"),
                            panel)

                # Build overlay frame (magenta KITTI style on original resolution)
                if is_video:
                    ov = overlay_lane_kitti(frame_bgr, clean, orig_hw, H, W)
                    frames_out.append(ov)

                saved += 1

    # Write overlay video if input was a video
    if is_video and frames_out:
        oh, ow = frames_out[0].shape[:2]
        cap    = cv2.VideoCapture(args.input)
        fps    = cap.get(cv2.CAP_PROP_FPS) or 30
        cap.release()
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        vw     = cv2.VideoWriter(overlay_path, fourcc, fps, (ow, oh))
        for fr in frames_out:
            vw.write(fr)
        vw.release()
        print(f"Lane overlay video saved: {overlay_path}")

    # Write per-frame metrics CSV
    with open(metrics_path, "w", newline="") as mf:
        writer = csv.DictWriter(mf, fieldnames=[
            "frame", "lane_pct", "raw_pct", "noise_removed",
            "mean_conf", "conf_p50", "conf_p75", "conf_p95",
            "threshold_used", "temporal_iou", "lane_center_x",
            "miou", "pixel_acc",
        ])
        writer.writeheader()
        writer.writerows(metrics_rows)

    print(f"\nDone. {saved} frame(s) saved to {args.output_dir}/")
    print("  raw/         - argmax predictions")
    print("  cleaned/     - confidence-filtered + morphologically cleaned")
    print("  heatmap/     - P(lane) confidence maps")
    print("  comparison/  - 4-panel side-by-side with metric overlay")
    if is_video:
        print(f"  lane_overlay.mp4 - original video with green lane overlay")
    print(f"  kitti_seg/   - KITTI-style coloured segmentation (red=bg, magenta=lane)")
    if args.fit_lanes:
        print(f"  fitted/      - Polynomial curve fits (thin smooth lane lines)")
    print(f"  metrics.csv  - per-frame lane coverage, confidence, mIoU, pixel accuracy")

    # Summary stats
    if metrics_rows:
        def _f(key): return [float(r[key]) for r in metrics_rows]
        avg_lane   = np.mean(_f("lane_pct"))
        avg_raw    = np.mean(_f("raw_pct"))
        avg_noise  = np.mean(_f("noise_removed"))
        avg_conf   = np.mean(_f("mean_conf"))
        avg_tiou   = np.mean(_f("temporal_iou"))
        std_lane   = np.std(_f("lane_pct"))
        print(f"\nMetrics summary ({saved} frames):")
        print(f"  Avg lane coverage (clean) : {avg_lane:.2f}%  (std {std_lane:.2f})")
        print(f"  Avg raw coverage          : {avg_raw:.2f}%")
        print(f"  Avg noise removed         : {avg_noise:.2f}%")
        print(f"  Avg lane confidence       : {avg_conf:.4f}")
        print(f"  Avg temporal IoU          : {avg_tiou:.4f}  (1.0=perfectly stable)")
        if metrics_rows[0]["miou"] != "N/A":
            avg_miou = np.mean(_f("miou"))
            avg_acc  = np.mean(_f("pixel_acc"))
            print(f"  Avg mIoU (vs GT)          : {avg_miou:.4f}")
            print(f"  Avg pixel accuracy (vs GT): {avg_acc:.4f}")


if __name__ == "__main__":
    main()
