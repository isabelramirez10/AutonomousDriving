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
import os

import cv2
import numpy as np
import torch
from scipy import ndimage
from tqdm import tqdm

from train_pt import SimpleFusionLaneNet

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
    "image_height": 512,
    "image_width":  512,
    # ── Post-processing ───────────────────────────────────────────────────────
    "threshold":    0.50,   # P(lane) required to keep a pixel (lower = more lane shown)
    "min_blob":     100,    # smallest connected-component kept in pixels
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
    model = SimpleFusionLaneNet(in_channels=4, num_classes=2).to(device)
    ckpt  = torch.load(model_path, map_location=device, weights_only=False)
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


def preprocess(frame_bgr, H, W):
    """BGR uint8 frame → normalised float32 tensor [4, H, W]."""
    rgb = cv2.cvtColor(
        cv2.resize(frame_bgr, (W, H), interpolation=cv2.INTER_LINEAR),
        cv2.COLOR_BGR2RGB,
    ).astype(np.float32)                               # [H, W, 3]
    rgb = rgb.transpose(2, 0, 1) / 255.0              # [3, H, W]
    rgb = (rgb - _MEAN) / _STD                        # normalise
    reg = np.ones((1, H, W), dtype=np.float32)        # region mask = all road
    return torch.from_numpy(np.concatenate([rgb, reg], axis=0))  # [4, H, W]


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


def denorm_bgr(tensor_4hw, H, W):
    """Denormalise a [4,H,W] tensor back to uint8 BGR for display."""
    rgb = tensor_4hw[:3].numpy() * _STD + _MEAN
    rgb = np.clip(rgb * 255, 0, 255).astype(np.uint8).transpose(1, 2, 0)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


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

    # Create output directories
    subdirs = ["raw", "cleaned", "heatmap", "comparison"]
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

    saved  = 0
    frames_out = []   # store overlay frames for video writer

    with torch.no_grad():
        for b_start in tqdm(range(0, n_padded, args.batch_size), desc="infer"):
            batch_frames = all_frames[b_start: b_start + args.batch_size]
            tensors  = torch.stack([preprocess(f, H, W) for f, _ in batch_frames])
            logits   = model(tensors.to(device))
            raw_pred = logits.argmax(dim=1).cpu().numpy().astype(np.uint8)
            filt, conf_t = confidence_filter(logits.cpu(), args.threshold)

            for i, (frame_bgr, orig_hw) in enumerate(batch_frames):
                idx = b_start + i
                if idx >= n_total:   # skip padded frames
                    break

                raw_mask     = raw_pred[i]
                filt_mask    = filt[i].numpy()
                clean        = clean_mask(filt_mask, args.min_blob)
                conf_np      = conf_t[i].numpy()

                # Save outputs
                stem = f"pred_{saved:04d}"
                cv2.imwrite(os.path.join(args.output_dir, "raw",        f"{stem}.png"),
                            raw_mask * 255)
                cv2.imwrite(os.path.join(args.output_dir, "cleaned",    f"{stem}.png"),
                            clean    * 255)
                cv2.imwrite(os.path.join(args.output_dir, "heatmap",    f"{stem}.png"),
                            make_heatmap(conf_np))

                orig_bgr  = denorm_bgr(tensors[i].cpu(), H, W)
                raw_bgr   = cv2.cvtColor(raw_mask * 255, cv2.COLOR_GRAY2BGR)
                clean_bgr = cv2.cvtColor(clean    * 255, cv2.COLOR_GRAY2BGR)
                panel     = np.concatenate(
                    [orig_bgr, raw_bgr, clean_bgr, make_heatmap(conf_np)], axis=1
                )
                cv2.imwrite(os.path.join(args.output_dir, "comparison", f"{stem}.png"),
                            panel)

                # Build overlay frame (original resolution)
                if is_video:
                    ov = overlay_lane(frame_bgr, clean, orig_hw, H, W)
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

    print(f"\nDone. {saved} frame(s) saved to {args.output_dir}/")
    print("  raw/         - argmax predictions")
    print("  cleaned/     - confidence-filtered + morphologically cleaned")
    print("  heatmap/     - P(lane) confidence maps")
    print("  comparison/  - 4-panel side-by-side")
    if is_video:
        print(f"  lane_overlay.mp4 - original video with green lane overlay")


if __name__ == "__main__":
    main()
