"""
FusionLane — Fusion vs Camera-Only Evaluation.

Evaluates the trained model on the KITTI validation split in two modes:
  1. Camera-only  : RGB input only; LiDAR channel = all-ones
  2. Fusion       : RGB + real Velodyne depth map as channel 4

For each sample it computes:
  - mIoU   (mean Intersection over Union across background + lane classes)
  - Dice   (F1 score over the lane class)
  - Recall (what fraction of GT lane pixels were detected)
  - Precision (what fraction of predicted lane pixels are correct)

Outputs
-------
  outputs/eval_fusion/
    camera_only/   comparison PNG per sample (image | cam-only pred | GT)
    fusion/        comparison PNG per sample (image | depth | fusion pred | GT)
    metrics.csv    per-sample scores for both modes
    summary.txt    overall statistics + fusion improvement

Usage
-----
  python eval_fusion.py
  python eval_fusion.py --data_dir ./data --output_dir ./outputs/eval_fusion
"""

import argparse
import csv
import os
import statistics

import cv2
import numpy as np
import torch

_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)


# ─────────────────────────────────────────────────────────────────────────────
# Preprocessing
# ─────────────────────────────────────────────────────────────────────────────

def preprocess(img_bgr, H, W, depth_u8=None):
    """Return [4, H, W] float32 tensor. depth_u8 is uint8 grayscale or None."""
    rgb = cv2.cvtColor(
        cv2.resize(img_bgr, (W, H), interpolation=cv2.INTER_LINEAR),
        cv2.COLOR_BGR2RGB,
    ).astype(np.float32).transpose(2, 0, 1) / 255.0
    rgb = (rgb - _MEAN) / _STD

    if depth_u8 is not None:
        d = cv2.resize(depth_u8, (W, H), interpolation=cv2.INTER_LINEAR)
        reg = (d.astype(np.float32) / 255.0)[np.newaxis]
    else:
        reg = np.ones((1, H, W), dtype=np.float32)

    return torch.from_numpy(np.concatenate([rgb, reg], axis=0))


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(pred_binary, gt_binary):
    """
    pred_binary, gt_binary: bool numpy arrays [H, W]
    Returns dict with miou, dice, recall, precision.
    """
    # Per-class IoU then average
    ious = []
    for cls in [False, True]:
        inter = ((pred_binary == cls) & (gt_binary == cls)).sum()
        union = ((pred_binary == cls) | (gt_binary == cls)).sum()
        ious.append(inter / union if union > 0 else 1.0)
    miou = float(np.mean(ious))

    # Lane-class specific
    tp = (pred_binary & gt_binary).sum()
    fp = (pred_binary & ~gt_binary).sum()
    fn = (~pred_binary & gt_binary).sum()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    dice      = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0

    return dict(miou=miou, dice=dice, recall=recall, precision=precision)


# ─────────────────────────────────────────────────────────────────────────────
# Visualization helpers
# ─────────────────────────────────────────────────────────────────────────────

def _add_label(img, text, color=(255, 255, 255)):
    out = img.copy()
    cv2.putText(out, text, (6, 18), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(out, text, (6, 18), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, color, 1, cv2.LINE_AA)
    return out


def make_comparison(img_bgr, depth_u8, pred_bin, gt_bin, metrics, mode, size=321):
    """Build a comparison panel for a single sample."""
    H   = size
    oh, ow = img_bgr.shape[:2]

    def _prep(arr_bgr):
        return cv2.resize(arr_bgr, (size, H), interpolation=cv2.INTER_LINEAR)

    img_s = _prep(img_bgr)

    # Depth channel visualisation (jet colormap, zeros masked to black)
    if depth_u8 is not None:
        depth_col = cv2.applyColorMap(depth_u8, cv2.COLORMAP_JET)
        depth_col[depth_u8 == 0] = 0
        depth_s = _prep(depth_col)
    else:
        depth_s = np.zeros((H, size, 3), dtype=np.uint8)
        cv2.putText(depth_s, "no depth", (10, H // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 80, 80), 1)

    # Upscale masks to original image resolution for overlay
    pred_up = cv2.resize(pred_bin.astype(np.uint8), (ow, oh),
                         interpolation=cv2.INTER_NEAREST).astype(bool)
    gt_up   = cv2.resize(gt_bin.astype(np.uint8), (ow, oh),
                         interpolation=cv2.INTER_NEAREST).astype(bool) if gt_bin.shape != img_bgr.shape[:2] else gt_bin

    # Prediction — green on original
    pred_ov = img_bgr.copy()
    green   = np.zeros_like(pred_ov); green[:, :, 1] = 255
    pred_ov[pred_up] = cv2.addWeighted(pred_ov, 0.45, green, 0.55, 0)[pred_up]
    pred_s = _prep(pred_ov)

    # GT — cyan on original
    gt_ov  = img_bgr.copy()
    cyan   = np.zeros_like(gt_ov); cyan[:, :, 0] = 200; cyan[:, :, 1] = 200
    gt_ov[gt_up]  = cv2.addWeighted(gt_ov,  0.45, cyan, 0.55, 0)[gt_up]
    gt_s = _prep(gt_ov)

    score_str = (f"mIoU={metrics['miou']:.3f}  "
                 f"Dice={metrics['dice']:.3f}  "
                 f"Rec={metrics['recall']:.3f}  "
                 f"Prec={metrics['precision']:.3f}")

    img_s   = _add_label(img_s,   "Camera image")
    depth_s = _add_label(depth_s, "LiDAR depth (channel 4)")
    pred_s  = _add_label(pred_s,  f"{mode} prediction")
    gt_s    = _add_label(gt_s,    "Ground truth")

    panel = np.hstack([img_s, depth_s, pred_s, gt_s])

    # Score bar
    bar = np.zeros((22, panel.shape[1], 3), dtype=np.uint8)
    cv2.putText(bar, score_str, (6, 15), cv2.FONT_HERSHEY_SIMPLEX,
                0.42, (220, 220, 220), 1, cv2.LINE_AA)
    return np.vstack([panel, bar])


# ─────────────────────────────────────────────────────────────────────────────
# Main evaluation
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="FusionLane fusion vs camera-only evaluation")
    p.add_argument("--data_dir",    type=str, default="./data",
                   help="Root of the prepared dataset (must have val/images, val/masks, val/depths).")
    p.add_argument("--model_path",  type=str, default="./outputs/best_model.pth")
    p.add_argument("--output_dir",  type=str, default="./outputs/eval_fusion")
    p.add_argument("--image_height",type=int, default=256)
    p.add_argument("--image_width", type=int, default=256)
    p.add_argument("--threshold",   type=float, default=0.50)
    return p.parse_args()


def main():
    args = parse_args()
    H, W = args.image_height, args.image_width

    # ── Directories ──────────────────────────────────────────────────────────
    img_dir   = os.path.join(args.data_dir, "val", "images")
    mask_dir  = os.path.join(args.data_dir, "val", "masks")
    depth_dir = os.path.join(args.data_dir, "val", "depths")

    for d in [img_dir, mask_dir]:
        if not os.path.isdir(d):
            raise FileNotFoundError(f"Directory not found: {d}")

    has_depth = os.path.isdir(depth_dir)
    print(f"Camera images : {img_dir}")
    print(f"GT masks      : {mask_dir}")
    print(f"LiDAR depths  : {depth_dir}  {'[FOUND]' if has_depth else '[NOT FOUND — fusion mode unavailable]'}")

    os.makedirs(os.path.join(args.output_dir, "camera_only"), exist_ok=True)
    if has_depth:
        os.makedirs(os.path.join(args.output_dir, "fusion"), exist_ok=True)

    # ── Model ────────────────────────────────────────────────────────────────
    from train_pt import build_model, compute_miou
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device        : {device}")

    ckpt  = torch.load(args.model_path, map_location=device, weights_only=False)
    model = build_model(num_classes=2).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Model loaded  : {args.model_path}  (best epoch {ckpt.get('epoch','?')}, mIoU {ckpt.get('best_miou', '?')})")

    # ── Sample list ──────────────────────────────────────────────────────────
    img_ext  = {'.jpg', '.jpeg', '.png', '.bmp'}
    samples  = sorted(
        f for f in os.listdir(img_dir)
        if os.path.splitext(f)[1].lower() in img_ext
    )
    print(f"Evaluating    : {len(samples)} validation samples\n")

    rows = []

    with torch.no_grad():
        for stem_ext in samples:
            stem = os.path.splitext(stem_ext)[0]

            img_bgr = cv2.imread(os.path.join(img_dir, stem_ext))
            if img_bgr is None:
                continue

            # GT mask
            gt_path = None
            for ext in ('.png', '.jpg', '.bmp'):
                p = os.path.join(mask_dir, stem + ext)
                if os.path.exists(p):
                    gt_path = p; break
            if gt_path is None:
                print(f"  [warn] No GT mask for {stem}, skipping.")
                continue
            gt_raw = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
            gt_bin = gt_raw > 127                        # bool [H0, W0]

            # LiDAR depth
            depth_u8 = None
            if has_depth:
                for ext in ('.png', '.jpg'):
                    p = os.path.join(depth_dir, stem + ext)
                    if os.path.exists(p):
                        depth_u8 = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
                        break

            def _run(use_depth):
                t = preprocess(img_bgr, H, W, depth_u8 if use_depth else None)
                logits = model(t.unsqueeze(0).to(device))
                prob   = torch.softmax(logits, dim=1)[0, 1].cpu().numpy()
                pred   = prob >= args.threshold
                # Upscale to GT resolution for metric computation
                pred_up = cv2.resize(pred.astype(np.uint8), (gt_bin.shape[1], gt_bin.shape[0]),
                                     interpolation=cv2.INTER_NEAREST).astype(bool)
                return _run_metrics(pred_up, gt_bin), pred, prob

            def _run_metrics(pred_up, gt_bin):
                return compute_metrics(pred_up, gt_bin)

            # Camera-only run
            cam_metrics, cam_pred, cam_prob = _run(use_depth=False)

            # Fusion run (if depths available)
            fus_metrics, fus_pred, fus_prob = (None, None, None)
            if has_depth and depth_u8 is not None:
                fus_metrics, fus_pred, fus_prob = _run(use_depth=True)

            # Save comparison images
            cam_panel = make_comparison(
                img_bgr, None, cam_pred, gt_bin, cam_metrics, "Camera-only")
            cv2.imwrite(os.path.join(args.output_dir, "camera_only", f"{stem}.png"), cam_panel)

            if fus_metrics is not None:
                fus_panel = make_comparison(
                    img_bgr, depth_u8, fus_pred, gt_bin, fus_metrics, "Fusion")
                cv2.imwrite(os.path.join(args.output_dir, "fusion", f"{stem}.png"), fus_panel)

            # Record
            row = {
                "sample":         stem,
                "cam_miou":       f"{cam_metrics['miou']:.4f}",
                "cam_dice":       f"{cam_metrics['dice']:.4f}",
                "cam_recall":     f"{cam_metrics['recall']:.4f}",
                "cam_precision":  f"{cam_metrics['precision']:.4f}",
                "fus_miou":       f"{fus_metrics['miou']:.4f}"      if fus_metrics else "N/A",
                "fus_dice":       f"{fus_metrics['dice']:.4f}"      if fus_metrics else "N/A",
                "fus_recall":     f"{fus_metrics['recall']:.4f}"    if fus_metrics else "N/A",
                "fus_precision":  f"{fus_metrics['precision']:.4f}" if fus_metrics else "N/A",
                "miou_delta":     f"{fus_metrics['miou']-cam_metrics['miou']:+.4f}" if fus_metrics else "N/A",
            }
            rows.append(row)

            tag = f"{stem[:20]:20s}"
            print(f"  {tag}  cam mIoU={cam_metrics['miou']:.4f}"
                  + (f"  fus mIoU={fus_metrics['miou']:.4f}"
                     f"  delta={fus_metrics['miou']-cam_metrics['miou']:+.4f}"
                     if fus_metrics else ""))

    # ── CSV ──────────────────────────────────────────────────────────────────
    csv_path = os.path.join(args.output_dir, "metrics.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # ── Summary ──────────────────────────────────────────────────────────────
    def _mean(key):
        vals = [float(r[key]) for r in rows if r[key] != "N/A"]
        return statistics.mean(vals) if vals else float("nan")

    cam_miou   = _mean("cam_miou")
    cam_dice   = _mean("cam_dice")
    cam_rec    = _mean("cam_recall")
    cam_prec   = _mean("cam_precision")
    fus_miou   = _mean("fus_miou")
    fus_dice   = _mean("fus_dice")
    fus_rec    = _mean("fus_recall")
    fus_prec   = _mean("fus_precision")
    has_fusion = fus_miou == fus_miou   # not NaN

    summary = []
    def p(s=""): summary.append(s); print(s)

    p("=" * 60)
    p("  FUSIONLANE — FULL EVALUATION RESULTS")
    p(f"  Dataset : KITTI Road val split  ({len(rows)} samples)")
    p(f"  Model   : {args.model_path}")
    p(f"  Threshold: {args.threshold}")
    p("=" * 60)
    p()
    p(f"  {'Metric':<18}  {'Camera-only':>13}  {'Fusion (Cam+LiDAR)':>19}  {'Delta':>8}")
    p("  " + "-" * 62)
    p(f"  {'mIoU':<18}  {cam_miou:>13.4f}  {fus_miou:>19.4f}  {fus_miou-cam_miou:>+8.4f}")
    p(f"  {'Dice (F1)':<18}  {cam_dice:>13.4f}  {fus_dice:>19.4f}  {fus_dice-cam_dice:>+8.4f}")
    p(f"  {'Recall':<18}  {cam_rec:>13.4f}  {fus_rec:>19.4f}  {fus_rec-cam_rec:>+8.4f}")
    p(f"  {'Precision':<18}  {cam_prec:>13.4f}  {fus_prec:>19.4f}  {fus_prec-cam_prec:>+8.4f}")
    p()

    # Samples where fusion wins / camera-only wins / tied
    if has_fusion:
        deltas = [float(r["miou_delta"]) for r in rows if r["miou_delta"] != "N/A"]
        wins   = sum(1 for d in deltas if d > 0.001)
        losses = sum(1 for d in deltas if d < -0.001)
        ties   = len(deltas) - wins - losses
        p(f"  Fusion better  on {wins}/{len(deltas)} samples ({wins/len(deltas)*100:.0f}%)")
        p(f"  Camera better  on {losses}/{len(deltas)} samples ({losses/len(deltas)*100:.0f}%)")
        p(f"  Effectively tied {ties}/{len(deltas)} samples ({ties/len(deltas)*100:.0f}%)")
        p()
        p(f"  Largest fusion gain : +{max(deltas):.4f} mIoU")
        p(f"  Largest fusion loss : {min(deltas):.4f} mIoU")
    p()
    p("  Interpretation for autonomous driving:")
    p("  LiDAR depth tells the model WHERE the road surface is.")
    p("  Camera RGB tells the model WHAT the lane markings look like.")
    p("  Together they enable lane detection even when one channel alone")
    p("  is ambiguous (e.g., faded markings on camera, specular LiDAR returns).")
    p()
    p(f"  Comparison images : {args.output_dir}/camera_only/  and  fusion/")
    p(f"  Per-sample CSV    : {csv_path}")
    p("=" * 60)

    sum_path = os.path.join(args.output_dir, "summary.txt")
    with open(sum_path, "w") as f:
        f.write("\n".join(summary))
    print(f"\n  Summary saved : {sum_path}")


if __name__ == "__main__":
    main()
