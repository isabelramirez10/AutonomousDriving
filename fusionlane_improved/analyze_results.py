"""
FusionLane — cross-run analysis script.

Reads metrics.csv from infer_media.py (DL model) and infer_baseline.py
(Hough) for any number of runs, then produces:

  outputs/analysis/
    report.txt          human-readable analysis report
    combined_dl.csv     all DL frame rows tagged with video label
    combined_bl.csv     all baseline frame rows tagged with video label

Usage
-----
  python analyze_results.py \\
      --dl      label1:path/to/inference1  label2:path/to/inference2 \\
      --baseline label1:path/to/baseline1  label2:path/to/baseline2  \\
      --output_dir ./outputs/analysis
"""

import argparse
import csv
import os
import statistics
import textwrap

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# CSV helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _floats(rows, key):
    vals = []
    for r in rows:
        v = r.get(key, "N/A")
        if v != "N/A":
            try:
                vals.append(float(v))
            except ValueError:
                pass
    return vals


def _pct(vals, p):
    if not vals:
        return float("nan")
    s = sorted(vals)
    idx = int(len(s) * p / 100)
    idx = min(idx, len(s) - 1)
    return s[idx]


def _longest_streak(arr, target=1):
    best = cur = 0
    for v in arr:
        if v == target:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


# ─────────────────────────────────────────────────────────────────────────────
# DL model analysis
# ─────────────────────────────────────────────────────────────────────────────

def analyze_dl(rows, label):
    frames = len(rows)
    lane   = _floats(rows, "lane_pct")
    raw    = _floats(rows, "raw_pct")
    noise  = _floats(rows, "noise_removed")
    conf   = _floats(rows, "mean_conf")
    tiou   = _floats(rows, "temporal_iou")
    cx     = _floats(rows, "lane_center_x")
    p95    = _floats(rows, "conf_p95")
    miou   = _floats(rows, "miou")
    acc    = _floats(rows, "pixel_acc")

    # Temporal stability: frames where tiou < 0.5 (large mask shift)
    unstable = sum(1 for v in tiou if v < 0.5)

    # Lane presence: frames where any lane detected (clean coverage > 1%)
    present = sum(1 for v in lane if v > 1.0)

    def _stat(vals):
        if not vals:
            return "N/A"
        return (f"mean={statistics.mean(vals):.3f}  "
                f"std={statistics.stdev(vals) if len(vals)>1 else 0:.3f}  "
                f"min={min(vals):.3f}  "
                f"p50={_pct(vals,50):.3f}  "
                f"p95={_pct(vals,95):.3f}  "
                f"max={max(vals):.3f}")

    return {
        "label":        label,
        "frames":       frames,
        "lane_pct":     _stat(lane),
        "raw_pct":      _stat(raw),
        "noise_removed":_stat(noise),
        "mean_conf":    _stat(conf),
        "temporal_iou": _stat(tiou),
        "lane_center_x":_stat(cx),
        "conf_p95":     _stat(p95),
        "miou":         _stat(miou)  if miou else "N/A (no GT)",
        "pixel_acc":    _stat(acc)   if acc  else "N/A (no GT)",
        # scalars for cross-run table
        "_lane_mean":   statistics.mean(lane) if lane else float("nan"),
        "_lane_std":    statistics.stdev(lane) if len(lane)>1 else 0.0,
        "_conf_mean":   statistics.mean(conf) if conf else float("nan"),
        "_tiou_mean":   statistics.mean(tiou) if tiou else float("nan"),
        "_noise_mean":  statistics.mean(noise) if noise else float("nan"),
        "_present_pct": present / frames * 100 if frames else 0,
        "_unstable_pct":unstable / frames * 100 if frames else 0,
        "_miou_mean":   statistics.mean(miou) if miou else float("nan"),
        "_acc_mean":    statistics.mean(acc)  if acc  else float("nan"),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Baseline analysis
# ─────────────────────────────────────────────────────────────────────────────

def analyze_baseline(rows, label):
    frames = len(rows)
    hough  = _floats(rows, "hough_lines")
    left   = [int(float(r["left_detected"]))  for r in rows]
    right  = [int(float(r["right_detected"])) for r in rows]
    both   = [1 if l and r else 0 for l, r in zip(left, right)]

    left_changes  = sum(1 for i in range(1, len(left))  if left[i]  != left[i-1])
    right_changes = sum(1 for i in range(1, len(right)) if right[i] != right[i-1])

    left_rate   = statistics.mean(left)  * 100 if left  else 0
    right_rate  = statistics.mean(right) * 100 if right else 0
    both_rate   = statistics.mean(both)  * 100 if both  else 0
    hough_mean  = statistics.mean(hough) if hough else 0
    hough_std   = statistics.stdev(hough) if len(hough)>1 else 0

    # Stability: 1 - (changes / possible_changes)
    possible     = max(frames - 1, 1)
    left_stab    = (1 - left_changes  / possible) * 100
    right_stab   = (1 - right_changes / possible) * 100

    return {
        "label":              label,
        "frames":             frames,
        "left_det_rate":      f"{left_rate:.1f}%",
        "right_det_rate":     f"{right_rate:.1f}%",
        "both_det_rate":      f"{both_rate:.1f}%",
        "hough_segments":     f"mean={hough_mean:.1f}  std={hough_std:.1f}  "
                              f"min={int(min(hough)) if hough else 0}  "
                              f"max={int(max(hough)) if hough else 0}",
        "left_stability":     f"{left_stab:.1f}%  ({left_changes} detection flips)",
        "right_stability":    f"{right_stab:.1f}%  ({right_changes} detection flips)",
        "left_longest_streak":_longest_streak(left),
        "right_longest_streak":_longest_streak(right),
        # scalars
        "_left_rate":   left_rate,
        "_right_rate":  right_rate,
        "_both_rate":   both_rate,
        "_left_stab":   left_stab,
        "_right_stab":  right_stab,
        "_hough_mean":  hough_mean,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Report writer
# ─────────────────────────────────────────────────────────────────────────────

def _bar(value, total=100, width=30, char="#"):
    filled = int(round(value / total * width))
    filled = max(0, min(filled, width))
    return char * filled + "." * (width - filled)


def write_report(dl_stats, bl_stats, out_path):
    sep  = "─" * 72
    sep2 = "═" * 72

    lines = []
    def p(*args): lines.append(" ".join(str(a) for a in args))

    p(sep2)
    p("  FUSIONLANE — MODEL VERSATILITY ANALYSIS REPORT")
    p(sep2)
    p()

    # ── Section 1: DL model per-video ─────────────────────────────────────
    p("1. DEEP-LEARNING MODEL (ResNet-18 U-Net + FusionLane post-processing)")
    p(sep)
    p()
    avg_cov = sum(s["_lane_mean"] for s in dl_stats) / max(len(dl_stats), 1)
    if avg_cov > 50:
        p("  NOTE: Avg lane coverage >50% -- model may not be trained on real labels yet.")
        p("  Metrics reflect untrained inference. Train on KITTI/TuSimple/CULane first.")
    else:
        p("  Model: ResNet-18 U-Net trained on KITTI Road (camera + LiDAR fusion).")
        p("  Training log: outputs/logs/training_log.csv")
    p()

    for s in dl_stats:
        p(f"  Video : {s['label']}  ({s['frames']} frames)")
        p(f"  {'Metric':<22}  {'Distribution'}")
        p(f"  {'-'*22}  {'-'*46}")
        for key in ["lane_pct", "raw_pct", "noise_removed",
                    "mean_conf", "conf_p95", "temporal_iou", "lane_center_x"]:
            p(f"  {key:<22}  {s[key]}")
        if s["miou"] != "N/A (no GT)":
            p(f"  {'miou (vs GT)':<22}  {s['miou']}")
            p(f"  {'pixel_acc (vs GT)':<22}  {s['pixel_acc']}")
        else:
            p(f"  mIoU / pixel_acc : N/A — no ground-truth masks provided.")
            p(f"  → Supply --gt_dir to infer_media.py to compute accuracy.")
        p()
        p(f"  Lane presence   : {s['_present_pct']:.1f}% of frames have >1% coverage")
        p(f"  Temporal stable : {100 - s['_unstable_pct']:.1f}% frames within IoU≥0.5 of prev")
        p()

    # Cross-video DL comparison
    if len(dl_stats) > 1:
        p(f"  {'Cross-video DL comparison':}")
        p(f"  {'Video':<20}  {'Avg lane%':>10}  {'Std':>6}  {'Avg conf':>9}  "
          f"{'Avg t-IoU':>10}  {'Noise rm%':>10}")
        p(f"  {'-'*20}  {'-'*10}  {'-'*6}  {'-'*9}  {'-'*10}  {'-'*10}")
        for s in dl_stats:
            bar = _bar(min(s['_lane_mean'], 100))
            p(f"  {s['label']:<20}  {s['_lane_mean']:>9.2f}%  "
              f"{s['_lane_std']:>6.2f}  "
              f"{s['_conf_mean']:>9.4f}  "
              f"{s['_tiou_mean']:>10.4f}  "
              f"{s['_noise_mean']:>9.2f}%")
        p()

    # ── Section 2: Baseline per-video ──────────────────────────────────────
    p("2. CLASSICAL BASELINE  (Canny edges + Probabilistic Hough Transform)")
    p(sep)
    p()

    for s in bl_stats:
        p(f"  Video : {s['label']}  ({s['frames']} frames)")
        p(f"  {'Metric':<30}  {'Value'}")
        p(f"  {'-'*30}  {'-'*30}")
        p(f"  {'Left  lane detection rate':<30}  {s['left_det_rate']}"
          f"  {_bar(s['_left_rate'])}")
        p(f"  {'Right lane detection rate':<30}  {s['right_det_rate']}"
          f"  {_bar(s['_right_rate'])}")
        p(f"  {'Both  lanes detected':<30}  {s['both_det_rate']}"
          f"  {_bar(s['_both_rate'])}")
        p(f"  {'Left  temporal stability':<30}  {s['left_stability']}")
        p(f"  {'Right temporal stability':<30}  {s['right_stability']}")
        p(f"  {'Hough segments / frame':<30}  {s['hough_segments']}")
        p(f"  {'Longest left  streak (frames)':<30}  {s['left_longest_streak']}")
        p(f"  {'Longest right streak (frames)':<30}  {s['right_longest_streak']}")
        p()

    # Cross-video baseline comparison
    if len(bl_stats) > 1:
        p(f"  {'Cross-video Baseline comparison':}")
        p(f"  {'Video':<20}  {'Left%':>6}  {'Right%':>7}  {'Both%':>6}  "
          f"{'L-stab%':>8}  {'R-stab%':>8}  {'Hough/f':>8}")
        p(f"  {'-'*20}  {'-'*6}  {'-'*7}  {'-'*6}  {'-'*8}  {'-'*8}  {'-'*8}")
        for s in bl_stats:
            p(f"  {s['label']:<20}  {s['_left_rate']:>5.1f}%  "
              f"{s['_right_rate']:>6.1f}%  "
              f"{s['_both_rate']:>5.1f}%  "
              f"{s['_left_stab']:>7.1f}%  "
              f"{s['_right_stab']:>7.1f}%  "
              f"{s['_hough_mean']:>8.1f}")
        p()

    # ── Section 3: Model versatility assessment ────────────────────────────
    p("3. MODEL VERSATILITY ASSESSMENT")
    p(sep)
    p()

    # DL temporal consistency interpretation
    if dl_stats:
        tiou_vals = [s["_tiou_mean"] for s in dl_stats]
        avg_tiou  = sum(tiou_vals) / len(tiou_vals)
        if avg_tiou >= 0.85:
            tiou_verdict = "EXCELLENT — predictions are highly stable frame-to-frame"
        elif avg_tiou >= 0.70:
            tiou_verdict = "GOOD — minor flickering between frames"
        elif avg_tiou >= 0.50:
            tiou_verdict = "MODERATE — visible jitter; post-processing helps"
        else:
            tiou_verdict = "POOR — heavy flickering; model needs training"
        p(f"  DL Temporal Consistency  ({avg_tiou:.3f} avg t-IoU):")
        p(f"    {tiou_verdict}")
        p()

    # Baseline versatility across videos
    if len(bl_stats) > 1:
        rates = [(s["_left_rate"] + s["_right_rate"]) / 2 for s in bl_stats]
        spread = max(rates) - min(rates)
        if spread < 10:
            bl_verdict = "CONSISTENT across scenes"
        elif spread < 30:
            bl_verdict = "VARIABLE — road/lighting differences affect detection"
        else:
            bl_verdict = "INCONSISTENT — Hough parameters need per-scene tuning"
        avg_rate = sum(rates) / len(rates)
        p(f"  Baseline Detection Consistency (avg={avg_rate:.1f}%, spread={spread:.1f}%):")
        p(f"    {bl_verdict}")
        p()

    p("  Confidence interpretation (trained model):")
    p("    - mean_conf is computed over detected lane pixels only (clean mask).")
    p("    - Values >0.85 = model is confident where it fires (good).")
    p("    - Low lane_pct + high mean_conf = selective, precise detection.")
    p("    - High lane_pct + low mean_conf = uncertain / over-predicted detections.")
    p()

    p("  Ground truth / annotation status:")
    has_gt = any(s["miou"] != "N/A (no GT)" for s in dl_stats)
    if has_gt:
        p("    GT masks supplied. mIoU and pixel accuracy are available.")
    else:
        p("    No GT masks were provided with --gt_dir.")
        p("    To get mIoU and pixel accuracy:")
        p("      Option A — use infer_pt.py on a labelled dataset (TuSimple/CULane)")
        p("      Option B — annotate video frames in CVAT, export masks,")
        p("                 then pass the mask folder as --gt_dir to infer_media.py")
        p("      Option C — run prepare_data.py → use val/masks/ as --gt_dir")
    p()

    p("  Recommendations:")
    p("    1. Download TuSimple or CULane data, or annotate in CVAT.")
    p("    2. Run prepare_data.py to build data/train/ and data/val/.")
    p("    3. Run train_pt.py --data_dir ./data --epochs 50 --lr 3e-4")
    p("    4. Re-run this analysis — expect mIoU 0.70-0.85 and lane_pct 5-20%.")
    p()
    p(sep2)

    report = "\n".join(lines)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(report)
    return report


# ─────────────────────────────────────────────────────────────────────────────
# Combined CSV writers
# ─────────────────────────────────────────────────────────────────────────────

def write_combined(label_path_pairs, out_path, kind):
    all_rows = []
    fieldnames = None
    for label, path in label_path_pairs:
        if not os.path.exists(path):
            continue
        rows = load_csv(path)
        if not rows:
            continue
        if fieldnames is None:
            fieldnames = ["video"] + list(rows[0].keys())
        for r in rows:
            all_rows.append({"video": label, **r})
    if not all_rows:
        return
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"  Combined {kind} CSV: {out_path}  ({len(all_rows)} rows)")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="FusionLane cross-run analysis")
    p.add_argument("--dl",       nargs="+", default=[],
                   metavar="LABEL:DIR",
                   help="DL inference output dirs, e.g.  video1:./outputs/inf1")
    p.add_argument("--baseline", nargs="+", default=[],
                   metavar="LABEL:DIR",
                   help="Baseline output dirs, e.g.      video1:./outputs/bl1")
    p.add_argument("--output_dir", type=str, default="./outputs/analysis")
    return p.parse_args()


def _parse_pairs(args_list):
    """'label:path' strings → list of (label, metrics_csv_path)."""
    result = []
    for item in args_list:
        if ":" in item:
            label, path = item.split(":", 1)
        else:
            label = os.path.basename(item.rstrip("/\\"))
            path  = item
        metrics = os.path.join(path, "metrics.csv")
        if not os.path.exists(metrics):
            print(f"  WARNING: metrics.csv not found at {metrics} — skipping")
            continue
        result.append((label, metrics))
    return result


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    dl_pairs = _parse_pairs(args.dl)
    bl_pairs = _parse_pairs(args.baseline)

    if not dl_pairs and not bl_pairs:
        print("No valid metrics.csv files found. Check --dl and --baseline paths.")
        return

    dl_stats = [analyze_dl(load_csv(p), lbl)       for lbl, p in dl_pairs]
    bl_stats = [analyze_baseline(load_csv(p), lbl)  for lbl, p in bl_pairs]

    report_path = os.path.join(args.output_dir, "report.txt")
    report = write_report(dl_stats, bl_stats, report_path)
    # Print safely on Windows terminals that use narrow code pages
    try:
        print(report)
    except UnicodeEncodeError:
        print(report.encode("ascii", errors="replace").decode("ascii"))
    print(f"\nReport saved: {report_path}")

    write_combined(dl_pairs, os.path.join(args.output_dir, "combined_dl.csv"),      "DL")
    write_combined(bl_pairs, os.path.join(args.output_dir, "combined_baseline.csv"),"Baseline")


if __name__ == "__main__":
    main()
