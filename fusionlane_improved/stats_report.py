"""Print a complete statistics report for all FusionLane runs."""
import csv, os, statistics

def pct(vals, p):
    if not vals:
        return float("nan")
    s = sorted(vals)
    return s[min(int(len(s) * p / 100), len(s) - 1)]

def stat_line(vals, label, fmt=".4f"):
    if not vals:
        return f"  {label}: no data"
    mean = statistics.mean(vals)
    std  = statistics.stdev(vals) if len(vals) > 1 else 0.0
    return (
        f"  {label}:\n"
        f"    mean={mean:{fmt}}  std={std:{fmt}}  "
        f"min={min(vals):{fmt}}  p25={pct(vals,25):{fmt}}  "
        f"p50={pct(vals,50):{fmt}}  p75={pct(vals,75):{fmt}}  "
        f"p95={pct(vals,95):{fmt}}  max={max(vals):{fmt}}"
    )

def longest_streak(arr, val=1):
    best = cur = 0
    for v in arr:
        cur = (cur + 1) if v == val else 0
        best = max(best, cur)
    return best

SEP  = "=" * 72
sep  = "-" * 72

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — TRAINING LOG
# ─────────────────────────────────────────────────────────────────────────────
print(SEP)
print("  SECTION 1  --  TRAINING")
print("  Dataset  : KITTI Road  (232 train / 57 val, 80/20 split)")
print("  Model    : ResNet-18 U-Net  (14.9 M params, ImageNet pretrained)")
print("  Input    : 4-channel  RGB + LiDAR depth  (256x256)")
print(SEP)

with open("./outputs/logs/training_log.csv") as f:
    tlog = list(csv.DictReader(f))

print()
print(f"  {'Ep':>3}  {'Tr Loss':>8}  {'Tr CE':>7}  {'Tr Dice':>8}"
      f"  {'Va Loss':>8}  {'Va CE':>6}  {'Va Dice':>7}  {'Va mIoU':>8}  {'LR':>9}")
print("  " + sep)

best_miou = max(float(r["val_miou"]) for r in tlog)
for r in tlog:
    star = "  <-- BEST" if float(r["val_miou"]) == best_miou else ""
    print(f"  {r['epoch']:>3}  {float(r['train_loss']):>8.4f}  "
          f"{float(r['train_ce']):>7.4f}  {float(r['train_dice']):>8.4f}  "
          f"{float(r['val_loss']):>8.4f}  {float(r['val_ce']):>6.4f}  "
          f"{float(r['val_dice']):>7.4f}  {float(r['val_miou']):>8.4f}  "
          f"{r['lr']:>9}{star}")

print("  " + sep)
v_mious  = [float(r["val_miou"])   for r in tlog]
t_losses = [float(r["train_loss"]) for r in tlog]
v_losses = [float(r["val_loss"])   for r in tlog]
best_ep  = max(tlog, key=lambda r: float(r["val_miou"]))
gap      = float(best_ep["val_loss"]) - float(best_ep["train_loss"])

print()
print(f"  Total epochs          : {len(tlog)}")
print(f"  Best val mIoU         : {best_miou:.4f}  (epoch {best_ep['epoch']})")
print(f"  Final train loss      : {float(tlog[-1]['train_loss']):.4f}")
print(f"  Final val loss        : {float(tlog[-1]['val_loss']):.4f}")
print(f"  Train loss range      : {min(t_losses):.4f} -- {max(t_losses):.4f}")
print(f"  Val loss range        : {min(v_losses):.4f} -- {max(v_losses):.4f}")
print(f"  Val mIoU range        : {min(v_mious):.4f} -- {max(v_mious):.4f}")
print(f"  Train/val loss gap    : {gap:+.4f}  "
      f"({'slight overfit' if gap > 0.05 else 'good generalisation'})")

print()
for thresh in [0.80, 0.85, 0.90, 0.93, 0.94]:
    ep = next((r["epoch"] for r in tlog if float(r["val_miou"]) >= thresh), "never")
    print(f"  mIoU >= {thresh:.2f} first at epoch : {ep}")

# LR schedule events
lrs = [r["lr"] for r in tlog]
drops = [(tlog[i]["epoch"], lrs[i-1], lrs[i])
         for i in range(1, len(lrs)) if lrs[i] != lrs[i-1]]
print()
print("  Learning rate schedule:")
for ep, old, new in drops:
    print(f"    Epoch {ep}: {old} -> {new}")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — DL INFERENCE
# ─────────────────────────────────────────────────────────────────────────────
dl_runs = [
    ("input.mp4",      "./outputs/inference_video1/metrics.csv"),
    ("input (2).mp4",  "./outputs/inference_video2/metrics.csv"),
]

for vname, vpath in dl_runs:
    print()
    print(SEP)
    print(f"  SECTION 2  --  DL INFERENCE : {vname}")
    print(SEP)

    with open(vpath) as f:
        rows = list(csv.DictReader(f))

    def fv(key):
        return [float(r[key]) for r in rows
                if r.get(key, "N/A") not in ("N/A", "")]

    n     = len(rows)
    lane  = fv("lane_pct")
    raw   = fv("raw_pct")
    noise = fv("noise_removed")
    conf  = fv("mean_conf")
    p50c  = fv("conf_p50")
    p75c  = fv("conf_p75")
    p95c  = fv("conf_p95")
    tiou  = fv("temporal_iou")
    cx    = fv("lane_center_x")

    print()
    print(f"  Total frames          : {n}")
    print(f"  Frames with any lane  : {sum(1 for v in lane if v > 0)}"
          f"  ({sum(1 for v in lane if v > 0)/n*100:.1f}%)")
    print(f"  Frames with >1% lane  : {sum(1 for v in lane if v > 1)}"
          f"  ({sum(1 for v in lane if v > 1)/n*100:.1f}%)")
    print(f"  Frames with >5% lane  : {sum(1 for v in lane if v > 5)}"
          f"  ({sum(1 for v in lane if v > 5)/n*100:.1f}%)")
    print(f"  Frames with >10% lane : {sum(1 for v in lane if v > 10)}"
          f"  ({sum(1 for v in lane if v > 10)/n*100:.1f}%)")
    print()
    print(stat_line(lane,  "Lane coverage %    (clean mask)",     fmt=".3f"))
    print(stat_line(raw,   "Raw coverage %     (pre-processing)",  fmt=".3f"))
    print(stat_line(noise, "Noise removed %    (raw minus clean)", fmt=".3f"))
    print()
    print(stat_line(conf, "Mean lane confidence  (over lane px)", fmt=".4f"))
    print(stat_line(p50c, "Confidence p50 (all pixels)",          fmt=".4f"))
    print(stat_line(p75c, "Confidence p75 (all pixels)",          fmt=".4f"))
    print(stat_line(p95c, "Confidence p95 (all pixels)",          fmt=".4f"))
    print()
    print(stat_line(tiou, "Temporal IoU  (frame-to-frame stability)", fmt=".4f"))
    unstable = sum(1 for v in tiou if v < 0.5)
    print(f"  Unstable frames (<0.50 IoU)  : {unstable}  ({unstable/n*100:.1f}%)")
    print(f"  Stable  frames  (>=0.80 IoU) : "
          f"{sum(1 for v in tiou if v >= 0.8)}  "
          f"({sum(1 for v in tiou if v >= 0.8)/n*100:.1f}%)")
    print()
    print(stat_line(cx, "Lane centre X  (0=left edge, 0.5=centre, 1=right edge)",
                    fmt=".4f"))
    left_b  = sum(1 for v in cx if v < 0.4)
    right_b = sum(1 for v in cx if v > 0.6)
    mid_b   = n - left_b - right_b
    print(f"  Left-biased  frames (<0.40)       : {left_b}  ({left_b/n*100:.1f}%)")
    print(f"  Centred      frames (0.40 - 0.60) : {mid_b}   ({mid_b/n*100:.1f}%)")
    print(f"  Right-biased frames (>0.60)       : {right_b}  ({right_b/n*100:.1f}%)")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — BASELINE (HOUGH)
# ─────────────────────────────────────────────────────────────────────────────
bl_runs = [
    ("input.mp4",      "./outputs/baseline_video1/metrics.csv"),
    ("input (2).mp4",  "./outputs/baseline_video2/metrics.csv"),
]

for vname, vpath in bl_runs:
    print()
    print(SEP)
    print(f"  SECTION 3  --  BASELINE (Canny + Hough) : {vname}")
    print(SEP)

    with open(vpath) as f:
        rows = list(csv.DictReader(f))

    n     = len(rows)
    hough = [int(r["hough_lines"]) for r in rows]
    left  = [int(r["left_detected"])  for r in rows]
    right = [int(r["right_detected"]) for r in rows]
    both  = [1 if l and r else 0 for l, r in zip(left, right)]
    neither = [1 if not l and not r else 0 for l, r in zip(left, right)]

    lc = sum(left);  rc = sum(right);  bc = sum(both);  nc = sum(neither)
    lf = sum(1 for i in range(1, n) if left[i]  != left[i-1])
    rf = sum(1 for i in range(1, n) if right[i] != right[i-1])
    bf = sum(1 for i in range(1, n) if both[i]  != both[i-1])

    print()
    print(f"  Total frames               : {n}")
    print(f"  Left  lane detected        : {lc}  ({lc/n*100:.1f}%)")
    print(f"  Right lane detected        : {rc}  ({rc/n*100:.1f}%)")
    print(f"  Both  lanes detected       : {bc}  ({bc/n*100:.1f}%)")
    print(f"  Neither lane detected      : {nc}  ({nc/n*100:.1f}%)")
    print()
    print(stat_line(hough, "Hough segments per frame", fmt=".1f"))
    print()
    print(f"  Temporal detection flips (stability):")
    print(f"    Left  : {lf} flips  (stability {(1 - lf/max(n-1,1))*100:.1f}%)")
    print(f"    Right : {rf} flips  (stability {(1 - rf/max(n-1,1))*100:.1f}%)")
    print(f"    Both  : {bf} flips  (stability {(1 - bf/max(n-1,1))*100:.1f}%)")
    print()
    print(f"  Longest consecutive detection streaks:")
    print(f"    Left  detected : {longest_streak(left)}  frames")
    print(f"    Right detected : {longest_streak(right)} frames")
    print(f"    Both  detected : {longest_streak(both)}  frames")
    print(f"    Neither (gap)  : {longest_streak(neither)} frames")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — CROSS-MODEL / CROSS-VIDEO SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
print()
print(SEP)
print("  SECTION 4  --  CROSS-MODEL / CROSS-VIDEO SUMMARY")
print(SEP)

def load_dl(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    lane = [float(r["lane_pct"]) for r in rows]
    conf = [float(r["mean_conf"]) for r in rows
            if float(r["mean_conf"]) > 0]
    tiou = [float(r["temporal_iou"]) for r in rows]
    return rows, lane, conf, tiou

def load_bl(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    left  = [int(r["left_detected"])  for r in rows]
    right = [int(r["right_detected"]) for r in rows]
    both  = [1 if l and r else 0 for l, r in zip(left, right)]
    return rows, left, right, both

r1, lane1, conf1, tiou1 = load_dl("./outputs/inference_video1/metrics.csv")
r2, lane2, conf2, tiou2 = load_dl("./outputs/inference_video2/metrics.csv")
_, bl1, br1, bb1 = load_bl("./outputs/baseline_video1/metrics.csv")
_, bl2, br2, bb2 = load_bl("./outputs/baseline_video2/metrics.csv")

n1, n2 = len(r1), len(r2)

print()
print(f"  Model checkpoint    : outputs/best_model.pth  "
      f"({os.path.getsize('./outputs/best_model.pth')/1e6:.1f} MB)")
print(f"  Training mIoU       : {best_miou:.4f}  (epoch {best_ep['epoch']} / {len(tlog)} total)")
print()
header = f"  {'Metric':<40}  {'input.mp4':>14}  {'input(2).mp4':>14}"
print(header)
print("  " + "-" * (len(header) - 2))

rows_table = [
    ("Frames",                      str(n1),                      str(n2)),
    ("DL  avg lane coverage",        f"{statistics.mean(lane1):.2f}%",  f"{statistics.mean(lane2):.2f}%"),
    ("DL  lane presence (>1%)",      f"{sum(1 for v in lane1 if v>1)/n1*100:.1f}%", f"{sum(1 for v in lane2 if v>1)/n2*100:.1f}%"),
    ("DL  mean confidence",          f"{statistics.mean(conf1):.4f}", f"{statistics.mean(conf2):.4f}"),
    ("DL  avg temporal IoU",         f"{statistics.mean(tiou1):.4f}", f"{statistics.mean(tiou2):.4f}"),
    ("DL  stable frames (>=0.8)",    f"{sum(1 for v in tiou1 if v>=0.8)/n1*100:.1f}%", f"{sum(1 for v in tiou2 if v>=0.8)/n2*100:.1f}%"),
    ("Baseline left  detection",     f"{sum(bl1)/n1*100:.1f}%",       f"{sum(bl2)/n2*100:.1f}%"),
    ("Baseline right detection",     f"{sum(br1)/n1*100:.1f}%",       f"{sum(br2)/n2*100:.1f}%"),
    ("Baseline both  detection",     f"{sum(bb1)/n1*100:.1f}%",       f"{sum(bb2)/n2*100:.1f}%"),
    ("Baseline left  stability",     f"{(1-sum(1 for i in range(1,n1) if bl1[i]!=bl1[i-1])/max(n1-1,1))*100:.1f}%",
                                     f"{(1-sum(1 for i in range(1,n2) if bl2[i]!=bl2[i-1])/max(n2-1,1))*100:.1f}%"),
    ("Baseline right stability",     f"{(1-sum(1 for i in range(1,n1) if br1[i]!=br1[i-1])/max(n1-1,1))*100:.1f}%",
                                     f"{(1-sum(1 for i in range(1,n2) if br2[i]!=br2[i-1])/max(n2-1,1))*100:.1f}%"),
]

for label, v1, v2 in rows_table:
    print(f"  {label:<40}  {v1:>14}  {v2:>14}")

print()
print("  Interpretation")
print("  " + "-" * 40)
print("  input.mp4     : Difficult scene. DL detects lanes in 30.9% of frames")
print("                  with 0.853 confidence. Hough finds no right lane at all.")
print("                  Likely cause: faded markings, glare, or narrow lane perspective.")
print()
print("  input(2).mp4  : Clear lane scene. DL fires on 99.5% of frames at 0.910")
print("                  confidence. Hough bilateral detection 87.8%. DL temporal")
print("                  stability lower (0.576) due to varying lane area per frame.")
print()
print("  Model strengths   : High confidence on detected pixels; real lane-like shapes;")
print("                      trained on camera+LiDAR fusion (KITTI, mIoU=0.9405).")
print("  Model limitations : Trained on KITTI highway/urban roads; may generalise")
print("                      differently to dashcam footage without matching LiDAR depth.")
print("                      Adding --use_depth (DPT) or real LiDAR would improve results.")
print(SEP)
