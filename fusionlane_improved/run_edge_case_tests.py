"""
FusionLane — edge case test suite.
Run with:  python run_edge_case_tests.py
"""
import sys, os, traceback, tempfile, shutil
import cv2, numpy as np, torch

sys.path.insert(0, os.path.dirname(__file__))

PASS = []
FAIL = []

def check(name, fn):
    try:
        fn()
        PASS.append(name)
        print(f"  PASS  {name}")
    except Exception as e:
        FAIL.append((name, str(e)))
        print(f"  FAIL  {name}: {e}")

# =============================================================================
# 1. PREPROCESSING
# =============================================================================
from infer_media import (preprocess, apply_clahe, smooth_confidence,
                         adaptive_otsu_threshold, clean_mask, apply_roi_mask,
                         fit_lane_polynomials, make_kitti_seg)

print("\n[1] PREPROCESSING")

def t_preprocess_black():
    t = preprocess(np.zeros((480, 640, 3), np.uint8), 256, 256)
    assert t.shape == (4, 256, 256)
    assert not torch.isnan(t).any()

def t_preprocess_white():
    t = preprocess(np.full((480, 640, 3), 255, np.uint8), 256, 256)
    assert t.shape == (4, 256, 256)
    assert not torch.isnan(t).any()

def t_preprocess_with_depth():
    depth = np.random.rand(256, 256).astype(np.float32)
    t = preprocess(np.zeros((480, 640, 3), np.uint8), 256, 256, depth)
    assert t.shape == (4, 256, 256)
    assert float((t[3] - torch.from_numpy(depth)).abs().max()) < 1e-5

def t_preprocess_tiny_input():
    t = preprocess(np.zeros((1, 1, 3), np.uint8), 256, 256)
    assert t.shape == (4, 256, 256)

def t_clahe_black():
    out = apply_clahe(np.zeros((480, 640, 3), np.uint8))
    assert out.shape == (480, 640, 3)

def t_clahe_white():
    out = apply_clahe(np.full((480, 640, 3), 255, np.uint8))
    assert out.shape == (480, 640, 3)

def t_clahe_random():
    img = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
    out = apply_clahe(img)
    assert out.shape == img.shape

check("preprocess: all-black frame", t_preprocess_black)
check("preprocess: all-white frame", t_preprocess_white)
check("preprocess: real depth channel", t_preprocess_with_depth)
check("preprocess: 1x1 pixel input", t_preprocess_tiny_input)
check("apply_clahe: all-black", t_clahe_black)
check("apply_clahe: all-white", t_clahe_white)
check("apply_clahe: random image", t_clahe_random)

# =============================================================================
# 2. TEMPORAL SMOOTHING
# =============================================================================
print("\n[2] TEMPORAL EMA SMOOTHING")

def t_ema_first_frame():
    curr = np.random.rand(256, 256).astype(np.float32)
    out  = smooth_confidence(curr, None, 0.6)
    assert np.allclose(out, curr), "First frame should pass through unchanged"

def t_ema_alpha_zero():
    curr = np.ones((256, 256), np.float32)
    prev = np.zeros((256, 256), np.float32)
    out  = smooth_confidence(curr, prev, 0.0)
    assert np.allclose(out, curr), "alpha=0 should ignore history"

def t_ema_alpha_one():
    curr = np.ones((256, 256), np.float32)
    prev = np.zeros((256, 256), np.float32)
    out  = smooth_confidence(curr, prev, 1.0)
    assert np.allclose(out, prev), "alpha=1 should freeze on previous frame"

def t_ema_stays_bounded():
    for _ in range(50):
        curr = np.random.rand(256, 256).astype(np.float32)
        prev = np.random.rand(256, 256).astype(np.float32)
        out  = smooth_confidence(curr, prev, 0.6)
        assert out.min() >= 0.0 and out.max() <= 1.0, "EMA output out of [0,1]"

check("smooth_confidence: first frame (prev=None)", t_ema_first_frame)
check("smooth_confidence: alpha=0 passes curr through", t_ema_alpha_zero)
check("smooth_confidence: alpha=1 freezes on prev", t_ema_alpha_one)
check("smooth_confidence: output always in [0,1]", t_ema_stays_bounded)

# =============================================================================
# 3. ADAPTIVE OTSU THRESHOLD
# =============================================================================
print("\n[3] ADAPTIVE OTSU THRESHOLD")

def t_otsu_all_zeros():
    t = adaptive_otsu_threshold(np.zeros((256, 256), np.float32))
    assert 0.10 <= t <= 0.70, f"Otsu on zeros out of range: {t}"

def t_otsu_all_ones():
    t = adaptive_otsu_threshold(np.ones((256, 256), np.float32))
    assert 0.10 <= t <= 0.70, f"Otsu on ones out of range: {t}"

def t_otsu_bimodal():
    conf = np.full((256, 256), 0.02, np.float32)
    conf[100:150, 100:150] = 0.90
    t = adaptive_otsu_threshold(conf)
    assert 0.10 <= t <= 0.70, f"Bimodal Otsu out of range: {t}"

def t_otsu_floor():
    # Uniform near-zero — Otsu returns 0, should be clamped to floor=0.10
    t = adaptive_otsu_threshold(np.full((256, 256), 0.001, np.float32))
    assert t >= 0.10, f"Floor not enforced: {t}"

def t_otsu_ceiling():
    # Uniform near-1 — Otsu may return high value, should be clamped to 0.70
    t = adaptive_otsu_threshold(np.full((256, 256), 0.99, np.float32))
    assert t <= 0.70, f"Ceiling not enforced: {t}"

check("adaptive_otsu: all-zeros map", t_otsu_all_zeros)
check("adaptive_otsu: all-ones map", t_otsu_all_ones)
check("adaptive_otsu: bimodal confidence map", t_otsu_bimodal)
check("adaptive_otsu: floor enforced at 0.10", t_otsu_floor)
check("adaptive_otsu: ceiling enforced at 0.70", t_otsu_ceiling)

# =============================================================================
# 4. CLEAN MASK / ROI / POLYNOMIAL FITTING
# =============================================================================
print("\n[4] CLEAN MASK / ROI / POLYNOMIAL FITTING")

def t_clean_empty():
    r = clean_mask(np.zeros((256, 256), np.uint8), 100)
    assert r.sum() == 0

def t_clean_all_ones():
    r = clean_mask(np.ones((256, 256), np.uint8), 100)
    assert r.sum() > 0, "All-lane mask should survive cleaning"

def t_clean_tiny_blob():
    m = np.zeros((256, 256), np.uint8)
    m[100:103, 100:103] = 1     # 9 pixels
    assert clean_mask(m, 100).sum() == 0, "Tiny blob should be removed"

def t_clean_large_blob():
    m = np.zeros((256, 256), np.uint8)
    m[50:200, 50:200] = 1       # 22500 pixels
    assert clean_mask(m, 100).sum() > 0, "Large blob should be kept"

def t_roi_no_mask():
    m = np.ones((256, 256), np.uint8)
    assert apply_roi_mask(m, 0.0).sum() == 256 * 256

def t_roi_full_mask():
    m = np.ones((256, 256), np.uint8)
    assert apply_roi_mask(m, 1.0).sum() == 0, "roi_top=1.0 should zero everything"

def t_roi_half():
    m = np.ones((256, 256), np.uint8)
    out = apply_roi_mask(m, 0.5)
    assert out[:128, :].sum() == 0, "Top half not zeroed"
    assert out[128:, :].sum() == 128 * 256, "Bottom half modified unexpectedly"

def t_poly_empty():
    assert fit_lane_polynomials(np.zeros((256, 256), np.uint8)).sum() == 0

def t_poly_vertical_line():
    m = np.zeros((256, 256), np.uint8)
    m[10:240, 127:130] = 1
    assert fit_lane_polynomials(m, min_pixels=30).sum() > 0

def t_poly_tiny_component():
    m = np.zeros((256, 256), np.uint8)
    m[100:103, 100:103] = 1
    assert fit_lane_polynomials(m, min_pixels=30).sum() == 0, "Tiny component should be skipped"

def t_poly_horizontal_fallback():
    m = np.zeros((256, 256), np.uint8)
    m[128:130, 50:200] = 1     # y_span < 5 -> falls back to blob
    assert fit_lane_polynomials(m, min_pixels=30).sum() > 0

def t_poly_output_in_bounds():
    m = np.zeros((256, 256), np.uint8)
    m[20:220, 60:70] = 1
    out = fit_lane_polynomials(m, min_pixels=30)
    assert out.shape == (256, 256), "Shape mismatch"
    assert set(np.unique(out)).issubset({0, 1}), "Output should be binary 0/1"

check("clean_mask: empty input", t_clean_empty)
check("clean_mask: all-lane input", t_clean_all_ones)
check("clean_mask: tiny blob removed (< min_blob)", t_clean_tiny_blob)
check("clean_mask: large blob kept (> min_blob)", t_clean_large_blob)
check("apply_roi_mask: roi_top=0.0 no change", t_roi_no_mask)
check("apply_roi_mask: roi_top=1.0 zeros all", t_roi_full_mask)
check("apply_roi_mask: roi_top=0.5 correct split", t_roi_half)
check("fit_lane_polynomials: empty mask", t_poly_empty)
check("fit_lane_polynomials: vertical line fitted", t_poly_vertical_line)
check("fit_lane_polynomials: tiny component skipped", t_poly_tiny_component)
check("fit_lane_polynomials: horizontal blob fallback", t_poly_horizontal_fallback)
check("fit_lane_polynomials: output stays binary 0/1", t_poly_output_in_bounds)

# =============================================================================
# 5. KITTI SEG COLORMAP
# =============================================================================
print("\n[5] KITTI SEG COLORMAP")

def t_kitti_background():
    seg = make_kitti_seg(np.zeros((256, 256), np.uint8))
    assert seg.shape == (256, 256, 3)
    # Background = BGR [0, 0, 220] — all blue channel should be 220
    assert (seg[:, :, 2] == 220).all(), "Background red channel wrong"
    assert (seg[:, :, 0] == 0).all(),   "Background blue channel wrong"

def t_kitti_lane():
    seg = make_kitti_seg(np.ones((256, 256), np.uint8))
    assert seg.shape == (256, 256, 3)
    # Lane = BGR [220, 0, 220] — magenta
    assert (seg[:, :, 0] == 220).all(), "Lane blue channel wrong"
    assert (seg[:, :, 2] == 220).all(), "Lane red channel wrong"

check("make_kitti_seg: background is red BGR[0,0,220]", t_kitti_background)
check("make_kitti_seg: lane is magenta BGR[220,0,220]", t_kitti_lane)

# =============================================================================
# 6. DATASET LOADER EDGE CASES
# =============================================================================
print("\n[6] DATASET LOADER")
from dataset_pt import FusionLaneDataset
from torch.utils.data import DataLoader

def t_dataset_dummy():
    ds = FusionLaneDataset("./nonexistent_path", split="train",
                           image_h=64, image_w=64)
    assert len(ds) > 0, "Dummy dataset should have samples"
    item = ds[0]
    assert item["image"].shape == (4, 64, 64)
    assert item["label"].shape == (64, 64)

def t_dataset_paired_no_depth():
    tmp = tempfile.mkdtemp()
    try:
        os.makedirs(f"{tmp}/train/images")
        os.makedirs(f"{tmp}/train/masks")
        for i in range(3):
            cv2.imwrite(f"{tmp}/train/images/test_{i:04d}.jpg",
                        np.random.randint(0,255,(64,64,3),np.uint8))
            cv2.imwrite(f"{tmp}/train/masks/test_{i:04d}.png",
                        (np.random.rand(64,64) > 0.5).astype(np.uint8)*255)
        ds = FusionLaneDataset(tmp, split="train", image_h=64, image_w=64)
        assert len(ds) == 3
        item = ds[0]
        assert item["image"].shape == (4, 64, 64)
        # Channel 4 should be all-ones (no depth folder)
        assert (item["image"][3] == 1.0).all(), "No-depth channel should be all-ones"
    finally:
        shutil.rmtree(tmp)

def t_dataset_paired_with_depth():
    tmp = tempfile.mkdtemp()
    try:
        os.makedirs(f"{tmp}/train/images")
        os.makedirs(f"{tmp}/train/masks")
        os.makedirs(f"{tmp}/train/depths")
        for i in range(2):
            cv2.imwrite(f"{tmp}/train/images/test_{i:04d}.jpg",
                        np.random.randint(0,255,(64,64,3),np.uint8))
            cv2.imwrite(f"{tmp}/train/masks/test_{i:04d}.png",
                        (np.random.rand(64,64)>0.5).astype(np.uint8)*255)
            depth = np.random.randint(0,256,(64,64),np.uint8)
            cv2.imwrite(f"{tmp}/train/depths/test_{i:04d}.png", depth)
        ds = FusionLaneDataset(tmp, split="train", image_h=64, image_w=64)
        assert len(ds) == 2
        item = ds[0]
        # Channel 4 should NOT be all-ones when depth exists
        assert not (item["image"][3] == 1.0).all(), "Depth channel should use actual depth"
    finally:
        shutil.rmtree(tmp)

def t_dataset_empty_folder():
    tmp = tempfile.mkdtemp()
    try:
        os.makedirs(f"{tmp}/train/images")
        os.makedirs(f"{tmp}/train/masks")
        # No files — should fall back to dummy
        ds = FusionLaneDataset(tmp, split="train", image_h=64, image_w=64)
        assert len(ds) > 0, "Should fall back to dummy data"
    finally:
        shutil.rmtree(tmp)

def t_dataset_batch_divisible():
    ds = FusionLaneDataset("./nonexistent_path", split="train",
                           image_h=64, image_w=64)
    loader = DataLoader(ds, batch_size=4, drop_last=True)
    batch = next(iter(loader))
    assert batch["image"].shape[0] == 4

check("dataset: dummy fallback for missing path", t_dataset_dummy)
check("dataset: paired folder without depths/ (ch4=ones)", t_dataset_paired_no_depth)
check("dataset: paired folder with depths/ (ch4=real depth)", t_dataset_paired_with_depth)
check("dataset: empty images/ folder falls back to dummy", t_dataset_empty_folder)
check("dataset: DataLoader batch_size=4 works", t_dataset_batch_divisible)

# =============================================================================
# 7. EVAL_FUSION EDGE CASES
# =============================================================================
print("\n[7] EVAL_FUSION METRICS")
from eval_fusion import compute_metrics

def t_metrics_perfect():
    gt   = np.ones((256, 256), bool)
    pred = np.ones((256, 256), bool)
    m = compute_metrics(pred, gt)
    assert abs(m["miou"] - 1.0) < 1e-6, f"Perfect pred mIoU != 1.0: {m['miou']}"
    assert abs(m["dice"] - 1.0) < 1e-6
    assert abs(m["recall"] - 1.0) < 1e-6
    assert abs(m["precision"] - 1.0) < 1e-6

def t_metrics_all_wrong():
    gt   = np.ones((256, 256), bool)
    pred = np.zeros((256, 256), bool)
    m = compute_metrics(pred, gt)
    assert m["recall"] == 0.0, "All-wrong recall should be 0"
    assert m["dice"] == 0.0

def t_metrics_empty_gt():
    gt   = np.zeros((256, 256), bool)
    pred = np.zeros((256, 256), bool)
    m = compute_metrics(pred, gt)
    assert m["miou"] == 1.0, "Empty GT + empty pred should give mIoU=1"

def t_metrics_all_fp():
    # Predict all lane, GT is all background
    gt   = np.zeros((256, 256), bool)
    pred = np.ones((256, 256), bool)
    m = compute_metrics(pred, gt)
    assert m["precision"] == 0.0, "All FP should give precision=0"
    assert m["recall"] == 0.0, "No TP so recall=0"

check("compute_metrics: perfect prediction (mIoU=1)", t_metrics_perfect)
check("compute_metrics: all wrong (recall=0, dice=0)", t_metrics_all_wrong)
check("compute_metrics: empty GT + empty pred (mIoU=1)", t_metrics_empty_gt)
check("compute_metrics: all false positives (precision=0)", t_metrics_all_fp)

# =============================================================================
# 8. COMPARE_OUTPUTS EDGE CASES
# =============================================================================
print("\n[8] COMPARE_OUTPUTS")
from compare_outputs import _load_and_resize, _add_column_header, _add_row_header

def t_load_missing_file():
    img = _load_and_resize("/nonexistent/path/pred_9999.png")
    assert img.shape == (321, 321, 3), "Missing file should return 321x321 placeholder"

def t_load_small_image():
    tmp = tempfile.mktemp(suffix=".png")
    try:
        cv2.imwrite(tmp, np.zeros((10, 10, 3), np.uint8))
        img = _load_and_resize(tmp)
        assert img.shape == (321, 321, 3), "Should upscale small image to 321x321"
    finally:
        if os.path.exists(tmp): os.remove(tmp)

def t_column_header():
    img    = np.zeros((321, 321, 3), np.uint8)
    result = _add_column_header(img, "Test Label")
    assert result.shape[0] > 321, "Header bar should increase image height"
    assert result.shape[1] == 321

def t_row_header():
    panel = _add_row_header(339, "f010")
    assert panel.ndim == 3, "Row header should be 3-channel"

check("compare_outputs: missing image returns placeholder", t_load_missing_file)
check("compare_outputs: small image upscaled to 321x321", t_load_small_image)
check("compare_outputs: column header adds height", t_column_header)
check("compare_outputs: row header is 3-channel", t_row_header)

# =============================================================================
# 9. MODEL FORWARD PASS EDGE CASES
# =============================================================================
print("\n[9] MODEL FORWARD PASS")
from train_pt import build_model, compute_miou

def t_model_forward_256():
    m = build_model(num_classes=2)
    m.eval()
    with torch.no_grad():
        out = m(torch.zeros(1, 4, 256, 256))
    assert out.shape == (1, 2, 256, 256)
    assert not torch.isnan(out).any()

def t_model_forward_512():
    m = build_model(num_classes=2)
    m.eval()
    with torch.no_grad():
        out = m(torch.zeros(1, 4, 512, 512))
    assert out.shape == (1, 2, 512, 512)
    assert not torch.isnan(out).any()

def t_model_all_zeros_input():
    m = build_model(num_classes=2)
    m.eval()
    with torch.no_grad():
        out = m(torch.zeros(2, 4, 256, 256))
    assert not torch.isnan(out).any(), "NaN in output for zero input"
    assert not torch.isinf(out).any(), "Inf in output for zero input"

def t_model_extreme_input():
    m = build_model(num_classes=2)
    m.eval()
    with torch.no_grad():
        x = torch.full((1, 4, 256, 256), 1e4)
        out = m(x)
    # Logits may be large but should not be NaN
    assert not torch.isnan(out).any(), "NaN for extreme input"

def t_compute_miou_perfect():
    logits = torch.zeros(1, 2, 8, 8)
    logits[:, 1] = 10.0          # strongly predict lane
    labels = torch.ones(1, 8, 8, dtype=torch.long)
    miou   = compute_miou(logits, labels)
    assert abs(miou - 1.0) < 1e-4, f"Perfect mIoU should be 1.0, got {miou}"

def t_compute_miou_all_wrong():
    logits = torch.zeros(1, 2, 8, 8)
    logits[:, 0] = 10.0          # strongly predict background
    labels = torch.ones(1, 8, 8, dtype=torch.long)
    miou   = compute_miou(logits, labels)
    assert miou < 0.5, f"All-wrong mIoU should be low, got {miou}"

check("model: forward pass 256x256", t_model_forward_256)
check("model: forward pass 512x512 (non-training size)", t_model_forward_512)
check("model: all-zeros input produces no NaN/Inf", t_model_all_zeros_input)
check("model: extreme input (1e4) produces no NaN", t_model_extreme_input)
check("compute_miou: perfect prediction = 1.0", t_compute_miou_perfect)
check("compute_miou: all-wrong prediction < 0.5", t_compute_miou_all_wrong)

# =============================================================================
# SUMMARY
# =============================================================================
total = len(PASS) + len(FAIL)
print(f"\n{'='*60}")
print(f"  EDGE CASE TEST RESULTS: {len(PASS)}/{total} passed")
print(f"{'='*60}")
if FAIL:
    print(f"\n  FAILURES ({len(FAIL)}):")
    for name, err in FAIL:
        print(f"    FAIL: {name}")
        print(f"          {err}")
else:
    print("  All tests passed — no bugs found.")
