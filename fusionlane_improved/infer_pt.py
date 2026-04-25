import argparse
import os

import cv2
import numpy as np
import torch
from scipy import ndimage
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset_pt import FusionLaneDataset
from train_pt import SimpleFusionLaneNet

# =============================================================================
# EASY CONFIG — edit these values instead of using command-line flags.
# data_dir accepts: TFRecord folder | image folder | video file path
# Command-line flags always override these if you pass them explicitly.
# =============================================================================
CONFIG = {
    "data_dir":     "./data",                   # TFRecord folder | image folder | video file
    "model_path":   "./outputs/best_model.pth", # checkpoint produced by train_pt.py
    "output_dir":   "./outputs/inference",      # base folder; 4 subfolders created inside
    "image_height": 512,                        # must match the height used during training
    "image_width":  512,                        # must match the width  used during training
    "batch_size":   4,                          # must be divisible by 4
    "threshold":    0.65,                       # P(lane) min to keep a pixel (0.0–1.0)
    "min_blob":     200,                        # smallest connected component kept (pixels)
    "num_workers":  0,                          # DataLoader workers
}
# =============================================================================


def confidence_filter(logits, threshold=0.65):
    probs = torch.softmax(logits, dim=1)
    lane_conf = probs[:, 1, :, :]
    lane_mask = (lane_conf >= threshold).to(torch.uint8)
    return lane_mask, lane_conf


def cleanup_mask(mask, min_blob=200):
    # Clean small noisy fragments while preserving contiguous lanes.
    m = mask.astype(bool)
    m = ndimage.binary_opening(m, structure=np.ones((3, 3), dtype=bool))
    m = ndimage.binary_closing(m, structure=np.ones((5, 5), dtype=bool))

    labeled, n = ndimage.label(m)
    if n == 0:
        return np.zeros_like(mask, dtype=np.uint8)

    counts = np.bincount(labeled.ravel())
    keep = np.zeros_like(m, dtype=bool)
    for comp_idx in range(1, len(counts)):
        if counts[comp_idx] >= int(min_blob):
            keep |= labeled == comp_idx

    return keep.astype(np.uint8)


def build_model(model_path, device):
    model = SimpleFusionLaneNet(in_channels=4, num_classes=2).to(device)
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def denorm_to_bgr(image_tensor):
    # Input expected as normalized [4, H, W]. Convert RGB back to uint8 BGR.
    mean = torch.tensor([0.485, 0.456, 0.406], device=image_tensor.device).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=image_tensor.device).view(3, 1, 1)
    rgb = image_tensor[:3] * std + mean
    rgb = (rgb.clamp(0.0, 1.0) * 255.0).byte().cpu().numpy().transpose(1, 2, 0)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def ensure_dirs(base):
    subdirs = ["raw", "cleaned", "heatmap", "comparison"]
    for sub in subdirs:
        os.makedirs(os.path.join(base, sub), exist_ok=True)


def parse_args():
    parser = argparse.ArgumentParser(description="FusionLane Improved inference")
    parser.add_argument("--data_dir",     type=str,   default=CONFIG["data_dir"])
    parser.add_argument("--model_path",   type=str,   default=CONFIG["model_path"])
    parser.add_argument("--output_dir",   type=str,   default=CONFIG["output_dir"])
    parser.add_argument("--image_height", type=int,   default=CONFIG["image_height"])
    parser.add_argument("--image_width",  type=int,   default=CONFIG["image_width"])
    parser.add_argument("--batch_size",   type=int,   default=CONFIG["batch_size"])
    parser.add_argument("--threshold",    type=float, default=CONFIG["threshold"])
    parser.add_argument("--min_blob",     type=int,   default=CONFIG["min_blob"])
    parser.add_argument("--num_workers",  type=int,   default=CONFIG["num_workers"])
    return parser.parse_args()


def main():
    args = parse_args()

    assert args.batch_size % 4 == 0, "batch_size must be divisible by 4"
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model checkpoint not found: {args.model_path}")

    ensure_dirs(args.output_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(args.model_path, device)
    print(f"Loaded checkpoint from {args.model_path}")

    ds = FusionLaneDataset(
        data_dir=args.data_dir,
        split="test",
        image_h=args.image_height,
        image_w=args.image_width,
    )
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
    )

    saved = 0
    with torch.no_grad():
        for batch in tqdm(loader, desc="infer"):
            image = batch["image"].to(device)
            logits = model(image)
            raw_pred = torch.argmax(logits, dim=1).cpu().numpy().astype(np.uint8)
            filt_mask, lane_conf = confidence_filter(logits, threshold=args.threshold)

            for i in range(image.size(0)):
                index = saved
                raw = raw_pred[i] * 255
                cleaned = cleanup_mask(filt_mask[i].cpu().numpy(), min_blob=args.min_blob) * 255
                conf = lane_conf[i].cpu().numpy()

                raw_path = os.path.join(args.output_dir, "raw", f"pred_{index:04d}.png")
                cleaned_path = os.path.join(args.output_dir, "cleaned", f"pred_{index:04d}.png")
                heat_path = os.path.join(args.output_dir, "heatmap", f"pred_{index:04d}.png")
                cmp_path = os.path.join(args.output_dir, "comparison", f"pred_{index:04d}.png")

                cv2.imwrite(raw_path, raw)
                cv2.imwrite(cleaned_path, cleaned)

                conf_u8 = np.clip(conf * 255.0, 0, 255).astype(np.uint8)
                heat = cv2.applyColorMap(conf_u8, cv2.COLORMAP_JET)
                cv2.imwrite(heat_path, heat)

                base = denorm_to_bgr(image[i])
                raw_bgr = cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR)
                cleaned_bgr = cv2.cvtColor(cleaned, cv2.COLOR_GRAY2BGR)
                panel = np.concatenate([base, raw_bgr, cleaned_bgr, heat], axis=1)
                cv2.imwrite(cmp_path, panel)

                saved += 1

    print(f"Inference complete. {saved} samples saved to: {args.output_dir}/")
    print("  raw/         - argmax predictions (original behavior)")
    print("  cleaned/     - confidence-filtered + morphologically cleaned")
    print("  heatmap/     - P(lane) confidence maps")
    print("  comparison/  - side-by-side for manual inspection")


if __name__ == "__main__":
    main()
