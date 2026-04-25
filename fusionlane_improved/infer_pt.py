import argparse
import os

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset_pt import FusionLaneDataset
from infer_media import confidence_filter, clean_mask, load_model, make_heatmap, denorm_bgr

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
    "threshold":    0.65,                       # P(lane) min to keep a pixel (0.0-1.0)
    "min_blob":     200,                        # smallest connected component kept (pixels)
    "num_workers":  0,                          # DataLoader workers
}
# =============================================================================


def ensure_dirs(base):
    for sub in ["raw", "cleaned", "heatmap", "comparison"]:
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
    model  = load_model(args.model_path, device)

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
            image    = batch["image"].to(device)
            logits   = model(image)
            raw_pred = torch.argmax(logits, dim=1).cpu().numpy().astype(np.uint8)
            filt_mask, lane_conf = confidence_filter(logits.cpu(), args.threshold)

            for i in range(image.size(0)):
                stem    = f"pred_{saved:04d}"
                raw     = raw_pred[i] * 255
                cleaned = clean_mask(filt_mask[i].numpy(), args.min_blob) * 255
                conf    = lane_conf[i].numpy()

                cv2.imwrite(os.path.join(args.output_dir, "raw",        f"{stem}.png"), raw)
                cv2.imwrite(os.path.join(args.output_dir, "cleaned",    f"{stem}.png"), cleaned)
                cv2.imwrite(os.path.join(args.output_dir, "heatmap",    f"{stem}.png"), make_heatmap(conf))

                base      = denorm_bgr(image[i].cpu())
                raw_bgr   = cv2.cvtColor(raw,     cv2.COLOR_GRAY2BGR)
                clean_bgr = cv2.cvtColor(cleaned, cv2.COLOR_GRAY2BGR)
                panel     = np.concatenate([base, raw_bgr, clean_bgr, make_heatmap(conf)], axis=1)
                cv2.imwrite(os.path.join(args.output_dir, "comparison", f"{stem}.png"), panel)

                saved += 1

    print(f"Inference complete. {saved} samples saved to: {args.output_dir}/")
    print("  raw/         - argmax predictions")
    print("  cleaned/     - confidence-filtered + morphologically cleaned")
    print("  heatmap/     - P(lane) confidence maps")
    print("  comparison/  - side-by-side for manual inspection")


if __name__ == "__main__":
    main()
