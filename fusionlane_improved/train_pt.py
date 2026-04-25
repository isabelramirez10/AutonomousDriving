import argparse
import csv
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset_pt import FusionLaneDataset

# =============================================================================
# EASY CONFIG — edit these values instead of using command-line flags.
# Command-line flags always override these if you pass them explicitly.
# =============================================================================
CONFIG = {
    "data_dir":     "./data",    # TFRecord folder | image folder | video file path
    "model_dir":    "./outputs", # where best_model.pth is saved
    "epochs":       10,          # max training epochs
    "batch_size":   4,           # must be divisible by 4
    "num_workers":  0,           # DataLoader workers (keep 0 with TF-based dataset)
    "image_height": 512,         # resize all frames to this height
    "image_width":  512,         # resize all frames to this width
    "loss_alpha":   0.5,         # CE fraction in hybrid loss (0=pure Dice, 1=pure CE)
    "lane_weight":  2.0,         # class weight for lane pixels in CE loss
    "bg_weight":    1.0,         # class weight for background pixels in CE loss
    "lr":           1e-3,        # initial Adam learning rate
    "grad_clip":    1.0,         # gradient clipping max-norm (0 = disabled)
    "patience":     5,           # early-stopping: stop if no improvement for N epochs
    "scheduler":    "plateau",   # LR scheduler: "plateau" | "cosine" | "none"
}
# =============================================================================


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class SimpleFusionLaneNet(nn.Module):
    """
    Lightweight encoder-decoder for binary lane segmentation.

    Architecture:
      Encoder: three strided Conv2d blocks (×2 downsampling each) producing
               feature maps at 1/4 of the input resolution.
      Decoder: two ConvTranspose2d blocks restoring spatial resolution,
               followed by a 1×1 classification head.

    Input  : [B, 4, H, W]  — RGB (ImageNet-normalised) + region mask channel
    Output : [B, 2, H, W]  — background and lane logits
    """

    def __init__(self, in_channels=4, num_classes=2):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, num_classes, kernel_size=1),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------

class DiceLoss(nn.Module):
    """
    Soft Dice loss for binary lane segmentation.

    Computes 1 - Dice coefficient over the lane class (index 1).
    The smooth term prevents division by zero on all-background batches.

    Reference:
        Milletari, F., Navab, N., & Ahmadi, S. A. (2016). V-Net: Fully
        Convolutional Neural Networks for Volumetric Medical Image Segmentation.
        3DV 2016. https://doi.org/10.1109/3DV.2016.79
    """

    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, target):
        probs    = torch.softmax(logits, dim=1)[:, 1, :, :]   # P(lane) [B,H,W]
        target_f = (target == 1).float()

        probs    = probs.reshape(probs.size(0), -1)
        target_f = target_f.reshape(target_f.size(0), -1)

        intersection = (probs * target_f).sum(dim=1)
        denom        = probs.sum(dim=1) + target_f.sum(dim=1)
        dice         = (2.0 * intersection + self.smooth) / (denom + self.smooth)
        return 1.0 - dice.mean()


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_miou(logits, labels, num_classes=2):
    """Mean Intersection-over-Union across all classes."""
    pred = logits.argmax(dim=1)
    ious = []
    for cls in range(num_classes):
        inter = ((pred == cls) & (labels == cls)).sum().item()
        union = ((pred == cls) | (labels == cls)).sum().item()
        if union > 0:
            ious.append(inter / union)
    return sum(ious) / len(ious) if ious else 0.0


# ---------------------------------------------------------------------------
# Model builder
# ---------------------------------------------------------------------------

def build_model(num_classes=2):
    """
    Returns FusionLaneModel from model_pt.py if present; falls back to
    SimpleFusionLaneNet otherwise.
    """
    try:
        from model_pt import FusionLaneModel
        model = FusionLaneModel(num_classes=num_classes)
        print(f"Using FusionLaneModel from model_pt.py  (num_classes={num_classes})")
        return model
    except Exception:
        print(f"Using SimpleFusionLaneNet  (num_classes={num_classes})")
        return SimpleFusionLaneNet(in_channels=4, num_classes=num_classes)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="FusionLane Improved training")
    parser.add_argument("--data_dir",     type=str,   default=CONFIG["data_dir"])
    parser.add_argument("--model_dir",    type=str,   default=CONFIG["model_dir"])
    parser.add_argument("--epochs",       type=int,   default=CONFIG["epochs"])
    parser.add_argument("--batch_size",   type=int,   default=CONFIG["batch_size"])
    parser.add_argument("--num_workers",  type=int,   default=CONFIG["num_workers"])
    parser.add_argument("--image_height", type=int,   default=CONFIG["image_height"])
    parser.add_argument("--image_width",  type=int,   default=CONFIG["image_width"])
    parser.add_argument("--lr",           type=float, default=CONFIG["lr"])
    parser.add_argument("--loss_alpha",   type=float, default=CONFIG["loss_alpha"],
                        help="CE fraction in hybrid loss (0=pure Dice, 1=pure CE)")
    parser.add_argument("--lane_weight",  type=float, default=CONFIG["lane_weight"])
    parser.add_argument("--bg_weight",    type=float, default=CONFIG["bg_weight"])
    parser.add_argument("--grad_clip",    type=float, default=CONFIG["grad_clip"],
                        help="Gradient clipping max-norm; 0 to disable")
    parser.add_argument("--patience",     type=int,   default=CONFIG["patience"],
                        help="Early-stopping patience in epochs")
    parser.add_argument("--scheduler",    type=str,   default=CONFIG["scheduler"],
                        choices=["plateau", "cosine", "none"])
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Training / validation loops
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, ce_loss, dice_loss,
                    device, alpha, grad_clip):
    model.train()
    running = {"loss": 0.0, "ce": 0.0, "dice": 0.0}

    pbar = tqdm(loader, desc="  train", leave=False)
    for batch in pbar:
        image = batch["image"].to(device)
        label = batch["label"].to(device)

        optimizer.zero_grad()
        logits = model(image)

        ce   = ce_loss(logits, label)
        dice = dice_loss(logits, label)
        loss = alpha * ce + (1.0 - alpha) * dice

        loss.backward()

        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)

        optimizer.step()

        running["loss"] += loss.item()
        running["ce"]   += ce.item()
        running["dice"] += dice.item()

        pbar.set_postfix(
            loss=f"{loss.item():.4f}",
            ce=f"{ce.item():.4f}",
            dice=f"{dice.item():.4f}",
        )

    n = max(len(loader), 1)
    return {k: v / n for k, v in running.items()}


@torch.no_grad()
def validate(model, loader, ce_loss, dice_loss, device, alpha):
    model.eval()
    running = {"loss": 0.0, "ce": 0.0, "dice": 0.0, "miou": 0.0}

    pbar = tqdm(loader, desc="  val  ", leave=False)
    for batch in pbar:
        image  = batch["image"].to(device)
        label  = batch["label"].to(device)
        logits = model(image)

        ce   = ce_loss(logits, label)
        dice = dice_loss(logits, label)
        loss = alpha * ce + (1.0 - alpha) * dice
        miou = compute_miou(logits, label)

        running["loss"] += loss.item()
        running["ce"]   += ce.item()
        running["dice"] += dice.item()
        running["miou"] += miou

        pbar.set_postfix(
            loss=f"{loss.item():.4f}",
            ce=f"{ce.item():.4f}",
            dice=f"{dice.item():.4f}",
            miou=f"{miou:.4f}",
        )

    n = max(len(loader), 1)
    return {k: v / n for k, v in running.items()}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    assert args.batch_size % 4 == 0, \
        f"batch_size must be divisible by 4, got {args.batch_size}"

    os.makedirs(args.model_dir, exist_ok=True)
    log_dir = os.path.join(args.model_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")
    if device.type == "cuda":
        print(f"GPU    : {torch.cuda.get_device_name(0)}")

    # Datasets — shuffle=False to preserve 4-frame temporal sequence groups
    train_ds = FusionLaneDataset(args.data_dir, split="train",
                                 image_h=args.image_height, image_w=args.image_width)
    val_ds   = FusionLaneDataset(args.data_dir, split="val",
                                 image_h=args.image_height, image_w=args.image_width)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=False,               # keep temporal order
                              num_workers=args.num_workers, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False,
                              num_workers=args.num_workers, drop_last=False)

    model = build_model(num_classes=2).to(device)

    class_weights = torch.tensor(
        [args.bg_weight, args.lane_weight], dtype=torch.float32
    ).to(device)
    ce_loss   = nn.CrossEntropyLoss(weight=class_weights)
    dice_loss = DiceLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Learning-rate scheduler
    if args.scheduler == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=3
        )
    elif args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=args.lr * 1e-2
        )
    else:
        scheduler = None

    best_miou      = -1.0
    best_path      = os.path.join(args.model_dir, "best_model.pth")
    epochs_no_impr = 0   # early-stopping counter

    # CSV training log
    log_path = os.path.join(log_dir, "training_log.csv")
    log_file = open(log_path, "w", newline="")
    log_csv  = csv.DictWriter(
        log_file,
        fieldnames=["epoch", "train_loss", "train_ce", "train_dice",
                    "val_loss", "val_ce", "val_dice", "val_miou", "lr"]
    )
    log_csv.writeheader()

    print(f"\nTraining for up to {args.epochs} epoch(s)  "
          f"[patience={args.patience}, scheduler={args.scheduler}]\n")

    try:
        for epoch in range(1, args.epochs + 1):
            current_lr = optimizer.param_groups[0]["lr"]
            print(f"Epoch {epoch}/{args.epochs}  lr={current_lr:.2e}")

            train_stats = train_one_epoch(
                model, train_loader, optimizer, ce_loss, dice_loss,
                device, args.loss_alpha, args.grad_clip
            )
            val_stats = validate(
                model, val_loader, ce_loss, dice_loss, device, args.loss_alpha
            )

            print(f"  train: loss={train_stats['loss']:.4f}  "
                  f"ce={train_stats['ce']:.4f}  dice={train_stats['dice']:.4f}")
            print(f"  val:   loss={val_stats['loss']:.4f}  "
                  f"ce={val_stats['ce']:.4f}  dice={val_stats['dice']:.4f}  "
                  f"mIoU={val_stats['miou']:.4f}")

            # LR scheduler step
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_stats["miou"])
            elif scheduler is not None:
                scheduler.step()

            # CSV log
            log_csv.writerow({
                "epoch":      epoch,
                "train_loss": f"{train_stats['loss']:.6f}",
                "train_ce":   f"{train_stats['ce']:.6f}",
                "train_dice": f"{train_stats['dice']:.6f}",
                "val_loss":   f"{val_stats['loss']:.6f}",
                "val_ce":     f"{val_stats['ce']:.6f}",
                "val_dice":   f"{val_stats['dice']:.6f}",
                "val_miou":   f"{val_stats['miou']:.6f}",
                "lr":         f"{current_lr:.2e}",
            })
            log_file.flush()

            # Checkpoint
            if val_stats["miou"] > best_miou:
                best_miou      = val_stats["miou"]
                epochs_no_impr = 0
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "best_miou":        best_miou,
                        "epoch":            epoch,
                        "args":             vars(args),
                    },
                    best_path,
                )
                print(f"  -> New best model saved: {best_path}  (mIoU={best_miou:.4f})")
            else:
                epochs_no_impr += 1
                if epochs_no_impr >= args.patience:
                    print(f"\nEarly stopping: no improvement for {args.patience} epochs.")
                    break

    finally:
        log_file.close()

    print(f"\nTraining complete. Best val mIoU: {best_miou:.4f}")
    print(f"Training log saved: {log_path}")


if __name__ == "__main__":
    main()
