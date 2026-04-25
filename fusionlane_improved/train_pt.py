import argparse
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
    "epochs":       10,          # number of training epochs
    "batch_size":   4,           # must be divisible by 4
    "num_workers":  0,           # DataLoader workers (keep 0 with TF-based dataset)
    "image_height": 512,         # resize all frames to this height
    "image_width":  512,         # resize all frames to this width
    "loss_alpha":   0.5,         # CE fraction in hybrid loss (0=pure Dice, 1=pure CE)
    "lane_weight":  2.0,         # class weight for lane pixels in CE loss
    "bg_weight":    1.0,         # class weight for background pixels in CE loss
    "lr":           1e-3,        # Adam learning rate
}
# =============================================================================


class SimpleFusionLaneNet(nn.Module):
    """Small segmentation network for smoke tests and baseline training."""

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
        x = self.encoder(x)
        return self.decoder(x)


class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, target):
        probs = torch.softmax(logits, dim=1)[:, 1, :, :]
        target_f = (target == 1).float()

        probs = probs.reshape(probs.size(0), -1)
        target_f = target_f.reshape(target_f.size(0), -1)

        intersection = (probs * target_f).sum(dim=1)
        denom = probs.sum(dim=1) + target_f.sum(dim=1)
        dice = (2.0 * intersection + self.smooth) / (denom + self.smooth)
        return 1.0 - dice.mean()


def compute_miou(logits, labels, num_classes=2):
    pred = logits.argmax(dim=1)
    ious = []
    for cls in range(num_classes):
        p = pred == cls
        l = labels == cls
        inter = (p & l).sum().item()
        union = (p | l).sum().item()
        if union > 0:
            ious.append(inter / union)
    return sum(ious) / len(ious) if ious else 0.0


def build_model(num_classes=2):
    try:
        from model_pt import FusionLaneModel

        model = FusionLaneModel(num_classes=num_classes)
        # If model signature differs, fallback to a simple reference model.
        return model
    except Exception:
        return SimpleFusionLaneNet(in_channels=4, num_classes=num_classes)


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
                        help="Weight for CE in hybrid loss. Dice weight is (1-loss_alpha).")
    parser.add_argument("--lane_weight",  type=float, default=CONFIG["lane_weight"])
    parser.add_argument("--bg_weight",    type=float, default=CONFIG["bg_weight"])
    return parser.parse_args()


def train_one_epoch(model, loader, optimizer, ce_loss, dice_loss, device, alpha):
    model.train()
    running = {"loss": 0.0, "ce": 0.0, "dice": 0.0}

    pbar = tqdm(loader, desc="train", leave=False)
    for batch in pbar:
        image = batch["image"].to(device)
        label = batch["label"].to(device)

        optimizer.zero_grad()
        logits = model(image)

        ce = ce_loss(logits, label)
        dice = dice_loss(logits, label)
        loss = alpha * ce + (1.0 - alpha) * dice

        loss.backward()
        optimizer.step()

        running["loss"] += loss.item()
        running["ce"] += ce.item()
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

    pbar = tqdm(loader, desc="val", leave=False)
    for batch in pbar:
        image = batch["image"].to(device)
        label = batch["label"].to(device)

        logits = model(image)
        ce = ce_loss(logits, label)
        dice = dice_loss(logits, label)
        loss = alpha * ce + (1.0 - alpha) * dice
        miou = compute_miou(logits, label, num_classes=2)

        running["loss"] += loss.item()
        running["ce"] += ce.item()
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


def main():
    args = parse_args()

    assert args.batch_size % 4 == 0, "batch_size must be divisible by 4"
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(os.path.join(args.model_dir, "logs"), exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_ds = FusionLaneDataset(
        data_dir=args.data_dir,
        split="train",
        image_h=args.image_height,
        image_w=args.image_width,
    )
    val_ds = FusionLaneDataset(
        data_dir=args.data_dir,
        split="val",
        image_h=args.image_height,
        image_w=args.image_width,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
    )

    model = build_model(num_classes=2).to(device)
    class_weights = torch.tensor([args.bg_weight, args.lane_weight], dtype=torch.float32).to(device)
    ce_loss = nn.CrossEntropyLoss(weight=class_weights)
    dice_loss = DiceLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_miou = -1.0
    best_path = os.path.join(args.model_dir, "best_model.pth")

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        train_stats = train_one_epoch(model, train_loader, optimizer, ce_loss, dice_loss, device, args.loss_alpha)
        val_stats = validate(model, val_loader, ce_loss, dice_loss, device, args.loss_alpha)

        print(
            "train: loss={loss:.4f} ce={ce:.4f} dice={dice:.4f}".format(**train_stats)
        )
        print(
            "val:   loss={loss:.4f} ce={ce:.4f} dice={dice:.4f} mIoU={miou:.4f}".format(**val_stats)
        )

        if val_stats["miou"] > best_miou:
            best_miou = val_stats["miou"]
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "best_miou": best_miou,
                    "args": vars(args),
                },
                best_path,
            )
            print(f"New best model saved: {best_path}")

    print(f"Training complete. Best val mIoU: {best_miou:.4f}")


if __name__ == "__main__":
    main()
