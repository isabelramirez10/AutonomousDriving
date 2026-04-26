"""
FusionLane — ResNet-18 U-Net backbone (model_pt.py).

This file is auto-detected by train_pt.py:
    build_model() → FusionLaneModel (this file)
    (falls back to SimpleFusionLaneNet if this file is absent or broken)

Architecture
------------
Encoder : ResNet-18 pretrained on ImageNet-1K (torchvision).
          Weights (~45 MB) are downloaded automatically on the first run.
          The standard 3-channel conv1 is patched to accept 4 channels
          (RGB + region mask).  The extra channel is initialised to the
          mean of the three RGB filters so pretraining is preserved.

Decoder : Lightweight U-Net decoder with skip connections at each
          encoder stage.  Four transpose-conv blocks restore the spatial
          resolution from H/32 back to H.

Head    : 1×1 conv → 2 classes (background / lane).

Input   : FloatTensor [B, 4, H, W]  —  RGB (ImageNet-normalised) + region mask
Output  : FloatTensor [B, 2, H, W]  —  raw logits

Why pretrained weights help
---------------------------
ResNet-18 was trained on 1.2 M images and has learned rich low- and
mid-level features (edges, textures, shapes) that transfer directly to
road imagery.  Using these weights means:
  • Much faster convergence — meaningful lane detection in 5–10 epochs
    instead of 30–50 with random initialisation.
  • Better generalisation from smaller labelled datasets.
  • The backbone acts as a strong feature extractor from day one.
"""

import torch
import torch.nn as nn

try:
    from torchvision.models import resnet18, ResNet18_Weights
    def _resnet18_pretrained():
        return resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
except (ImportError, AttributeError):          # older torchvision
    from torchvision.models import resnet18
    def _resnet18_pretrained():
        return resnet18(pretrained=True)        # noqa: deprecated arg


# ─────────────────────────────────────────────────────────────────────────────
# Building blocks
# ─────────────────────────────────────────────────────────────────────────────

class _UpBlock(nn.Module):
    """Transpose-conv upsample (×2) followed by a 3×3 conv."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up   = nn.ConvTranspose2d(in_ch, out_ch,
                                       kernel_size=4, stride=2, padding=1,
                                       bias=False)
        self.bn1  = nn.BatchNorm2d(out_ch)
        self.conv = nn.Conv2d(out_ch, out_ch, kernel_size=3,
                              padding=1, bias=False)
        self.bn2  = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.bn1(self.up(x)))
        x = self.relu(self.bn2(self.conv(x)))
        return x


class _FuseBlock(nn.Module):
    """1×1 conv to fuse concatenated encoder skip + decoder features."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        self.bn   = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


# ─────────────────────────────────────────────────────────────────────────────
# Main model
# ─────────────────────────────────────────────────────────────────────────────

class FusionLaneModel(nn.Module):
    """
    ResNet-18 encoder  +  U-Net decoder for binary lane segmentation.

    Encoder spatial sizes (H = image height, W = image width):
        enc0  : H/2   × W/2,   64 ch
        enc1  : H/4   × W/4,   64 ch
        enc2  : H/8   × W/8,  128 ch
        enc3  : H/16  × W/16, 256 ch
        enc4  : H/32  × W/32, 512 ch

    Decoder (U-Net skip connections):
        dec4  : H/16, 256 ch  (upsample enc4 + fuse with enc3)
        dec3  : H/8,  128 ch  (upsample dec4 + fuse with enc2)
        dec2  : H/4,   64 ch  (upsample dec3 + fuse with enc1)
        dec1  : H/2,   32 ch  (upsample dec2 + fuse with enc0)
        out   : H,      2 ch  (upsample dec1 → head)
    """

    def __init__(self, num_classes: int = 2):
        super().__init__()

        print("[FusionLaneModel] Loading ImageNet pretrained ResNet-18 weights …")
        bb = _resnet18_pretrained()
        print("[FusionLaneModel] Pretrained weights loaded.")

        # ── Patch conv1 for 4-channel input ──────────────────────────────────
        old = bb.conv1                                    # (64, 3, 7, 7)
        new = nn.Conv2d(4, 64, kernel_size=7, stride=2,
                        padding=3, bias=False)
        with torch.no_grad():
            new.weight[:, :3] = old.weight               # copy RGB weights
            new.weight[:,  3] = old.weight.mean(dim=1)   # 4th ch = RGB mean
        bb.conv1 = new

        # ── Encoder stages ───────────────────────────────────────────────────
        self.enc0 = nn.Sequential(bb.conv1, bb.bn1, bb.relu)   # → 64,  H/2
        self.pool = bb.maxpool                                   # → 64,  H/4
        self.enc1 = bb.layer1                                    # → 64,  H/4
        self.enc2 = bb.layer2                                    # → 128, H/8
        self.enc3 = bb.layer3                                    # → 256, H/16
        self.enc4 = bb.layer4                                    # → 512, H/32

        # ── Decoder ──────────────────────────────────────────────────────────
        self.up4   = _UpBlock(512, 256)        # H/32 → H/16
        self.fuse4 = _FuseBlock(256 + 256, 256)

        self.up3   = _UpBlock(256, 128)        # H/16 → H/8
        self.fuse3 = _FuseBlock(128 + 128, 128)

        self.up2   = _UpBlock(128, 64)         # H/8  → H/4
        self.fuse2 = _FuseBlock(64 + 64, 64)

        self.up1   = _UpBlock(64, 32)          # H/4  → H/2
        self.fuse1 = _FuseBlock(32 + 64, 32)

        self.up0   = _UpBlock(32, 32)          # H/2  → H
        self.head  = nn.Conv2d(32, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encode
        e0 = self.enc0(x)            # 64,  H/2
        p  = self.pool(e0)           # 64,  H/4
        e1 = self.enc1(p)            # 64,  H/4
        e2 = self.enc2(e1)           # 128, H/8
        e3 = self.enc3(e2)           # 256, H/16
        e4 = self.enc4(e3)           # 512, H/32

        # Decode with skip connections
        d = self.fuse4(torch.cat([self.up4(e4), e3], dim=1))   # 256, H/16
        d = self.fuse3(torch.cat([self.up3(d),  e2], dim=1))   # 128, H/8
        d = self.fuse2(torch.cat([self.up2(d),  e1], dim=1))   # 64,  H/4
        d = self.fuse1(torch.cat([self.up1(d),  e0], dim=1))   # 32,  H/2
        d = self.up0(d)                                          # 32,  H
        return self.head(d)                                      # 2,   H


# ─────────────────────────────────────────────────────────────────────────────
# Quick architecture test  (python model_pt.py)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    model = FusionLaneModel(num_classes=2)
    model.eval()

    total  = sum(p.numel() for p in model.parameters())
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"\nTotal parameters : {total:,}")
    print(f"Frozen parameters: {frozen:,}")

    dummy = torch.randn(2, 4, 512, 512)
    with torch.no_grad():
        out = model(dummy)
    print(f"Input  shape : {list(dummy.shape)}")
    print(f"Output shape : {list(out.shape)}")
    assert out.shape == (2, 2, 512, 512), "Shape mismatch!"
    print("Architecture test passed.")
