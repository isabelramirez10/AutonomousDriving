"""
FusionLane model — PyTorch rewrite.

Architecture (faithful to the original TF/tf-slim code):
  XceptionBackbone  →  ASPP  →  ConvLSTM  →  Decoder  →  logits

Key design notes to match the original:
  • Xception layers: slim.conv2d / slim.separable_conv2d with activation_fn=None
    (from xception_arg_scope).  ReLU and BN are applied MANUALLY afterward,
    giving the pattern:  conv → ReLU → BN.
  • ASPP layers: slim.conv2d inside resnet_v2.resnet_arg_scope, which adds
    BN + ReLU internally, giving:  conv → BN → ReLU.
  • LSTM-section conv and all Decoder convs: inside xception_arg_scope
    (activation_fn=None), with NO manual ReLU/BN after them → plain linear conv.
  • BN momentum=1−decay=0.0003, eps=0.001 to match TF defaults.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_BN_DECAY   = 0.9997
_BN_MOM     = 1.0 - _BN_DECAY   # 0.0003  (PyTorch momentum convention)
_BN_EPS     = 0.001


def _bn(channels):
    return nn.BatchNorm2d(channels, momentum=_BN_MOM, eps=_BN_EPS)


class _ConvReluBn(nn.Module):
    """
    Plain Conv2d (no bias, no activation inside)  → ReLU → BN.
    Matches the manual  slim.conv2d → relu → batch_norm  pattern in Xception.
    """
    def __init__(self, in_ch, out_ch, kernel=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel,
                              stride=stride, padding=padding, bias=False)
        self.bn   = _bn(out_ch)

    def forward(self, x):
        return self.bn(F.relu(self.conv(x), inplace=True))


class _SepConvReluBn(nn.Module):
    """
    Depthwise-separable conv (no bias, no activation inside) → ReLU → BN.
    Matches the manual  slim.separable_conv2d → relu → batch_norm  pattern.
    """
    def __init__(self, in_ch, out_ch, stride=1, dilation=1):
        super().__init__()
        pad = dilation           # keeps spatial size for 3×3 kernel
        self.dw = nn.Conv2d(in_ch, in_ch, 3,
                            stride=stride, padding=pad, dilation=dilation,
                            groups=in_ch, bias=False)
        self.pw = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn = _bn(out_ch)

    def forward(self, x):
        return self.bn(F.relu(self.pw(self.dw(x)), inplace=True))


# ---------------------------------------------------------------------------
# Xception Backbone
# ---------------------------------------------------------------------------

class XceptionBackbone(nn.Module):
    """
    Custom Xception encoder that fuses a region/mask branch into the RGB branch.

    Spatial downsampling (for 321×321 input):
        /1  →  /2  →  /4  →  /8  →  /16
       321   161    81    41    21

    Returns
    -------
    enc_out : [B, 1024, H/16, W/16]   encoder feature map
    low_level : [B, 192, H/4,  W/4]   low-level features for skip connection
    """

    def __init__(self):
        super().__init__()

        # --- Region branch (net_a) ---
        # input: [B, 1, H, W]
        self.a_conv1 = nn.Conv2d(1,  8,  3, padding=1, bias=False)
        self.a_bn1   = _bn(8)
        self.a_conv2 = nn.Conv2d(8,  16, 3, padding=1, bias=False)
        self.a_bn2   = _bn(16)
        self.a_sep3  = _SepConvReluBn(16, 32, stride=2)   # → H/2

        # --- Main branch (net) ---
        # input: [B, 3, H, W]
        self.b_conv1 = nn.Conv2d(3,  24, 3, padding=1, bias=False)
        self.b_bn1   = _bn(24)
        self.b_conv2 = nn.Conv2d(24, 48, 3, padding=1, bias=False)
        self.b_bn2   = _bn(48)
        self.b_sep1  = _SepConvReluBn(48, 96, stride=2)   # → H/2

        # After fuse1 concat(96+32=128):
        self.res0_conv = nn.Conv2d(128, 192, 1, stride=2, bias=False)
        self.res0_bn   = _bn(192)

        # --- Block 1 ---
        # region branch
        self.a_sep4 = _SepConvReluBn(32, 64)
        self.a_sep5 = _SepConvReluBn(64, 64, stride=2)    # → H/4

        # main branch  (input 128ch from fuse1)
        self.b1_sep1 = _SepConvReluBn(128, 192)
        self.b1_sep2 = _SepConvReluBn(192, 192)
        self.b1_sep3 = _SepConvReluBn(192, 192, stride=2) # → H/4

        # After fuse2 concat(192+64=256):
        self.res1_conv = nn.Conv2d(256, 384, 1, stride=2, bias=False)
        self.res1_bn   = _bn(384)

        # --- Block 2 ---
        # region branch
        self.a_sep6 = _SepConvReluBn(64,  128)
        self.a_sep7 = _SepConvReluBn(128, 128, stride=2)  # → H/8

        # main branch  (input 256ch from fuse2)
        self.b2_sep1 = _SepConvReluBn(256, 384)
        self.b2_sep2 = _SepConvReluBn(384, 384)
        self.b2_sep3 = _SepConvReluBn(384, 384, stride=2) # → H/8

        # After fuse3 concat(384+128=512):

        # --- Middle flow  (8 residual blocks) ---
        self.middle = nn.ModuleList([
            nn.ModuleList([
                _SepConvReluBn(512, 512),
                _SepConvReluBn(512, 512),
                _SepConvReluBn(512, 512),
            ])
            for _ in range(8)
        ])

        # --- Exit flow ---
        self.exit_sep1 = _SepConvReluBn(512, 512)
        self.exit_sep2 = _SepConvReluBn(512, 512)
        self.exit_sep3 = _SepConvReluBn(512, 1024, stride=2)  # → H/16
        self.exit_bn   = _bn(1024)   # final BN on exit (matches original)

    def forward(self, x, region):
        """
        x      : [B, 3, H, W]
        region : [B, 1, H, W]
        """

        # -- Region branch --
        na = self.a_bn1(F.relu(self.a_conv1(region), inplace=True))
        na = self.a_bn2(F.relu(self.a_conv2(na),     inplace=True))
        na = self.a_sep3(na)    # [B, 32, H/2, W/2]

        # -- Main branch --
        nb = self.b_bn1(F.relu(self.b_conv1(x),  inplace=True))
        nb = self.b_bn2(F.relu(self.b_conv2(nb), inplace=True))
        nb = self.b_sep1(nb)    # [B, 96, H/2, W/2]
        nb = F.relu(nb, inplace=True)   # extra relu present in original

        # Fuse 1
        net = torch.cat([nb, na], dim=1)              # [B, 128, H/2, W/2]
        residual = self.res0_bn(self.res0_conv(net))  # [B, 192, H/4, W/4]

        # Block 1 — region
        na = self.a_sep4(na)    # [B, 64, H/2, W/2]
        na = self.a_sep5(na)    # [B, 64, H/4, W/4]

        # Block 1 — main
        net = self.b1_sep1(net)  # [B, 192, H/2, W/2]
        net = self.b1_sep2(net)  # [B, 192, H/2, W/2]
        net = self.b1_sep3(net)  # [B, 192, H/4, W/4]
        net = net + residual
        low_level = net          # [B, 192, H/4, W/4]  ← skip connection

        # Fuse 2
        net = torch.cat([net, na], dim=1)             # [B, 256, H/4, W/4]
        residual = self.res1_bn(self.res1_conv(net))  # [B, 384, H/8, W/8]

        # Block 2 — region
        na = self.a_sep6(na)    # [B, 128, H/4, W/4]
        na = self.a_sep7(na)    # [B, 128, H/8, W/8]

        # Block 2 — main
        net = self.b2_sep1(net)  # [B, 384, H/4, W/4]
        net = self.b2_sep2(net)  # [B, 384, H/4, W/4]
        net = self.b2_sep3(net)  # [B, 384, H/8, W/8]
        net = net + residual

        # Fuse 3
        net = torch.cat([net, na], dim=1)  # [B, 512, H/8, W/8]

        # Middle flow
        for block in self.middle:
            res = net
            for layer in block:
                net = layer(net)
            net = net + res

        # Exit flow
        net = self.exit_sep1(net)   # [B, 512, H/8, W/8]
        net = self.exit_sep2(net)   # [B, 512, H/8, W/8]
        net = self.exit_sep3(net)   # [B, 1024, H/16, W/16]
        net = F.relu(net, inplace=True)
        net = self.exit_bn(net)

        return net, low_level


# ---------------------------------------------------------------------------
# ASPP — Atrous Spatial Pyramid Pooling
# ---------------------------------------------------------------------------

class ASPP(nn.Module):
    """
    Inside resnet_v2.resnet_arg_scope: each conv gets BN + ReLU automatically.
    Pattern: Conv → BN → ReLU   (opposite order from Xception backbone).
    """

    def __init__(self, in_ch=1024, depth=256):
        super().__init__()

        def _conv_bn_relu(in_c, out_c, kernel=1, padding=0, dilation=1):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel,
                          padding=padding, dilation=dilation, bias=False),
                _bn(out_c),
                nn.ReLU(inplace=True),
            )

        self.c1x1      = _conv_bn_relu(in_ch,  depth)
        self.c3x3_r6   = _conv_bn_relu(in_ch,  depth, 3, padding=6,  dilation=6)
        self.c3x3_r12  = _conv_bn_relu(in_ch,  depth, 3, padding=12, dilation=12)
        self.c3x3_r18  = _conv_bn_relu(in_ch,  depth, 3, padding=18, dilation=18)
        self.global_br = _conv_bn_relu(in_ch,  depth)

        # 5 branches → depth*5 → project to depth
        self.proj = _conv_bn_relu(depth * 5, depth)

    def forward(self, x):
        size = x.shape[2:]
        c1  = self.c1x1(x)
        c2  = self.c3x3_r6(x)
        c3  = self.c3x3_r12(x)
        c4  = self.c3x3_r18(x)
        gp  = x.mean(dim=[2, 3], keepdim=True)
        gp  = self.global_br(gp)
        gp  = F.interpolate(gp, size=size, mode='bilinear', align_corners=True)
        out = torch.cat([c1, c2, c3, c4, gp], dim=1)
        return self.proj(out)          # [B, depth, H/16, W/16]


# ---------------------------------------------------------------------------
# ConvLSTM
# ---------------------------------------------------------------------------

class _ConvLSTMCell(nn.Module):
    def __init__(self, in_ch, hidden_ch, kernel=3):
        super().__init__()
        self.hidden_ch = hidden_ch
        pad = kernel // 2
        # Single conv computes all 4 gates at once
        self.gates = nn.Conv2d(in_ch + hidden_ch, 4 * hidden_ch, kernel, padding=pad)

    def forward(self, x, h, c):
        i, f, g, o = self.gates(torch.cat([x, h], dim=1)).chunk(4, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        g = torch.tanh(g)
        o = torch.sigmoid(o)
        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next


class ConvLSTM(nn.Module):
    """
    Single-layer ConvLSTM with input dropout.
    input  : [B, T, C, H, W]
    output : [B, T, hidden_ch, H, W]   (return_sequences=True)
    """

    def __init__(self, in_ch, hidden_ch=64, kernel=3, dropout=0.6):
        super().__init__()
        self.cell     = _ConvLSTMCell(in_ch, hidden_ch, kernel)
        self.dropout  = dropout
        self.hidden_ch = hidden_ch

    def forward(self, x, training=False):
        B, T, C, H, W = x.shape
        device = x.device
        h = torch.zeros(B, self.hidden_ch, H, W, device=device, dtype=x.dtype)
        c = torch.zeros(B, self.hidden_ch, H, W, device=device, dtype=x.dtype)
        outputs = []
        for t in range(T):
            inp = x[:, t]
            if training and self.dropout > 0:
                inp = F.dropout(inp, p=self.dropout, training=True)
            h, c = self.cell(inp, h, c)
            outputs.append(h.unsqueeze(1))
        return torch.cat(outputs, dim=1)   # [B, T, hidden, H, W]


# ---------------------------------------------------------------------------
# Full FusionLane Model
# ---------------------------------------------------------------------------

class FusionLaneModel(nn.Module):
    """
    Full FusionLane segmentation model.

    forward(image, region, training=False)
        image  : [B, 3, H, W]   raw pixel values (means are 0 so no subtraction)
        region : [B, 1, H, W]   road-region mask
    returns logits [B, num_classes, H, W]

    Note: batch_size must be divisible by 4 (the ConvLSTM treats groups of 4
    samples as a temporal sequence, matching the original architecture).
    """

    def __init__(self, num_classes=7, batch_norm_decay=0.9997):
        super().__init__()

        self.backbone  = XceptionBackbone()
        self.aspp      = ASPP(in_ch=1024, depth=256)

        # LSTM input projection — plain conv, no BN/activation (inside xception_arg_scope)
        self.lstm_proj = nn.Conv2d(256, 64, 1, bias=True)

        self.conv_lstm = ConvLSTM(in_ch=64, hidden_ch=64, kernel=3, dropout=0.6)

        # Decoder — all plain conv, no BN/activation (inside xception_arg_scope)
        # Low-level skip: 192 → 24
        self.ll_conv   = nn.Conv2d(192, 24, 1, bias=True)

        # After LSTM reshape: 64-ch feature maps
        self.dec_c1    = nn.Conv2d(64, 64, 3, padding=1, bias=True)
        self.dec_c2    = nn.Conv2d(64, 64, 3, padding=1, bias=True)

        # After concat with low-level (64+24=88):
        self.dec_c3    = nn.Conv2d(88, 44, 3, padding=1, bias=True)
        self.dec_c4    = nn.Conv2d(44, 44, 3, padding=1, bias=True)

        # Input-derived feature: RGB → 1 channel (at full resolution)
        self.inp_conv  = nn.Conv2d(3, 1, 1, bias=True)

        # Final classifier: 44+1=45 → num_classes  (no activation, no BN)
        self.final     = nn.Conv2d(45, num_classes, 1, bias=True)

    def forward(self, image, region, training=False):
        """
        image   : [B, 3, H, W]
        region  : [B, 1, H, W]
        """
        B = image.shape[0]
        assert B % 4 == 0, f"batch size must be divisible by 4, got {B}"

        # ── Encoder ──────────────────────────────────────────────────────
        enc, low_level = self.backbone(image, region)
        # enc       : [B, 1024, H/16, W/16]
        # low_level : [B,  192, H/4,  W/4 ]

        # ── ASPP ─────────────────────────────────────────────────────────
        aspp = self.aspp(enc)          # [B, 256, H/16, W/16]

        # ── ConvLSTM ─────────────────────────────────────────────────────
        h16, w16 = aspp.shape[2], aspp.shape[3]
        z = self.lstm_proj(aspp)       # [B, 64, H/16, W/16]  — plain linear
        # Group every 4 batch items as a temporal sequence
        z = z.view(B // 4, 4, 64, h16, w16)
        z = self.conv_lstm(z, training=training)   # [B/4, 4, 64, H/16, W/16]
        z = z.contiguous().view(B, 64, h16, w16)   # [B, 64, H/16, W/16]

        # ── Decoder ──────────────────────────────────────────────────────
        ll      = self.ll_conv(low_level)           # [B, 24, H/4, W/4]
        ll_size = ll.shape[2:]                      # (H/4, W/4)

        # Process LSTM output and upsample to low-level resolution
        d = self.dec_c1(z)                          # [B, 64, H/16, W/16]
        d = self.dec_c2(d)                          # [B, 64, H/16, W/16]
        d = F.interpolate(d, size=ll_size,
                          mode='bilinear', align_corners=True)  # [B, 64, H/4, W/4]

        # Concat with skip
        d = torch.cat([d, ll], dim=1)               # [B, 88, H/4, W/4]
        d = self.dec_c3(d)                          # [B, 44, H/4, W/4]
        d = self.dec_c4(d)                          # [B, 44, H/4, W/4]

        # Upsample to full resolution
        full_size = image.shape[2:]
        d = F.interpolate(d, size=full_size,
                          mode='bilinear', align_corners=True)  # [B, 44, H, W]

        # Input-derived feature (RGB → 1ch, at full resolution)
        inp_feat = self.inp_conv(image)             # [B, 1, H, W]

        # Final concat and classifier
        d      = torch.cat([d, inp_feat], dim=1)   # [B, 45, H, W]
        logits = self.final(d)                      # [B, num_classes, H, W]

        return logits
