"""
marker_boost.py — emphasise rare lane-marker classes at inference time.

Two independent, composable knobs:

1) apply_logit_bias(logits, bias)
   Adds a per-class bias to raw model logits BEFORE argmax.  Equivalent to
   Bayesian prior correction when bias = -log(class_prior)  (Menon et al.,
   "Long-tail learning via logit adjustment", 2020).  Arrows (3) and stop
   lines (5) are swamped by line pixels in training; a small positive bias
   flips predictions when the marker class is a close second.

2) boost_labels(pred) + decode_labels(pred)
   Morphologically dilate chosen marker classes in the argmaxed label map
   so sparse correct predictions form solid visible blobs in the rendered
   image.  Runs on already-saved predictions — no model needed.
"""

from __future__ import annotations

import json
from typing import Iterable, Mapping, Optional, Union

import numpy as np

# ---------------------------------------------------------------------------
# Class registry (matches preprocessing.py / infer_pt.py)
# ---------------------------------------------------------------------------

NUM_CLASSES = 7

CLASS_NAMES = [
    "background",   # 0
    "solid_line",   # 1
    "dash_line",    # 2
    "arrow",        # 3
    "other_line",   # 4
    "stop_line",    # 5
    "back_points",  # 6
]

LABEL_COLOURS = np.array([
    (0,   0,   0),    # 0 background
    (220, 0,   0),    # 1 solid_line
    (0,   220, 0),    # 2 dash_line
    (220, 220, 0),    # 3 arrow
    (0,   0,   220),  # 4 other_line
    (220, 0,   220),  # 5 stop_line
    (220, 220, 220),  # 6 back_points
], dtype=np.uint8)

# Default marker classes to boost (arrow + stop_line)
MARKER_CLASSES = (3, 5)

# Default logit bias for marker classes (tuned empirically)
_DEFAULT_BIAS = {3: 1.0, 5: 1.0}


# ---------------------------------------------------------------------------
# Knob 1 — logit bias (operates on torch tensors)
# ---------------------------------------------------------------------------

def apply_logit_bias(logits, bias: Optional[Mapping[int, float]]):
    """Add per-class bias to logits before argmax.

    Args:
        logits : torch.Tensor [B, C, H, W] — raw model output.
        bias   : dict {class_id: float_value} or None.

    Returns:
        logits tensor with bias added (same shape, same dtype).
    """
    if not bias:
        return logits
    import torch
    b = torch.zeros(logits.shape[1], dtype=logits.dtype, device=logits.device)
    for cls, val in bias.items():
        b[int(cls)] = val
    return logits + b.view(1, -1, 1, 1)


# ---------------------------------------------------------------------------
# Knob 2 — label-map dilation (operates on numpy arrays)
# ---------------------------------------------------------------------------

def _dilate_mask_numpy(mask: np.ndarray, kernel_size: int) -> np.ndarray:
    """Binary dilation using a square structuring element."""
    from scipy.ndimage import binary_dilation
    struct = np.ones((kernel_size, kernel_size), dtype=bool)
    return binary_dilation(mask, structure=struct)


def dilate_labels(
    pred: np.ndarray,
    classes: Iterable[int],
    kernel_size: int = 5,
) -> np.ndarray:
    """Morphologically dilate specified classes in a label map.

    Args:
        pred        : [H, W] int array of class IDs.
        classes     : class IDs to dilate, in order of increasing priority
                      (later classes overwrite earlier ones where they overlap).
        kernel_size : odd int; 1 = no dilation.

    Returns:
        New [H, W] int array.
    """
    if kernel_size <= 1:
        return pred.copy()
    if kernel_size % 2 == 0:
        raise ValueError("kernel_size must be odd")

    out = pred.copy()
    for cls in classes:
        mask = (pred == cls)
        if not mask.any():
            continue
        out[_dilate_mask_numpy(mask, kernel_size)] = cls
    return out


def boost_labels(
    pred: np.ndarray,
    marker_classes: Iterable[int] = MARKER_CLASSES,
    dilate_kernel: int = 5,
) -> np.ndarray:
    """Convenience wrapper: dilate marker classes in a label map."""
    return dilate_labels(pred, marker_classes, kernel_size=dilate_kernel)


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def decode_labels(pred: np.ndarray, palette: np.ndarray = LABEL_COLOURS) -> np.ndarray:
    """Render a [H, W] int label map to [H, W, 3] uint8 RGB."""
    clipped = np.clip(pred, 0, len(palette) - 1)
    return palette[clipped]


def class_pixel_counts(pred: np.ndarray, num_classes: int = NUM_CLASSES) -> np.ndarray:
    """Return length-C vector of pixel counts per class."""
    return np.bincount(pred.ravel(), minlength=num_classes)[:num_classes]


# ---------------------------------------------------------------------------
# CLI helper — pure Python, importable without torch
# ---------------------------------------------------------------------------

def parse_logit_bias(spec: Optional[str]):
    """Parse a --logit_bias CLI value.

    Accepts:
        None / '' / 'none'    → returns None  (no bias)
        'default'             → returns {3: 1.0, 5: 1.0}
        '{"3": 1.5, "5": 2}'  → returns {3: 1.5, 5: 2.0}

    Returns:
        dict {int: float} or None
    """
    if not spec or spec.lower() == 'none':
        return None
    if spec.lower() == 'default':
        return dict(_DEFAULT_BIAS)
    try:
        raw = json.loads(spec)
        return {int(k): float(v) for k, v in raw.items()}
    except (json.JSONDecodeError, ValueError) as e:
        raise ValueError(
            f"--logit_bias must be 'none', 'default', or a JSON dict, got: {spec!r}"
        ) from e
