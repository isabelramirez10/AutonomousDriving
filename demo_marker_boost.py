"""
demo_marker_boost.py — apply the marker-boost visual pass to an already-
rendered prediction PNG and save a before/after side-by-side.

Does NOT require the model or dataset.  Reverses the palette in the PNG back
to class IDs, runs boost_labels, and re-renders.

Usage:
    python demo_marker_boost.py INPUT.png OUTPUT.png [--kernel 5] [--classes 3 5]
"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from marker_boost import (
    LABEL_COLOURS,
    MARKER_CLASSES,
    CLASS_NAMES,
    boost_labels,
    decode_labels,
    class_pixel_counts,
)


def rgb_to_labels(rgb: np.ndarray, palette: np.ndarray = LABEL_COLOURS) -> np.ndarray:
    """Nearest-colour palette inversion: [H, W, 3] RGB → [H, W] class IDs."""
    H, W, _ = rgb.shape
    diff = rgb.reshape(-1, 1, 3).astype(int) - palette.astype(int)  # [N, C, 3]
    dist = (diff ** 2).sum(axis=2)                                   # [N, C]
    return dist.argmin(axis=1).reshape(H, W).astype(np.int32)


def side_by_side(a: np.ndarray, b: np.ndarray, gap: int = 4) -> np.ndarray:
    h, w, _ = a.shape
    out = np.zeros((h, w * 2 + gap, 3), dtype=np.uint8)
    out[:, :w] = a
    out[:, w + gap:] = b
    return out


def _format_counts(counts: np.ndarray) -> str:
    total = counts.sum()
    parts = []
    for cls, n in enumerate(counts):
        if n == 0:
            continue
        pct = 100.0 * n / total
        parts.append(f"  {cls} {CLASS_NAMES[cls]:<14} {n:>8d}  ({pct:5.2f}%)")
    return "\n".join(parts)


def main():
    p = argparse.ArgumentParser(
        description="Visualise marker_boost dilation on a rendered prediction PNG.",
    )
    p.add_argument("input",  type=Path, help="Path to rendered prediction PNG")
    p.add_argument("output", type=Path, help="Path to side-by-side output PNG")
    p.add_argument("--kernel", type=int, default=5,
                   help="Dilation kernel size, must be odd (default: 5)")
    p.add_argument("--classes", type=int, nargs="+",
                   default=list(MARKER_CLASSES),
                   help=f"Class IDs to boost (default: {list(MARKER_CLASSES)} = arrow, stop_line)")
    args = p.parse_args()

    rgb_in = np.asarray(Image.open(args.input).convert("RGB"))
    pred   = rgb_to_labels(rgb_in)

    boosted_pred = boost_labels(
        pred,
        marker_classes=tuple(args.classes),
        dilate_kernel=args.kernel,
    )
    rgb_out = decode_labels(boosted_pred)

    combined = side_by_side(rgb_in, rgb_out)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(combined).save(args.output)

    print(f"Input : {args.input}")
    print(f"Output: {args.output}   (left = original, right = boosted)")
    print(f"Kernel: {args.kernel}   Boosted classes: {args.classes}")
    print()
    print("Class pixel distribution — before:")
    print(_format_counts(class_pixel_counts(pred)))
    print()
    print("Class pixel distribution — after:")
    print(_format_counts(class_pixel_counts(boosted_pred)))


if __name__ == "__main__":
    main()
