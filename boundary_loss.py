"""
boundary_loss.py — boundary loss for multi-class segmentation.

Implementation of the distance-transform-based boundary loss from

    Kervadec, Bouchtiba, Desrosiers, Granger, Dolz, Ben Ayed.
    "Boundary loss for highly unbalanced segmentation."
    MIDL 2019, extended in Medical Image Analysis 2021.
    https://arxiv.org/abs/1812.07032

Intuition
---------
Cross-entropy treats every pixel equally. A pixel one pixel inside a lane
line and a pixel a hundred meters away from any lane line contribute the
same amount. Boundary loss instead weights each pixel by its SIGNED distance
to the nearest class boundary in the ground truth:

    L_boundary = Σ_classes Σ_pixels  φ(pixel) · softmax(pixel, class)

where φ(pixel) is positive inside the ground-truth region of that class and
negative outside (or vice versa). Predicting high probability for class c
far from c's boundary (where |φ| is large) is penalised heavily; predicting
it at the boundary itself (|φ| ~ 0) is nearly free.

For thin structures like lane lines this is a natural match: the loss
cares disproportionately about getting the EDGES right, which is precisely
where CE tends to blur.

Use with a ramp-up schedule — boundary loss alone is unstable early in
training because the distance field is huge. The standard recipe is

    total_loss = (1 - α) * ce_loss + α * boundary_loss
    with α ramping 0 → target linearly over some number of epochs.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Signed distance transform (CPU, via scipy — runs on labels, not logits)
# ---------------------------------------------------------------------------

def _signed_distance_1class(mask: np.ndarray) -> np.ndarray:
    """Signed distance transform of a binary 2-D mask, in pixel units.

    Positive outside the mask, negative inside (matches the convention in the
    Kervadec paper where the integrand is φ·softmax and we want
    high-probability predictions FAR FROM the boundary to be expensive when
    they are in the wrong region).

    Args:
      mask: bool ndarray [H, W].

    Returns:
      float32 ndarray [H, W]. All zeros if the mask is empty or full
      (no boundary exists).
    """
    from scipy.ndimage import distance_transform_edt

    pos = mask.astype(bool)
    neg = ~pos

    if not pos.any() or not neg.any():
        # No boundary — return zeros to avoid NaNs / inf distances.
        return np.zeros(mask.shape, dtype=np.float32)

    dist_out = distance_transform_edt(neg)  # distance to nearest True pixel, for pixels outside
    dist_in  = distance_transform_edt(pos)  # distance to nearest False pixel, for pixels inside
    sdf = dist_out.astype(np.float32) - dist_in.astype(np.float32)
    return sdf


def compute_distance_maps(
    labels: torch.Tensor,
    num_classes: int,
    ignore_label: int = 255,
) -> torch.Tensor:
    """Per-class signed distance maps for a batch of label maps.

    Args:
      labels:       LongTensor [B, H, W] with class IDs; pixels equal to
                    `ignore_label` are excluded from every class's mask.
      num_classes:  int, C.
      ignore_label: int, sentinel for void pixels.

    Returns:
      FloatTensor [B, C, H, W] on the SAME device as `labels`. Each [b, c]
      plane is the signed distance to class c's boundary in frame b.
    """
    lbl_cpu = labels.detach().cpu().numpy()
    B, H, W = lbl_cpu.shape
    sdf = np.zeros((B, num_classes, H, W), dtype=np.float32)

    for b in range(B):
        lb = lbl_cpu[b]
        # Treat ignore pixels as "not this class" for the boundary, which
        # means they contribute no penalty (φ bounded, softmax finite → finite
        # product; the ignore mask below zeros them out entirely).
        for c in range(num_classes):
            mask_c = (lb == c)
            if mask_c.any():
                sdf[b, c] = _signed_distance_1class(mask_c)
            # else: all zeros, fine.

    return torch.from_numpy(sdf).to(labels.device)


# ---------------------------------------------------------------------------
# The loss module
# ---------------------------------------------------------------------------

class BoundaryLoss(nn.Module):
    """Distance-transform boundary loss for multi-class segmentation.

    Args:
      num_classes:     int, total number of classes.
      ignore_label:    int, void / padding value in label tensors.
      class_weights:   optional [C] tensor for per-class reweighting. Set
                       rare-class weights > 1.0 for same reason you would
                       in CE.
      use_softmax:     bool, if True the forward expects raw logits and
                       applies softmax; if False it expects probabilities.
    """
    def __init__(
        self,
        num_classes: int,
        ignore_label: int = 255,
        class_weights: Optional[Sequence[float]] = None,
        use_softmax: bool = True,
    ):
        super().__init__()
        self.num_classes  = num_classes
        self.ignore_label = ignore_label
        self.use_softmax  = use_softmax
        if class_weights is None:
            self.register_buffer("class_weights", torch.ones(num_classes))
        else:
            w = torch.as_tensor(class_weights, dtype=torch.float32)
            assert w.numel() == num_classes, \
                f"class_weights must have length {num_classes}"
            self.register_buffer("class_weights", w)

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        dist_maps: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
          logits:    [B, C, H, W] raw model output (or probs if use_softmax=False).
          labels:    [B, H, W] LongTensor.
          dist_maps: optional [B, C, H, W] precomputed distance maps. If None,
                     they're computed on the fly via compute_distance_maps
                     (moderately expensive; prefer precomputing in the
                     DataLoader for large-scale training).

        Returns:
          Scalar loss (per-pixel mean of valid pixels).
        """
        B, C, H, W = logits.shape
        assert C == self.num_classes, \
            f"logits C={C} but module was built with num_classes={self.num_classes}"

        if dist_maps is None:
            dist_maps = compute_distance_maps(
                labels, self.num_classes, self.ignore_label
            )

        if self.use_softmax:
            probs = F.softmax(logits, dim=1)
        else:
            probs = logits

        # Per-pixel, per-class integrand: φ(p) * prob(c | p).
        # Sum over C via broadcasting with the class weights.
        w = self.class_weights.to(probs.device).view(1, C, 1, 1)
        prod = dist_maps * probs * w                      # [B, C, H, W]

        # Mask out ignore pixels entirely.
        valid = (labels != self.ignore_label).unsqueeze(1)  # [B, 1, H, W]
        prod  = prod * valid

        # Mean over valid pixels, summed over classes.
        denom = valid.sum().clamp(min=1).float() * C
        return prod.sum() / denom


# ---------------------------------------------------------------------------
# Helper: ramp-up scheduler for alpha
# ---------------------------------------------------------------------------

def ramp_alpha(
    step: int,
    warmup_steps: int,
    target: float,
    start: float = 0.0,
) -> float:
    """Linearly ramp alpha from `start` to `target` over `warmup_steps`."""
    if warmup_steps <= 0:
        return target
    if step >= warmup_steps:
        return target
    frac = step / warmup_steps
    return start + (target - start) * frac
