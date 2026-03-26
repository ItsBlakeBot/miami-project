"""Feature Availability Masking for Weather Brain v3.1.

Handles missing weather features with learned defaults instead of
zero-fill. This is critical because weather data sources (HRRR, RTMA,
ASOS, buoy, GFS, ECMWF) have different availability windows and
failure modes.

Zero-fill is catastrophic for temperature prediction because 0 degF
is a valid (and very cold) temperature. Learned defaults provide a
smooth, differentiable fallback that the model can optimize to
minimize prediction error under missing data.

Usage:
    mask_module = FeatureMasking(n_features=64)
    # mask: 1 where available, 0 where missing
    masked = mask_module(features, mask)
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn
from torch import Tensor

log = logging.getLogger(__name__)


class FeatureMasking(nn.Module):
    """Handles missing features with learned defaults (not zero-fill).

    For each feature dimension, learns a default value that is used when
    the feature is unavailable. During training, the model learns defaults
    that minimize prediction error under the actual missing-data patterns
    seen in production.

    Parameters
    ----------
    n_features : int
        Number of input features to mask.
    init_scale : float
        Scale of initial default values (small random).
    """

    def __init__(self, n_features: int = 64, init_scale: float = 0.01) -> None:
        super().__init__()
        self.n_features = n_features
        self.defaults = nn.Parameter(torch.randn(n_features) * init_scale)

        # Optional: learned per-feature confidence scaling
        # When a feature is filled with a default, downstream layers
        # may want to know it's less reliable
        self.confidence_scale = nn.Parameter(torch.ones(n_features) * 2.0)  # sigmoid(2.0) ≈ 0.88

        n_params = sum(p.numel() for p in self.parameters())
        log.info(f"FeatureMasking: {n_params:,} parameters, "
                 f"n_features={n_features}")

    def forward(self, features: Tensor, mask: Tensor) -> Tensor:
        """Apply feature masking with learned defaults.

        Parameters
        ----------
        features : Tensor
            Shape (..., n_features) — raw feature values.
        mask : Tensor
            Shape (..., n_features) — 1.0 where available, 0.0 where missing.
            Should be broadcastable with features.

        Returns
        -------
        Tensor
            Shape (..., n_features) — features with missing values replaced
            by learned defaults.
        """
        # Replace missing features with learned defaults
        filled = features * mask + self.defaults * (1.0 - mask)

        # Scale by confidence (available features get full weight,
        # default-filled features get a learned scale)
        # sigmoid(2.0) ≈ 0.88, so default-filled features start with meaningful weight
        confidence = mask + torch.sigmoid(self.confidence_scale) * (1.0 - mask)
        return filled * confidence

    def get_defaults(self) -> Tensor:
        """Return current learned default values (detached)."""
        return self.defaults.detach().clone()
