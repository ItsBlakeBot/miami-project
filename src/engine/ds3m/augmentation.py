"""Data augmentation for DS3M pre-training.

Applies stochastic augmentations to feature tensors (T x 33) during training
to improve generalization of the Mamba encoder. Augmentations are calibrated
to the physical scales of Miami weather observations:

  - Temporal jitter: shifts sequences by ±1-3 timesteps to reduce overfitting
    to exact timing of weather transitions.
  - Feature noise: adds Gaussian noise scaled to instrument precision
    (0.5°F temp, 2kt wind, 0.05 cloud fraction).
  - Station dropout: zeros out entire feature groups to force the model
    to learn from partial information (mimics missing station data).
  - Regime splicing: concatenates halves from different samples to simulate
    abrupt regime transitions (frontal passages, sea breeze onset).

Feature vector layout (33 features):
  [0-10]  NWP model guidance
  [11-16] Surface observations (temp, dewpoint, wind dir/speed, cloud, running max)
  [17-22] Atmospheric (CAPE, CIN, pressure, cloud, wind speed/dir)
  [23-28] Temporal (hour sin/cos, DOY sin/cos, lead time, DST) — never augmented
  [29-32] Derived (dewpoint change, wind change)
"""

from __future__ import annotations

import torch
from torch import Tensor


class SequenceAugmentor:
    """Stochastic augmentation pipeline for DS3M training sequences.

    Applied to feature tensors of shape (T, 33) during pre-training.
    All augmentations use torch operations and respect the feature layout:
    temporal features (23-28) are never modified.

    Parameters
    ----------
    enabled : bool
        When False, ``__call__`` and ``splice`` return inputs unchanged.
        Set to False during validation / inference.
    """

    # Feature group index ranges
    NWP_SLICE = slice(0, 11)       # indices 0-10
    OBS_SLICE = slice(11, 17)      # indices 11-16
    ATMOS_SLICE = slice(17, 23)    # indices 17-22
    TEMPORAL_SLICE = slice(23, 29) # indices 23-28  (never touched)
    DERIVED_SLICE = slice(29, 33)  # indices 29-32

    # Observation feature indices within the full 33-vector
    OBS_TEMP_IDX = 11       # obs_temp (°F)
    OBS_DEWPOINT_IDX = 12   # obs_dewpoint (°F)
    OBS_WIND_DIR_IDX = 13   # obs_wind_dir (deg)
    OBS_WIND_SPEED_IDX = 14 # obs_wind_speed (kt)
    OBS_CLOUD_IDX = 15      # obs_cloud_cover (fraction 0-1)
    OBS_RUNMAX_IDX = 16     # obs_temp_max_so_far (°F)

    # Noise standard deviations calibrated to instrument precision
    _NOISE_SCALES = {
        11: 0.5,   # temp: ±0.5°F
        12: 0.5,   # dewpoint: ±0.5°F
        13: 2.0,   # wind dir: ±2° (small relative to 0-360)
        14: 2.0,   # wind speed: ±2kt
        15: 0.05,  # cloud cover: ±0.05 fraction
        16: 0.5,   # running max: ±0.5°F
    }

    # Feature groups eligible for station dropout
    _DROPOUT_GROUPS = [
        slice(0, 11),   # NWP
        slice(11, 17),  # Obs
        slice(17, 23),  # Atmos
    ]

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled

    def __call__(self, features: Tensor) -> Tensor:
        """Apply augmentations 1-3 to a single feature tensor.

        Parameters
        ----------
        features : Tensor
            Shape (T, 33). Modified in-place is avoided; a new tensor is
            returned so the original dataset sample is not corrupted.

        Returns
        -------
        Tensor
            Augmented features, same shape (T, 33).
        """
        if not self.enabled:
            return features

        x = features.clone()

        # 1. Temporal jitter (p=0.5)
        if torch.rand(1).item() < 0.5:
            x = self._temporal_jitter(x)

        # 2. Feature noise on obs indices 11-16 (p=0.7)
        if torch.rand(1).item() < 0.7:
            x = self._feature_noise(x)

        # 3. Station dropout (p=0.15 per group, at most one group)
        if torch.rand(1).item() < 0.15:
            x = self._station_dropout(x)

        return x

    def splice(self, feat1: Tensor, feat2: Tensor) -> Tensor:
        """Regime splicing: first half of feat1 + second half of feat2.

        Simulates abrupt regime transitions (frontal passages, sea breeze
        onset) by concatenating the first half of one sequence with the
        second half of another.

        Parameters
        ----------
        feat1 : Tensor
            Shape (T, 33). Provides the first half.
        feat2 : Tensor
            Shape (T, 33). Provides the second half.

        Returns
        -------
        Tensor
            Spliced features, shape (T, 33) where T = feat1.shape[0].
        """
        if not self.enabled:
            return feat1

        T = feat1.shape[0]
        mid = T // 2
        result = feat1.clone()
        # Replace second half with feat2's second half
        # If feat2 is shorter/longer, take from its midpoint onward and
        # truncate or pad to fill the remaining slots.
        T2 = feat2.shape[0]
        mid2 = T2 // 2
        n_copy = min(T - mid, T2 - mid2)
        result[mid : mid + n_copy] = feat2[mid2 : mid2 + n_copy]
        # If feat2's second half was shorter, pad remaining with its last row
        if n_copy < T - mid:
            result[mid + n_copy :] = feat2[-1].unsqueeze(0)
        return result

    # ── Private helpers ──────────────────────────────────────────────

    def _temporal_jitter(self, x: Tensor) -> Tensor:
        """Shift sequence by ±1-3 timesteps, filling edges with edge values."""
        shift = torch.randint(1, 4, (1,)).item()
        if torch.rand(1).item() < 0.5:
            shift = -shift

        x = torch.roll(x, shifts=int(shift), dims=0)

        # Fill the exposed edge with the nearest valid row
        if shift > 0:
            # Rolled forward: first `shift` rows are wrapped from the end.
            # Replace them with the first valid row (row at index shift).
            x[:shift] = x[shift].unsqueeze(0)
        else:
            # Rolled backward: last `|shift|` rows are wrapped from the start.
            x[shift:] = x[shift - 1].unsqueeze(0)

        return x

    def _feature_noise(self, x: Tensor) -> Tensor:
        """Add calibrated Gaussian noise to observation features (11-16)."""
        T = x.shape[0]
        for idx, sigma in self._NOISE_SCALES.items():
            noise = torch.randn(T, device=x.device, dtype=x.dtype) * sigma
            x[:, idx] = x[:, idx] + noise
        return x

    def _station_dropout(self, x: Tensor) -> Tensor:
        """Zero out one randomly chosen feature group (NWP, Obs, or Atmos)."""
        group_idx = torch.randint(0, len(self._DROPOUT_GROUPS), (1,)).item()
        group_slice = self._DROPOUT_GROUPS[group_idx]
        x[:, group_slice] = 0.0
        return x
