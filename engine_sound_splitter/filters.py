"""Complementary Butterworth crossover for engine/rattle band separation."""

import numpy as np
from scipy.signal import butter, sosfiltfilt


def complementary_crossover(
    samples: np.ndarray,
    sr: int,
    crossover_hz: float,
    order: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    """Split a signal into (low_band, high_band) with exact reconstruction.

    Zero-phase forward-backward Butterworth low-pass; the high band is computed
    by subtraction so `low + high == samples` holds bit-for-bit. Works on mono
    `(n,)` or multi-channel `(channels, n)` arrays.
    """
    sos = butter(order, crossover_hz, btype="low", fs=sr, output="sos")
    if samples.ndim == 1:
        low = sosfiltfilt(sos, samples).astype(np.float32)
        return low, (samples - low).astype(np.float32)
    low = np.stack([sosfiltfilt(sos, ch) for ch in samples]).astype(np.float32)
    return low, (samples - low).astype(np.float32)
