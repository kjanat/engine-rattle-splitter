"""Complementary Butterworth crossover for engine/rattle band separation."""

import numpy as np
from scipy.signal import butter, sosfiltfilt

from .audio_io import Float32Array


def complementary_crossover(
    samples: Float32Array,
    sr: int,
    crossover_hz: float,
    order: int = 4,
) -> tuple[Float32Array, Float32Array]:
    """Split a signal into (low_band, high_band) with exact reconstruction.

    Zero-phase forward-backward Butterworth low-pass; the high band is computed
    by subtraction so `low + high == samples` holds bit-for-bit. Works on mono
    `(n,)` or multi-channel `(channels, n)` arrays — `sosfiltfilt` operates
    along the last axis, so 2-D input is filtered per-channel without a loop.
    """
    sos = butter(order, crossover_hz, btype="low", fs=sr, output="sos")
    low: Float32Array = sosfiltfilt(sos, samples, axis=-1).astype(np.float32)
    high: Float32Array = (samples - low).astype(np.float32)
    return low, high
