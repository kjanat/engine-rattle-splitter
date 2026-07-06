"""Diagnostic log-frequency dB spectrogram renderer."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import NDArray

from .audio_io import Float32Array, decode

N_FFT = 4096
HOP = 1024


def _stft_db(samples: Float32Array, n_fft: int, hop: int) -> NDArray[np.float64]:
    """dB-scaled magnitude spectrogram, shape (freq_bins, n_frames)."""
    window: Float32Array = np.hanning(n_fft).astype(np.float32)
    n_frames = 1 + (len(samples) - n_fft) // hop
    frames: Float32Array = np.lib.stride_tricks.as_strided(
        samples,
        shape=(n_frames, n_fft),
        strides=(samples.strides[0] * hop, samples.strides[0]),
        writeable=False,
    )
    spec = np.fft.rfft(frames * window, axis=1).T
    return 20.0 * np.log10(np.maximum(np.abs(spec), 1e-8))


def run(input_path: Path, sample_rate: int, output_png: Path) -> None:
    """Render a log-frequency dB spectrogram PNG."""
    audio: Float32Array = decode(input_path, sample_rate, channels=1)
    duration: float = len(audio) / sample_rate
    db = _stft_db(audio, N_FFT, HOP)
    floor: float = float(np.percentile(db, 5))
    ceiling: float = float(np.percentile(db, 99))

    fig: Figure
    ax: Axes
    fig, ax = plt.subplots(figsize=(14, 6), dpi=120)
    _ = ax.imshow(
        db,
        origin="lower",
        aspect="auto",
        extent=(0.0, duration, 0.0, sample_rate / 2),
        cmap="magma",
        vmin=floor,
        vmax=ceiling,
    )
    ax.set_yscale("symlog", linthresh=100)
    _ = ax.set_ylim(20, sample_rate / 2)
    _ = ax.set_xlabel("time (s)")
    _ = ax.set_ylabel("frequency (Hz, log)")
    _ = ax.set_title(f"{input_path.name} — {duration:.1f}s, {sample_rate} Hz mono")
    fig.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png)
    print(f"wrote {output_png} ({db.shape[1]} frames, {db.shape[0]} bins)")
