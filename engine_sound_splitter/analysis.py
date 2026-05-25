"""Frame-level feature extraction and before/after segment contrast.

Produces a text report and a 3-panel PNG (time-series features, mean spectrum
comparison, per-band energy ratio) for diagnosing what changes at a given
timestamp in a recording.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import NDArray
from scipy.signal import ShortTimeFFT
from scipy.signal.windows import hann

from .audio_io import Float32Array, decode

type Float64Array = NDArray[np.float64]

N_FFT = 4096
HOP = 1024

BANDS: tuple[tuple[str, int, int], ...] = (
    ("sub-bass", 20, 60),
    ("bass", 60, 250),
    ("low-mid", 250, 500),
    ("mid", 500, 2000),
    ("high-mid", 2000, 4000),
    ("presence", 4000, 8000),
    ("brilliance", 8000, 16000),
    ("air", 16000, 24000),
)


def _frame_rms(samples: Float32Array, n_fft: int, hop: int) -> Float32Array:
    n_frames = 1 + (len(samples) - n_fft) // hop
    out: Float32Array = np.empty(n_frames, dtype=np.float32)
    for i in range(n_frames):
        chunk: Float32Array = samples[i * hop : i * hop + n_fft]
        sq = chunk**2
        out[i] = float(np.sqrt(float(np.mean(sq))))
    return out


def _frame_crest(samples: Float32Array, n_fft: int, hop: int) -> Float32Array:
    n_frames = 1 + (len(samples) - n_fft) // hop
    out: Float32Array = np.empty(n_frames, dtype=np.float32)
    for i in range(n_frames):
        chunk: Float32Array = samples[i * hop : i * hop + n_fft]
        sq = chunk**2
        rms = float(np.sqrt(float(np.mean(sq)))) + 1e-12
        peak = float(np.max(np.abs(chunk)))
        out[i] = peak / rms
    return out


def _spectral_centroid(mag: Float64Array, freqs: Float64Array) -> Float64Array:
    weighted = freqs[:, None] * mag
    return weighted.sum(axis=0) / (mag.sum(axis=0) + 1e-12)


def _spectral_flatness(mag: Float64Array) -> Float64Array:
    log_input = np.log(mag + 1e-12)
    log_mean = np.exp(np.mean(log_input, axis=0))
    arith_mean = np.mean(mag, axis=0)
    return log_mean / (arith_mean + 1e-12)


def _spectral_flux(mag: Float64Array) -> Float64Array:
    diff = np.diff(mag, axis=1)
    positive = np.maximum(diff, 0)
    return positive.sum(axis=0)


def _band_rms(mag: Float64Array, freqs: Float64Array, lo: float, hi: float) -> float:
    mask: NDArray[np.bool_] = (freqs >= lo) & (freqs < hi)
    band = mag[mask]
    sq = band**2
    return float(np.sqrt(float(np.mean(sq))))


def _print_stat(
    name: str,
    x: NDArray[np.floating],
    before: slice,
    after: slice,
) -> None:
    bm = float(np.mean(x[before]))
    am = float(np.mean(x[after]))
    delta = (bm - am) / (abs(am) + 1e-12) * 100
    print(f"  {name:22s} before={bm:8.4f}  after={am:8.4f}  Δ={delta:+6.1f}%")


def run(
    input_path: Path,
    sample_rate: int,
    split_at: float,
    output_png: Path,
) -> None:
    """Compute features, print before/after report, write diagnostic PNG."""
    audio: Float32Array = decode(input_path, sample_rate, channels=1)
    win: Float32Array = hann(N_FFT, sym=False).astype(np.float32)
    sft = ShortTimeFFT(win, hop=HOP, fs=sample_rate, mfft=N_FFT, scale_to="magnitude")
    spec = sft.stft(audio)
    mag: Float64Array = np.abs(spec).astype(np.float64)
    freqs: Float64Array = np.fft.rfftfreq(N_FFT, 1 / sample_rate).astype(np.float64)

    rms = _frame_rms(audio, N_FFT, HOP)
    crest = _frame_crest(audio, N_FFT, HOP)
    centroid = _spectral_centroid(mag, freqs)
    flatness = _spectral_flatness(mag)
    flux = _spectral_flux(mag)

    # ShortTimeFFT zero-pads at the edges (more frames than unpadded framing). Align.
    n = min(len(rms), mag.shape[1])
    rms, crest = rms[:n], crest[:n]
    mag = mag[:, :n]
    centroid, flatness = centroid[:n], flatness[:n]
    flux = flux[: n - 1]
    times: Float64Array = np.arange(n) * HOP / sample_rate

    split_frame = int(split_at * sample_rate / HOP)
    before, after = slice(0, split_frame), slice(split_frame, None)

    print(f"Comparison: 0–{split_at}s vs {split_at}s–end\n")
    print("Frame-level features:")
    _print_stat("RMS", rms, before, after)
    _print_stat("crest factor", crest, before, after)
    _print_stat("spectral centroid (Hz)", centroid, before, after)
    _print_stat("spectral flatness", flatness, before, after)
    _print_stat(
        "spectral flux", flux, slice(0, split_frame - 1), slice(split_frame - 1, None)
    )

    print("\nPer-band RMS (averaged magnitude across frames):")
    print(
        f"  {'band':10s} {'range (Hz)':14s} {'before':>10s} {'after':>10s} {'ratio':>8s}"
    )
    band_results: list[tuple[str, float]] = []
    for name, lo, hi in BANDS:
        be: float = _band_rms(mag[:, before], freqs, lo, hi)
        ae: float = _band_rms(mag[:, after], freqs, lo, hi)
        ratio: float = be / (ae + 1e-12)
        band_results.append((name, ratio))
        print(f"  {name:10s} {f'{lo}-{hi}':14s} {be:10.4f} {ae:10.4f} {ratio:8.2f}x")

    fig: Figure
    axes_arr: NDArray[np.object_]
    fig, axes_arr = plt.subplots(3, 1, figsize=(14, 10), dpi=120, squeeze=False)
    axes: list[Axes] = [axes_arr[i, 0] for i in range(3)]

    ax = axes[0]
    rms_max: float = float(rms.max())
    crest_max: float = float(crest.max())
    flux_max: float = float(flux.max())
    _ = ax.plot(times, rms / rms_max, label="RMS (norm)", color="tab:blue", lw=0.8)
    _ = ax.plot(
        times,
        crest / crest_max,
        label="crest factor (norm)",
        color="tab:orange",
        lw=0.8,
    )
    _ = ax.plot(
        times[1:],
        flux / flux_max,
        label="spectral flux (norm)",
        color="tab:red",
        lw=0.8,
    )
    _ = ax.axvline(split_at, color="black", lw=2, ls="--", label=f"{split_at}s mark")
    _ = ax.set_xlabel("time (s)")
    _ = ax.set_ylabel("normalized")
    _ = ax.set_title("frame-level features (each normalized to its own max)")
    _ = ax.legend(loc="upper right")
    ax.grid(alpha=0.3)

    ax = axes[1]
    mean_before: Float64Array = mag[:, before].mean(axis=1)
    mean_after: Float64Array = mag[:, after].mean(axis=1)
    _ = ax.semilogx(
        freqs[1:],
        20 * np.log10(mean_before[1:] + 1e-8),
        label=f"before {split_at}s",
        color="tab:red",
        lw=1.2,
    )
    _ = ax.semilogx(
        freqs[1:],
        20 * np.log10(mean_after[1:] + 1e-8),
        label=f"after {split_at}s",
        color="tab:blue",
        lw=1.2,
    )
    _ = ax.set_xlim(20, sample_rate / 2)
    _ = ax.set_xlabel("frequency (Hz, log)")
    _ = ax.set_ylabel("magnitude (dB)")
    _ = ax.set_title("mean magnitude spectrum")
    _ = ax.legend()
    ax.grid(alpha=0.3, which="both")

    ax = axes[2]
    names: list[str] = [b[0] for b in band_results]
    ratios: list[float] = [b[1] for b in band_results]
    colors: list[str] = ["tab:red" if r > 1.15 else "tab:gray" for r in ratios]
    _ = ax.bar(names, ratios, color=colors)
    _ = ax.axhline(1.0, color="black", lw=1, ls="--")
    _ = ax.set_ylabel("RMS ratio (before / after)")
    _ = ax.set_title(
        "per-band energy ratio — bars >1.0 (red) = excess energy during rattling"
    )
    ax.grid(alpha=0.3, axis="y")

    fig.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png)
    print(f"\nwrote {output_png}")
