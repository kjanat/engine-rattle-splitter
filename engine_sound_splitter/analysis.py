"""Frame-level feature extraction and before/after segment contrast.

Produces a text report and a 3-panel PNG (time-series features, mean spectrum
comparison, per-band energy ratio) for diagnosing what changes at a given
timestamp in a recording.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import ShortTimeFFT
from scipy.signal.windows import hann

from .audio_io import decode

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


def _frame_rms(samples: np.ndarray, n_fft: int, hop: int) -> np.ndarray:
    n_frames = 1 + (len(samples) - n_fft) // hop
    out = np.empty(n_frames, dtype=np.float32)
    for i in range(n_frames):
        chunk = samples[i * hop : i * hop + n_fft]
        out[i] = np.sqrt(np.mean(chunk**2))
    return out


def _frame_crest(samples: np.ndarray, n_fft: int, hop: int) -> np.ndarray:
    n_frames = 1 + (len(samples) - n_fft) // hop
    out = np.empty(n_frames, dtype=np.float32)
    for i in range(n_frames):
        chunk = samples[i * hop : i * hop + n_fft]
        rms = np.sqrt(np.mean(chunk**2)) + 1e-12
        out[i] = np.max(np.abs(chunk)) / rms
    return out


def _spectral_centroid(mag: np.ndarray, freqs: np.ndarray) -> np.ndarray:
    return (freqs[:, None] * mag).sum(axis=0) / (mag.sum(axis=0) + 1e-12)


def _spectral_flatness(mag: np.ndarray) -> np.ndarray:
    log_mean = np.exp(np.mean(np.log(mag + 1e-12), axis=0))
    return log_mean / (np.mean(mag, axis=0) + 1e-12)


def _spectral_flux(mag: np.ndarray) -> np.ndarray:
    return np.maximum(np.diff(mag, axis=1), 0).sum(axis=0)


def _band_rms(mag: np.ndarray, freqs: np.ndarray, lo: float, hi: float) -> float:
    mask = (freqs >= lo) & (freqs < hi)
    return float(np.sqrt(np.mean(mag[mask] ** 2)))


def _print_stat(name: str, x: np.ndarray, before: slice, after: slice) -> None:
    bm, am = float(np.mean(x[before])), float(np.mean(x[after]))
    delta = (bm - am) / (abs(am) + 1e-12) * 100
    print(f"  {name:22s} before={bm:8.4f}  after={am:8.4f}  Δ={delta:+6.1f}%")


def run(
    input_path: Path,
    sample_rate: int,
    split_at: float,
    output_png: Path,
) -> None:
    """Compute features, print before/after report, write diagnostic PNG."""
    audio = decode(input_path, sample_rate, channels=1)
    win = hann(N_FFT, sym=False).astype(np.float32)
    sft = ShortTimeFFT(win, hop=HOP, fs=sample_rate, mfft=N_FFT, scale_to="magnitude")
    spec = sft.stft(audio)
    mag = np.abs(spec)
    freqs = np.fft.rfftfreq(N_FFT, 1 / sample_rate)

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
    times = np.arange(n) * HOP / sample_rate

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
    header = f"  {'band':10s} {'range (Hz)':14s} {'before':>10s} {'after':>10s} {'ratio':>8s}"
    print(header)
    band_results = []
    for name, lo, hi in BANDS:
        be = _band_rms(mag[:, before], freqs, lo, hi)
        ae = _band_rms(mag[:, after], freqs, lo, hi)
        ratio = be / (ae + 1e-12)
        band_results.append((name, ratio))
        print(f"  {name:10s} {f'{lo}-{hi}':14s} {be:10.4f} {ae:10.4f} {ratio:8.2f}x")

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), dpi=120)

    ax = axes[0]
    ax.plot(times, rms / rms.max(), label="RMS (norm)", color="tab:blue", lw=0.8)
    ax.plot(
        times,
        crest / crest.max(),
        label="crest factor (norm)",
        color="tab:orange",
        lw=0.8,
    )
    ax.plot(
        times[1:],
        flux / flux.max(),
        label="spectral flux (norm)",
        color="tab:red",
        lw=0.8,
    )
    ax.axvline(split_at, color="black", lw=2, ls="--", label=f"{split_at}s mark")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("normalized")
    ax.set_title("frame-level features (each normalized to its own max)")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)

    ax = axes[1]
    mean_before = mag[:, before].mean(axis=1)
    mean_after = mag[:, after].mean(axis=1)
    ax.semilogx(
        freqs[1:],
        20 * np.log10(mean_before[1:] + 1e-8),
        label=f"before {split_at}s",
        color="tab:red",
        lw=1.2,
    )
    ax.semilogx(
        freqs[1:],
        20 * np.log10(mean_after[1:] + 1e-8),
        label=f"after {split_at}s",
        color="tab:blue",
        lw=1.2,
    )
    ax.set_xlim(20, sample_rate / 2)
    ax.set_xlabel("frequency (Hz, log)")
    ax.set_ylabel("magnitude (dB)")
    ax.set_title("mean magnitude spectrum")
    ax.legend()
    ax.grid(alpha=0.3, which="both")

    ax = axes[2]
    names = [b[0] for b in band_results]
    ratios = [b[1] for b in band_results]
    colors = ["tab:red" if r > 1.15 else "tab:gray" for r in ratios]
    ax.bar(names, ratios, color=colors)
    ax.axhline(1.0, color="black", lw=1, ls="--")
    ax.set_ylabel("RMS ratio (before / after)")
    ax.set_title(
        "per-band energy ratio — bars >1.0 (red) = excess energy during rattling"
    )
    ax.grid(alpha=0.3, axis="y")

    fig.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png)
    print(f"\nwrote {output_png}")
