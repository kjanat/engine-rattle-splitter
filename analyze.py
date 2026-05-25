"""Quantify what changes at the 13s mark.

Computes frame-level audio features over the original recording and
contrasts the rattling segment (0-NOISE_PROFILE_START) against the
clean engine segment (NOISE_PROFILE_START-end). Outputs a text summary
of before/after stat deltas and a PNG with three diagnostic panels:

  1. Time series of RMS, spectral flux, and crest factor.
  2. Mean magnitude spectrum before vs after (log scale).
  3. Per-octave-band energy ratio (before / after) — the "rattle spectrum".
"""

import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import ShortTimeFFT
from scipy.signal.windows import hann

INPUT = Path("Shitty motor (goede recording).m4a")
OUTPUT_PNG = Path("analysis.png")
SAMPLE_RATE = 48_000
N_FFT = 4096
HOP = 1024
SPLIT_AT = 13.0  # seconds

BANDS = [
    ("sub-bass", 20, 60),
    ("bass", 60, 250),
    ("low-mid", 250, 500),
    ("mid", 500, 2000),
    ("high-mid", 2000, 4000),
    ("presence", 4000, 8000),
    ("brilliance", 8000, 16000),
    ("air", 16000, 24000),
]


def decode_mono_f32(path: Path, sr: int) -> np.ndarray:
    cmd = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        str(path),
        "-f",
        "f32le",
        "-acodec",
        "pcm_f32le",
        "-ac",
        "1",
        "-ar",
        str(sr),
        "-",
    ]
    raw = subprocess.run(cmd, check=True, capture_output=True).stdout
    return np.frombuffer(raw, dtype=np.float32)


def frame_rms(samples: np.ndarray, n_fft: int, hop: int) -> np.ndarray:
    """RMS per STFT-aligned frame."""
    n_frames = 1 + (len(samples) - n_fft) // hop
    out = np.empty(n_frames, dtype=np.float32)
    for i in range(n_frames):
        chunk = samples[i * hop : i * hop + n_fft]
        out[i] = np.sqrt(np.mean(chunk**2))
    return out


def frame_crest(samples: np.ndarray, n_fft: int, hop: int) -> np.ndarray:
    """Peak / RMS per frame — high values indicate transients."""
    n_frames = 1 + (len(samples) - n_fft) // hop
    out = np.empty(n_frames, dtype=np.float32)
    for i in range(n_frames):
        chunk = samples[i * hop : i * hop + n_fft]
        rms = np.sqrt(np.mean(chunk**2)) + 1e-12
        out[i] = np.max(np.abs(chunk)) / rms
    return out


def spectral_centroid(mag: np.ndarray, freqs: np.ndarray) -> np.ndarray:
    return (freqs[:, None] * mag).sum(axis=0) / (mag.sum(axis=0) + 1e-12)


def spectral_flatness(mag: np.ndarray) -> np.ndarray:
    """Geometric / arithmetic mean. 0=tonal, 1=white noise."""
    log_mean = np.exp(np.mean(np.log(mag + 1e-12), axis=0))
    arith_mean = np.mean(mag, axis=0) + 1e-12
    return log_mean / arith_mean


def spectral_flux(mag: np.ndarray) -> np.ndarray:
    """Sum of positive frame-to-frame magnitude diffs. High = changing fast."""
    diff = np.diff(mag, axis=1)
    return np.maximum(diff, 0).sum(axis=0)


def band_energy(mag: np.ndarray, freqs: np.ndarray, lo: float, hi: float) -> float:
    mask = (freqs >= lo) & (freqs < hi)
    return float(np.sqrt(np.mean(mag[mask] ** 2)))


def main() -> int:
    if not INPUT.exists():
        print(f"missing: {INPUT}", file=sys.stderr)
        return 1

    audio = decode_mono_f32(INPUT, SAMPLE_RATE)
    win = hann(N_FFT, sym=False).astype(np.float32)
    sft = ShortTimeFFT(win, hop=HOP, fs=SAMPLE_RATE, mfft=N_FFT, scale_to="magnitude")
    spec = sft.stft(audio)
    mag = np.abs(spec)
    freqs = np.fft.rfftfreq(N_FFT, 1 / SAMPLE_RATE)

    rms = frame_rms(audio, N_FFT, HOP)
    crest = frame_crest(audio, N_FFT, HOP)
    centroid = spectral_centroid(mag, freqs)
    flatness = spectral_flatness(mag)
    flux = spectral_flux(mag)

    # ShortTimeFFT zero-pads at the edges; unpadded frames are fewer. Align.
    n_frames = min(len(rms), mag.shape[1])
    rms = rms[:n_frames]
    crest = crest[:n_frames]
    mag = mag[:, :n_frames]
    centroid = centroid[:n_frames]
    flatness = flatness[:n_frames]
    flux = flux[: n_frames - 1]
    times = np.arange(n_frames) * HOP / SAMPLE_RATE
    flux_times = times[1:]

    split_frame = int(SPLIT_AT * SAMPLE_RATE / HOP)
    before = slice(0, split_frame)
    after = slice(split_frame, None)

    def stat(name: str, x: np.ndarray, b: slice, a: slice) -> None:
        bm, am = float(np.mean(x[b])), float(np.mean(x[a]))
        delta_pct = (bm - am) / (abs(am) + 1e-12) * 100
        print(f"  {name:22s} before={bm:8.4f}  after={am:8.4f}  Δ={delta_pct:+6.1f}%")

    print(f"Comparison: 0–{SPLIT_AT}s (rattling) vs {SPLIT_AT}s–end (clean engine)\n")
    print("Frame-level features:")
    stat("RMS", rms, before, after)
    stat("crest factor", crest, before, after)
    stat("spectral centroid (Hz)", centroid, before, after)
    stat("spectral flatness", flatness, before, after)
    flux_split = int(SPLIT_AT * SAMPLE_RATE / HOP) - 1
    stat("spectral flux", flux, slice(0, flux_split), slice(flux_split, None))

    print("\nPer-band RMS (averaged magnitude across frames):")
    print(
        f"  {'band':10s} {'range (Hz)':14s} {'before':>10s} {'after':>10s} {'ratio':>8s}"
    )
    band_ratios = []
    for name, lo, hi in BANDS:
        be = band_energy(mag[:, before], freqs, lo, hi)
        ae = band_energy(mag[:, after], freqs, lo, hi)
        ratio = be / (ae + 1e-12)
        band_ratios.append((name, lo, hi, be, ae, ratio))
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
        flux_times,
        flux / flux.max(),
        label="spectral flux (norm)",
        color="tab:red",
        lw=0.8,
    )
    ax.axvline(SPLIT_AT, color="black", lw=2, ls="--", label=f"{SPLIT_AT}s mark")
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
        label="before 13s",
        color="tab:red",
        lw=1.2,
    )
    ax.semilogx(
        freqs[1:],
        20 * np.log10(mean_after[1:] + 1e-8),
        label="after 13s",
        color="tab:blue",
        lw=1.2,
    )
    ax.set_xlim(20, SAMPLE_RATE / 2)
    ax.set_xlabel("frequency (Hz, log)")
    ax.set_ylabel("magnitude (dB)")
    ax.set_title("mean magnitude spectrum")
    ax.legend()
    ax.grid(alpha=0.3, which="both")

    ax = axes[2]
    names = [b[0] for b in band_ratios]
    ratios = [b[5] for b in band_ratios]
    colors = ["tab:red" if r > 1.15 else "tab:gray" for r in ratios]
    ax.bar(names, ratios, color=colors)
    ax.axhline(1.0, color="black", lw=1, ls="--")
    ax.set_ylabel("RMS ratio (before / after)")
    ax.set_title(
        "per-band energy ratio — bars >1.0 (red) = excess energy during rattling"
    )
    ax.grid(alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(OUTPUT_PNG)
    print(f"\nwrote {OUTPUT_PNG}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
