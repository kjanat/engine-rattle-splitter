"""Render a diagnostic spectrogram of the input recording.

Pipes ffmpeg-decoded PCM into numpy, then plots a dB-scaled,
log-frequency STFT. Used to eyeball harmonic vs percussive content
before picking a separation strategy.
"""

import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

INPUT = Path("Shitty motor (goede recording).m4a")
OUTPUT = Path("spectrogram.png")
SAMPLE_RATE = 48_000
N_FFT = 4096
HOP = 1024


def decode_to_mono_f32(path: Path, sr: int) -> np.ndarray:
    """Decode any audio file ffmpeg understands into a mono float32 array."""
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


def stft_db(samples: np.ndarray, n_fft: int, hop: int) -> np.ndarray:
    """Return a dB-scaled magnitude spectrogram (freq_bins, frames)."""
    window = np.hanning(n_fft).astype(np.float32)
    n_frames = 1 + (len(samples) - n_fft) // hop
    frames = np.lib.stride_tricks.as_strided(
        samples,
        shape=(n_frames, n_fft),
        strides=(samples.strides[0] * hop, samples.strides[0]),
        writeable=False,
    )
    spec = np.fft.rfft(frames * window, axis=1).T
    mag = np.abs(spec)
    return 20.0 * np.log10(np.maximum(mag, 1e-8))


def main() -> int:
    if not INPUT.exists():
        print(f"missing: {INPUT}", file=sys.stderr)
        return 1

    audio = decode_to_mono_f32(INPUT, SAMPLE_RATE)
    duration = len(audio) / SAMPLE_RATE
    db = stft_db(audio, N_FFT, HOP)

    floor = np.percentile(db, 5)
    ceiling = np.percentile(db, 99)

    fig, ax = plt.subplots(figsize=(14, 6), dpi=120)
    extent = (0.0, duration, 0.0, SAMPLE_RATE / 2)
    ax.imshow(
        db,
        origin="lower",
        aspect="auto",
        extent=extent,
        cmap="magma",
        vmin=floor,
        vmax=ceiling,
    )
    ax.set_yscale("symlog", linthresh=100)
    ax.set_ylim(20, SAMPLE_RATE / 2)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("frequency (Hz, log)")
    ax.set_title(f"{INPUT.name} — {duration:.1f}s, {SAMPLE_RATE} Hz mono")
    fig.tight_layout()
    fig.savefig(OUTPUT)
    print(f"wrote {OUTPUT} ({db.shape[1]} frames, {db.shape[0]} bins)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
