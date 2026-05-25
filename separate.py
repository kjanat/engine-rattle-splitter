"""Split engine recording into engine + rattles via spectral subtraction.

Uses the user-provided ground truth that rattling stops at NOISE_PROFILE_START
seconds. The post-NOISE_PROFILE_START segment is a clean engine sample; its
median magnitude spectrum estimates the steady-state engine profile. For
every frame in the full recording we compute:

    rattle_mag = max(mag - ALPHA * noise_profile, BETA * mag)
    engine_mag = mag - rattle_mag

These magnitudes induce soft masks in [0,1] that sum to 1 per bin, so
engine + rattles reconstructs the original signal exactly. The BETA floor
prevents spectral holes that would otherwise produce metallic ("alien-speak")
artifacts on inverse STFT.
"""

import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.signal import ShortTimeFFT
from scipy.signal.windows import hann

INPUT = Path("Shitty motor (goede recording).m4a")
SAMPLE_RATE = 48_000
N_FFT = 4096
HOP = 1024
NOISE_PROFILE_START = 13.0  # seconds; rattling has stopped by this point
ALPHA = 1.5  # over-subtraction factor (1.0=exact, >1=more aggressive)
BETA = 0.05  # floor: minimum fraction of original kept in rattles stem


def decode_stereo_f32(path: Path, sr: int) -> np.ndarray:
    """Return shape (2, n_samples) float32 array."""
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
        "2",
        "-ar",
        str(sr),
        "-",
    ]
    raw = subprocess.run(cmd, check=True, capture_output=True).stdout
    return np.frombuffer(raw, dtype=np.float32).reshape(-1, 2).T.copy()


def estimate_noise_profile(spec: np.ndarray, t_start: float) -> np.ndarray:
    """Median magnitude spectrum across frames from t_start to end.

    Median (not mean) so a stray transient in the "clean" region can't poison
    the profile. Shape: (freq_bins, 1) for broadcasting.
    """
    frame_start = int(t_start * SAMPLE_RATE / HOP)
    return np.median(np.abs(spec[:, frame_start:]), axis=1, keepdims=True)


def subtract_channel(
    samples: np.ndarray, sft: ShortTimeFFT
) -> tuple[np.ndarray, np.ndarray]:
    spec = sft.stft(samples)
    mag = np.abs(spec)
    noise_profile = estimate_noise_profile(spec, NOISE_PROFILE_START)
    rattle_mag = np.maximum(mag - ALPHA * noise_profile, BETA * mag)
    rattle_mask = (rattle_mag / np.maximum(mag, 1e-12)).astype(np.float32)
    engine_mask = (1.0 - rattle_mask).astype(np.float32)
    n = len(samples)
    return sft.istft(spec * engine_mask, k1=n), sft.istft(spec * rattle_mask, k1=n)


def write_wav_f32(path: Path, sr: int, stereo: np.ndarray) -> None:
    interleaved = np.ascontiguousarray(stereo.T)
    wavfile.write(str(path), sr, interleaved)


def main() -> int:
    if not INPUT.exists():
        print(f"missing: {INPUT}", file=sys.stderr)
        return 1

    audio = decode_stereo_f32(INPUT, SAMPLE_RATE)
    win = hann(N_FFT, sym=False).astype(np.float32)
    sft = ShortTimeFFT(win, hop=HOP, fs=SAMPLE_RATE, mfft=N_FFT, scale_to="magnitude")

    left_e, left_r = subtract_channel(audio[0], sft)
    right_e, right_r = subtract_channel(audio[1], sft)

    engine = np.stack([left_e, right_e])
    rattles = np.stack([left_r, right_r])

    write_wav_f32(Path("engine.wav"), SAMPLE_RATE, engine)
    write_wav_f32(Path("rattles.wav"), SAMPLE_RATE, rattles)

    def energy(x):
        return float(np.sqrt(np.mean(x**2)))

    print(f"engine  RMS: {energy(engine):.4f}")
    print(f"rattles RMS: {energy(rattles):.4f}")
    print(f"input   RMS: {energy(audio):.4f}")

    print("\nrattles RMS per second (should drop sharply after 13s):")
    mono_rattles = rattles.mean(axis=0)
    secs = int(len(mono_rattles) / SAMPLE_RATE)
    for s in range(secs):
        chunk = mono_rattles[s * SAMPLE_RATE : (s + 1) * SAMPLE_RATE]
        marker = " <- rattles stop" if s == 13 else ""
        bar = "█" * int(energy(chunk) * 400)
        print(f"  {s:2d}s: {energy(chunk):.4f}  {bar}{marker}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
