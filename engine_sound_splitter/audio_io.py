"""ffmpeg-backed audio I/O — handles any codec ffmpeg understands."""

import subprocess
from pathlib import Path

import numpy as np
from scipy.io import wavfile


def decode(path: Path, sr: int, channels: int) -> np.ndarray:
    """Decode audio to float32.

    Returns shape `(channels, n_samples)` for stereo, `(n_samples,)` for mono.
    """
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
        str(channels),
        "-ar",
        str(sr),
        "-",
    ]
    raw = subprocess.run(cmd, check=True, capture_output=True).stdout
    samples = np.frombuffer(raw, dtype=np.float32)
    if channels == 1:
        return samples
    return samples.reshape(-1, channels).T.copy()


def write_wav(path: Path, sr: int, audio: np.ndarray) -> None:
    """Write float32 WAV. Accepts `(n,)` mono or `(channels, n)` multi-channel."""
    data = np.ascontiguousarray(audio.T) if audio.ndim == 2 else audio
    wavfile.write(str(path), sr, data)
