"""ffmpeg-backed audio I/O — handles any codec ffmpeg understands."""

import subprocess
from pathlib import Path
from typing import Literal, overload

import numpy as np
from numpy.typing import NDArray
from scipy.io import wavfile

type Float32Array = NDArray[np.float32]


@overload
def decode(path: Path, sr: int, channels: Literal[1]) -> Float32Array: ...
@overload
def decode(path: Path, sr: int, channels: int) -> Float32Array: ...
def decode(path: Path, sr: int, channels: int) -> Float32Array:
    """Decode audio to float32.

    Returns shape `(channels, n_samples)` for stereo, `(n_samples,)` for mono.
    """
    cmd: list[str] = [
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
    raw: bytes = subprocess.run(cmd, check=True, capture_output=True).stdout
    samples: Float32Array = np.frombuffer(raw, dtype=np.float32)
    if channels == 1:
        return samples
    return samples.reshape(-1, channels).T.copy()


def write_wav(path: Path, sr: int, audio: Float32Array) -> None:
    """Write float32 WAV. Accepts `(n,)` mono or `(channels, n)` multi-channel."""
    data: Float32Array = np.ascontiguousarray(audio.T) if audio.ndim == 2 else audio
    wavfile.write(str(path), sr, data)


def encode_mp3(wav_path: Path, mp3_path: Path, bitrate: str = "192k") -> None:
    """Encode an existing WAV to MP3 via ffmpeg, overwriting the destination."""
    cmd: list[str] = [
        "ffmpeg",
        "-v",
        "error",
        "-y",
        "-i",
        str(wav_path),
        "-codec:a",
        "libmp3lame",
        "-b:a",
        bitrate,
        str(mp3_path),
    ]
    _ = subprocess.run(cmd, check=True, capture_output=True)
