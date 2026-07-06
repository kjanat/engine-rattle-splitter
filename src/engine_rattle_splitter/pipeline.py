"""High-level orchestrators: decode → process → write."""

from pathlib import Path
from typing import TypedDict

import numpy as np

from .audio_io import Float32Array, decode, encode_mp3, write_wav
from .filters import complementary_crossover


class SplitStats(TypedDict):
    duration_s: float
    input_rms: float
    engine_rms: float
    rattles_rms: float
    engine_wav: Path
    rattles_wav: Path
    engine_mp3: Path
    rattles_mp3: Path


def _rms(x: Float32Array) -> float:
    mean_sq: float = float(np.mean(x**2))
    return float(np.sqrt(mean_sq))


def split(
    input_path: Path,
    output_dir: Path,
    sample_rate: int,
    crossover_hz: float,
    order: int,
) -> SplitStats:
    """Crossover-split the input into engine.wav (low) + rattles.wav (high).

    Also re-encodes the stems to MP3 alongside the WAVs so the lossy copies
    always reflect the current separator output.
    """
    audio: Float32Array = decode(input_path, sample_rate, channels=2)
    engine, rattles = complementary_crossover(audio, sample_rate, crossover_hz, order)
    output_dir.mkdir(parents=True, exist_ok=True)

    engine_wav = output_dir / "engine.wav"
    rattles_wav = output_dir / "rattles.wav"
    engine_mp3 = output_dir / "engine.mp3"
    rattles_mp3 = output_dir / "rattles.mp3"

    write_wav(engine_wav, sample_rate, engine)
    print(f"wrote {engine_wav}")
    write_wav(rattles_wav, sample_rate, rattles)
    print(f"wrote {rattles_wav}")
    encode_mp3(engine_wav, engine_mp3)
    print(f"wrote {engine_mp3}")
    encode_mp3(rattles_wav, rattles_mp3)
    print(f"wrote {rattles_mp3}")

    duration_s: float = int(audio.shape[1]) / sample_rate
    return SplitStats(
        duration_s=duration_s,
        input_rms=_rms(audio),
        engine_rms=_rms(engine),
        rattles_rms=_rms(rattles),
        engine_wav=engine_wav,
        rattles_wav=rattles_wav,
        engine_mp3=engine_mp3,
        rattles_mp3=rattles_mp3,
    )
