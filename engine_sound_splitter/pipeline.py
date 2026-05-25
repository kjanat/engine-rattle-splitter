"""High-level orchestrators: decode → process → write."""

from pathlib import Path
from typing import TypedDict

import numpy as np

from .audio_io import decode, write_wav
from .filters import complementary_crossover


class SplitStats(TypedDict):
    duration_s: float
    input_rms: float
    engine_rms: float
    rattles_rms: float
    engine_path: Path
    rattles_path: Path


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x**2)))


def split(
    input_path: Path,
    output_dir: Path,
    sample_rate: int,
    crossover_hz: float,
    order: int,
) -> SplitStats:
    """Crossover-split the input into engine.wav (low) + rattles.wav (high)."""
    audio = decode(input_path, sample_rate, channels=2)
    engine, rattles = complementary_crossover(audio, sample_rate, crossover_hz, order)
    output_dir.mkdir(parents=True, exist_ok=True)
    engine_path = output_dir / "engine.wav"
    rattles_path = output_dir / "rattles.wav"
    write_wav(engine_path, sample_rate, engine)
    write_wav(rattles_path, sample_rate, rattles)
    return SplitStats(
        duration_s=audio.shape[1] / sample_rate,
        input_rms=_rms(audio),
        engine_rms=_rms(engine),
        rattles_rms=_rms(rattles),
        engine_path=engine_path,
        rattles_path=rattles_path,
    )
