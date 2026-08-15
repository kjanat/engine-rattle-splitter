"""Video metadata and audio/video timeline alignment."""

import math
import subprocess
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy.signal import correlate, correlation_lags, resample_poly

from .audio_io import Float32Array

type Float64Array = NDArray[np.float64]


@dataclass(frozen=True)
class VideoMetadata:
    path: Path
    average_fps: float
    nominal_fps: float
    time_base_s: float
    duration_s: float | None
    frame_count: int | None
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class CaptureRate:
    fps: float
    provenance: Literal["physical override", "container assumption"]
    warning: str | None


@dataclass(frozen=True)
class TimelineAlignment:
    offset_s: float
    provenance: Literal["explicit", "audio correlation"]
    correlation: float | None


def probe_video(path: Path) -> VideoMetadata:
    """Read first-video-stream timing without decoding video frames."""
    probe_path = path.resolve()
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=avg_frame_rate,r_frame_rate,time_base,duration,nb_frames",
        "-of",
        "default=noprint_wrappers=1",
        str(probe_path),
    ]
    output = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    ).stdout
    values = _parse_key_values(output)
    average_fps = _positive_fraction(values.get("avg_frame_rate"), "avg_frame_rate")
    nominal_fps = _positive_fraction(values.get("r_frame_rate"), "r_frame_rate")
    time_base_s = _positive_fraction(values.get("time_base"), "time_base")
    duration_s = _optional_positive_float(values.get("duration"))
    frame_count = _optional_positive_int(values.get("nb_frames"))
    warnings: list[str] = [
        "container frame rate does not prove physical capture frame rate"
    ]
    if not math.isclose(average_fps, nominal_fps, rel_tol=1e-4):
        warnings.append(
            "average and nominal frame rates differ; variable timing or metadata ambiguity is possible"
        )
    if average_fps < 60.0:
        warnings.append(
            "slow-motion containers may report playback FPS instead of physical capture FPS"
        )
    return VideoMetadata(
        path=path,
        average_fps=average_fps,
        nominal_fps=nominal_fps,
        time_base_s=time_base_s,
        duration_s=duration_s,
        frame_count=frame_count,
        warnings=tuple(warnings),
    )


def resolve_capture_rate(
    metadata: VideoMetadata | None, physical_override_fps: float | None
) -> CaptureRate | None:
    if physical_override_fps is not None:
        _validate_positive(physical_override_fps, "capture FPS")
        return CaptureRate(
            fps=physical_override_fps,
            provenance="physical override",
            warning=None,
        )
    if metadata is None:
        return None
    return CaptureRate(
        fps=metadata.average_fps,
        provenance="container assumption",
        warning="override with --capture-fps when playback and capture rates differ",
    )


def explicit_alignment(offset_s: float) -> TimelineAlignment:
    if not math.isfinite(offset_s):
        msg = "audio/video offset must be finite"
        raise ValueError(msg)
    return TimelineAlignment(
        offset_s=offset_s,
        provenance="explicit",
        correlation=None,
    )


def align_audio_tracks(
    source_audio: Float32Array,
    video_audio: Float32Array,
    sample_rate: int,
) -> TimelineAlignment:
    """Estimate `video_time = audio_time + offset` from shared audio."""
    target_rate = min(1_000, sample_rate)
    divisor = math.gcd(sample_rate, target_rate)
    source = resample_poly(
        source_audio.astype(np.float64), target_rate // divisor, sample_rate // divisor
    ).astype(np.float64)
    video = resample_poly(
        video_audio.astype(np.float64), target_rate // divisor, sample_rate // divisor
    ).astype(np.float64)
    source = _standardize(source)
    video = _standardize(video)
    correlation = correlate(video, source, mode="full", method="fft")
    lags = correlation_lags(len(video), len(source), mode="full")
    video_energy = correlate(
        np.square(video),
        np.ones(len(source), dtype=np.float64),
        mode="full",
        method="fft",
    )
    source_energy = correlate(
        np.ones(len(video), dtype=np.float64),
        np.square(source),
        mode="full",
        method="fft",
    )
    overlap = correlate(
        np.ones(len(video), dtype=np.float64),
        np.ones(len(source), dtype=np.float64),
        mode="full",
        method="fft",
    )
    denominator = np.sqrt(np.maximum(video_energy * source_energy, 0.0))
    normalized = np.full_like(correlation, -math.inf, dtype=np.float64)
    shorter_track = min(len(source), len(video))
    minimum_overlap = min(
        shorter_track,
        2 * target_rate,
        max(round(0.5 * target_rate), round(0.5 * shorter_track)),
    )
    valid = (overlap >= minimum_overlap - 0.5) & (
        denominator > np.finfo(np.float64).tiny
    )
    normalized[valid] = np.abs(correlation[valid]) / denominator[valid]
    peak_index = int(np.argmax(normalized))
    peak_correlation = min(1.0, float(normalized[peak_index]))
    if not math.isfinite(peak_correlation) or peak_correlation < 0.2:
        msg = "audio tracks have no reliable shared alignment"
        raise ValueError(msg)
    alternatives = normalized.copy()
    exclusion = max(1, round(0.05 * target_rate))
    alternatives[
        max(0, peak_index - exclusion) : min(
            len(alternatives), peak_index + exclusion + 1
        )
    ] = -math.inf
    second_correlation = float(np.max(alternatives))
    if second_correlation >= 0.98 * peak_correlation:
        msg = "audio alignment is ambiguous"
        raise ValueError(msg)
    return TimelineAlignment(
        offset_s=float(lags[peak_index] / target_rate),
        provenance="audio correlation",
        correlation=peak_correlation,
    )


def _parse_key_values(output: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in output.splitlines():
        key, separator, value = line.partition("=")
        if separator and value and value != "N/A":
            values[key] = value
    return values


def _positive_fraction(value: str | None, name: str) -> float:
    if value is None:
        msg = f"video metadata is missing {name}"
        raise ValueError(msg)
    try:
        parsed = float(Fraction(value))
    except (ValueError, ZeroDivisionError) as error:
        msg = f"video metadata has invalid {name}"
        raise ValueError(msg) from error
    _validate_positive(parsed, name)
    return parsed


def _optional_positive_float(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(value)
    except ValueError:
        return None
    return parsed if math.isfinite(parsed) and parsed > 0.0 else None


def _optional_positive_int(value: str | None) -> int | None:
    if value is None:
        return None
    try:
        parsed = int(value)
    except ValueError:
        return None
    return parsed if parsed > 0 else None


def _standardize(samples: Float64Array) -> Float64Array:
    centered = samples - float(np.mean(samples))
    scale = float(np.std(centered))
    if scale <= np.finfo(np.float64).tiny:
        msg = "cannot align silent audio tracks"
        raise ValueError(msg)
    return (centered / scale).astype(np.float64)


def _validate_positive(value: float, name: str) -> None:
    if not math.isfinite(value) or value <= 0.0:
        msg = f"{name} must be finite and positive"
        raise ValueError(msg)
