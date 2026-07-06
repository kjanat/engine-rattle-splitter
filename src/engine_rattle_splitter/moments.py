"""Focused analysis artifacts for known noisy moments."""

import subprocess
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import NDArray

from .audio_io import Float32Array, decode

type Float64Array = NDArray[np.float64]
type BoolArray = NDArray[np.bool_]

N_FFT = 4096
HOP = 512

BANDS: tuple[tuple[str, int, int], ...] = (
    ("20-60", 20, 60),
    ("60-250", 60, 250),
    ("250-500", 250, 500),
    ("500-1000", 500, 1000),
    ("1000-2000", 1000, 2000),
    ("2000-4000", 2000, 4000),
    ("4000-8000", 4000, 8000),
    ("8000-16000", 8000, 16000),
    ("16000-24000", 16000, 24000),
)


@dataclass(frozen=True)
class FindingMoment:
    input_path: Path
    slug: str
    start_s: float
    end_s: float
    context_start_s: float
    context_end_s: float
    highlight: str


DEFAULT_FINDING = FindingMoment(
    input_path=Path("recordings/Laatste stuk. 30-50 acceleratie en deceleratie.m4a"),
    slug="finding_3m56_3m59",
    start_s=236.0,
    end_s=239.0,
    context_start_s=233.0,
    context_end_s=242.0,
    highlight="3:56-3:59 high-frequency rattle",
)


def build_default(output_dir: Path, sample_rate: int) -> None:
    """Generate reproducible site artifacts for the known 3:56-3:59 rattle."""
    if not DEFAULT_FINDING.input_path.exists():
        print(f"missing finding input: {DEFAULT_FINDING.input_path}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_audio_clips(DEFAULT_FINDING, output_dir)
    _write_plots(DEFAULT_FINDING, output_dir, sample_rate)


def _write_audio_clips(moment: FindingMoment, output_dir: Path) -> None:
    target_duration = moment.end_s - moment.start_s
    context_duration = moment.context_end_s - moment.context_start_s
    target = output_dir / f"{moment.slug}_target.wav"
    highpass = output_dir / f"{moment.slug}_target_highpass_2k.mp3"
    context = output_dir / f"{moment.slug}_context.wav"

    _ffmpeg_extract(moment.input_path, moment.start_s, target_duration, target)
    _ffmpeg_extract(
        moment.input_path,
        moment.start_s,
        target_duration,
        highpass,
        audio_filter="highpass=f=2000,volume=8dB",
    )
    _ffmpeg_extract(
        moment.input_path, moment.context_start_s, context_duration, context
    )


def _ffmpeg_extract(
    input_path: Path,
    start_s: float,
    duration_s: float,
    output_path: Path,
    *,
    audio_filter: str | None = None,
) -> None:
    cmd = [
        "ffmpeg",
        "-v",
        "error",
        "-y",
        "-hwaccel",
        "auto",
        "-ss",
        str(start_s),
        "-t",
        str(duration_s),
        "-i",
        str(input_path),
    ]
    if audio_filter is not None:
        cmd.extend(["-af", audio_filter])
    cmd.append(str(output_path))
    _ = subprocess.run(cmd, check=True, capture_output=True)
    print(f"wrote {output_path}")


def _write_plots(moment: FindingMoment, output_dir: Path, sample_rate: int) -> None:
    audio = decode(moment.input_path, sample_rate, channels=1)
    context_audio = _slice_audio(
        audio,
        sample_rate,
        start_s=moment.context_start_s,
        end_s=moment.context_end_s,
    )
    freqs, times, power = _stft_power(context_audio, sample_rate)
    absolute_times = times + moment.context_start_s

    target = _time_mask(absolute_times, moment.start_s, moment.end_s)
    neighbor = _time_mask(
        absolute_times, moment.context_start_s, moment.start_s
    ) | _time_mask(absolute_times, moment.end_s, moment.context_end_s)

    _plot_context_spectrogram(
        freqs=freqs,
        absolute_times=absolute_times,
        power=power,
        moment=moment,
        output_png=output_dir / f"{moment.slug}_context_spectrogram.png",
    )
    _plot_band_delta(
        freqs=freqs,
        power=power,
        target=target,
        neighbor=neighbor,
        moment=moment,
        output_png=output_dir / f"{moment.slug}_delta_bands.png",
    )


def _slice_audio(
    audio: Float32Array,
    sample_rate: int,
    *,
    start_s: float,
    end_s: float,
) -> Float32Array:
    start = max(0, int(start_s * sample_rate))
    end = min(len(audio), int(end_s * sample_rate))
    return audio[start:end]


def _stft_power(
    samples: Float32Array, sample_rate: int
) -> tuple[Float64Array, Float64Array, Float64Array]:
    window: Float64Array = np.hanning(N_FFT).astype(np.float64)
    n_frames = 1 + (len(samples) - N_FFT) // HOP
    frames = np.lib.stride_tricks.as_strided(
        samples,
        shape=(n_frames, N_FFT),
        strides=(samples.strides[0] * HOP, samples.strides[0]),
        writeable=False,
    )
    spec = np.fft.rfft(frames * window, axis=1).T
    freqs: Float64Array = np.fft.rfftfreq(N_FFT, 1 / sample_rate).astype(np.float64)
    times: Float64Array = (np.arange(n_frames, dtype=np.float64) * HOP) / sample_rate
    power: Float64Array = np.square(np.abs(spec)).astype(np.float64)
    return freqs, times, power


def _time_mask(times: Float64Array, start_s: float, end_s: float) -> BoolArray:
    return (times >= start_s) & (times < end_s)


def _band_mask(freqs: Float64Array, lo: float, hi: float) -> BoolArray:
    return (freqs >= lo) & (freqs < hi)


def _band_power_db(
    freqs: Float64Array,
    power: Float64Array,
    frame_mask: BoolArray,
    lo: float,
    hi: float,
) -> float:
    band = _band_mask(freqs, lo, hi)
    mean_power = float(np.mean(power[band][:, frame_mask]))
    return 10.0 * np.log10(mean_power + 1e-24)


def _plot_context_spectrogram(
    *,
    freqs: Float64Array,
    absolute_times: Float64Array,
    power: Float64Array,
    moment: FindingMoment,
    output_png: Path,
) -> None:
    db = 10.0 * np.log10(power + 1e-24)
    floor = float(np.percentile(db, 12))
    ceiling = float(np.percentile(db, 99.5))

    fig: Figure
    ax: Axes
    fig, ax = plt.subplots(figsize=(14, 6), dpi=140)
    image = ax.pcolormesh(
        absolute_times,
        freqs,
        db,
        shading="auto",
        cmap="magma",
        vmin=floor,
        vmax=ceiling,
    )
    _ = ax.axvspan(
        moment.start_s,
        moment.end_s,
        color="cyan",
        alpha=0.18,
        label="target 3:56-3:59",
    )
    for x_pos in (
        moment.context_start_s,
        moment.start_s,
        moment.end_s,
        moment.context_end_s,
    ):
        _ = ax.axvline(x_pos, color="white", lw=0.8, alpha=0.7)
    _ = ax.set_ylim(0, 12_000)
    _ = ax.set_xlabel("time (s)")
    _ = ax.set_ylabel("Hz")
    _ = ax.set_title(moment.highlight)
    _ = ax.legend(loc="upper right")
    _ = fig.colorbar(image, ax=ax, label="dB")
    fig.tight_layout()
    fig.savefig(output_png)
    plt.close(fig)
    print(f"wrote {output_png}")


def _plot_band_delta(
    *,
    freqs: Float64Array,
    power: Float64Array,
    target: BoolArray,
    neighbor: BoolArray,
    moment: FindingMoment,
    output_png: Path,
) -> None:
    labels = [name for name, _, _ in BANDS]
    deltas = [
        _band_power_db(freqs, power, target, lo, hi)
        - _band_power_db(freqs, power, neighbor, lo, hi)
        for _, lo, hi in BANDS
    ]
    colors = [
        "tab:red" if delta > 1.0 else "tab:blue" if delta < -1.0 else "tab:gray"
        for delta in deltas
    ]

    fig: Figure
    ax: Axes
    fig, ax = plt.subplots(figsize=(10, 4), dpi=140)
    _ = ax.bar(labels, deltas, color=colors)
    _ = ax.axhline(0.0, color="black", lw=1)
    _ = ax.set_ylabel("target - surrounding context (dB)")
    _ = ax.set_xlabel("frequency band (Hz)")
    _ = ax.set_title(moment.highlight)
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    fig.savefig(output_png)
    plt.close(fig)
    print(f"wrote {output_png}")
