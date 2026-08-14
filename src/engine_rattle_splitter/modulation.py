"""Rattle-envelope modulation frequency analysis."""

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import NDArray
from scipy.signal import butter, find_peaks, hilbert, resample_poly, sosfiltfilt, welch
from scipy.signal.windows import hann

from .audio_io import Float32Array, decode
from .filters import complementary_crossover

type Float64Array = NDArray[np.float64]
type AudioInputArray = NDArray[np.floating] | NDArray[np.complexfloating]
type VideoObservability = Literal["well sampled", "marginal", "at or above Nyquist"]

DEFAULT_CROSSOVER_HZ = 1800.0
DEFAULT_FILTER_ORDER = 4
ENVELOPE_CUTOFF_HZ = 100.0
ENVELOPE_SAMPLE_RATE = 400
MIN_MODULATION_HZ = 5.0
MAX_MODULATION_HZ = 100.0
WELCH_SEGMENT_S = 8.0
SPECTROGRAM_WINDOW_S = 2.0
SPECTROGRAM_HOP_S = 0.25
PEAK_PROMINENCE_DB = 6.0
PEAK_SEPARATION_HZ = 1.0
MAX_PEAKS = 6
MIN_DURATION_S = 1.0
SPECTRUM_FLOOR_DB = -120.0
SPECTROGRAM_FLOOR_DB = -60.0


@dataclass(frozen=True)
class ModulationPeak:
    """One prominent frequency in the rattle amplitude envelope."""

    frequency_hz: float
    level_db: float
    prominence_db: float


@dataclass(frozen=True)
class ModulationSpectrogram:
    """Time-resolved PSD of the rattle amplitude envelope."""

    times_s: Float64Array
    frequencies_hz: Float64Array
    psd_db: Float64Array
    window_duration_s: float
    hop_duration_s: float
    frequency_resolution_hz: float


@dataclass(frozen=True)
class ModulationResult:
    """Envelope, modulation spectrum, and detected candidate frequencies."""

    times_s: Float64Array
    envelope: Float64Array
    frequencies_hz: Float64Array
    spectrum_db: Float64Array
    peaks: tuple[ModulationPeak, ...]
    frequency_resolution_hz: float
    spectrogram: ModulationSpectrogram


class InsufficientAudioError(ValueError):
    """Raised when a clip is too short for meaningful modulation analysis."""


def analyze(
    samples: AudioInputArray,
    sample_rate: int,
    *,
    crossover_hz: float = DEFAULT_CROSSOVER_HZ,
    filter_order: int = DEFAULT_FILTER_ORDER,
) -> ModulationResult:
    """Measure low-frequency amplitude modulation of the high-band signal."""
    _validate_input(samples, sample_rate, crossover_hz, filter_order)
    with np.errstate(over="ignore", invalid="ignore"):
        real_samples: Float32Array = samples.astype(np.float32, copy=False)
    if not bool(np.all(np.isfinite(real_samples))):
        msg = "audio samples must remain finite after float32 conversion"
        raise ValueError(msg)

    pad_samples = min(sample_rate, len(real_samples) - 1)
    padded: Float32Array = np.pad(real_samples, pad_samples, mode="reflect").astype(
        np.float32, copy=False
    )
    _, rattle = complementary_crossover(
        padded,
        sample_rate,
        crossover_hz,
        filter_order,
    )
    analytic_envelope: Float64Array = np.abs(hilbert(rattle)).astype(np.float64)
    envelope_sos = butter(
        DEFAULT_FILTER_ORDER,
        ENVELOPE_CUTOFF_HZ,
        btype="low",
        fs=sample_rate,
        output="sos",
    )
    smoothed: Float64Array = sosfiltfilt(envelope_sos, analytic_envelope).astype(
        np.float64
    )
    envelope = smoothed[pad_samples:-pad_samples]
    envelope = np.maximum(envelope, 0.0).astype(np.float64)

    envelope = _resample_envelope(envelope, sample_rate)
    envelope = np.maximum(envelope, 0.0).astype(np.float64)
    times: Float64Array = np.arange(len(envelope), dtype=np.float64) / float(
        ENVELOPE_SAMPLE_RATE
    )

    spectrum_frequencies, full_spectrum_db, resolution = _spectrum(envelope)
    spectrogram = _modulation_spectrogram(envelope)
    peaks = _detect_peaks(spectrum_frequencies, full_spectrum_db, resolution)
    output_band = (spectrum_frequencies >= MIN_MODULATION_HZ) & (
        spectrum_frequencies <= MAX_MODULATION_HZ
    )
    frequencies = spectrum_frequencies[output_band].astype(np.float64)
    spectrum_db = full_spectrum_db[output_band].astype(np.float64)
    return ModulationResult(
        times_s=times,
        envelope=envelope,
        frequencies_hz=frequencies,
        spectrum_db=spectrum_db,
        peaks=peaks,
        frequency_resolution_hz=resolution,
        spectrogram=spectrogram,
    )


def _resample_envelope(envelope: Float64Array, sample_rate: int) -> Float64Array:
    divisor = math.gcd(sample_rate, ENVELOPE_SAMPLE_RATE)
    return resample_poly(
        envelope,
        ENVELOPE_SAMPLE_RATE // divisor,
        sample_rate // divisor,
        padtype="line",
    ).astype(np.float64)


def order_ratio(frequency_hz: float, rpm: float) -> float:
    """Express a measured modulation frequency as a fixed shaft-speed order."""
    _validate_rpm(rpm)
    return frequency_hz / (rpm / 60.0)


def video_observability(frequency_hz: float, video_fps: float) -> VideoObservability:
    """Classify a frequency for ideal uniformly sampled video."""
    _validate_video_fps(video_fps)
    if not math.isfinite(frequency_hz) or frequency_hz <= 0.0:
        msg = "frequency must be finite and positive"
        raise ValueError(msg)
    if frequency_hz <= video_fps / 4.0:
        return "well sampled"
    if frequency_hz < video_fps / 2.0:
        return "marginal"
    return "at or above Nyquist"


def run(
    input_path: Path,
    sample_rate: int,
    output_png: Path,
    *,
    rpm: float | None = None,
    video_fps: float | None = None,
    crossover_hz: float = DEFAULT_CROSSOVER_HZ,
    filter_order: int = DEFAULT_FILTER_ORDER,
) -> ModulationResult:
    """Decode, analyze, report, and render one recording."""
    if rpm is not None:
        _validate_rpm(rpm)
    if video_fps is not None:
        _validate_video_fps(video_fps)
    audio = decode(input_path, sample_rate, channels=1)
    result = analyze(
        audio,
        sample_rate,
        crossover_hz=crossover_hz,
        filter_order=filter_order,
    )
    render(
        result,
        input_name=input_path.name,
        output_png=output_png,
        crossover_hz=crossover_hz,
        rpm=rpm,
        video_fps=video_fps,
    )
    print_report(result, rpm=rpm, video_fps=video_fps)
    return result


def render(
    result: ModulationResult,
    *,
    input_name: str,
    output_png: Path,
    crossover_hz: float = DEFAULT_CROSSOVER_HZ,
    rpm: float | None = None,
    video_fps: float | None = None,
) -> None:
    """Render envelope, time-resolved modulation, and global spectrum."""
    if rpm is not None:
        _validate_rpm(rpm)
    if video_fps is not None:
        _validate_video_fps(video_fps)

    fig: Figure
    axes_arr: NDArray[np.object_]
    fig, axes_arr = plt.subplots(3, 1, figsize=(14, 11), dpi=140, squeeze=False)
    axes: list[Axes] = [axes_arr[index, 0] for index in range(3)]

    envelope_ax = axes[0]
    _ = envelope_ax.plot(
        result.times_s,
        result.envelope,
        color="tab:blue",
        lw=0.8,
    )
    _ = envelope_ax.set_xlim(0.0, float(result.times_s[-1]))
    _ = envelope_ax.set_xlabel("time (s)")
    _ = envelope_ax.set_ylabel("amplitude")
    _ = envelope_ax.set_title(
        f"Rattle-band analytic envelope (complementary high band above {crossover_hz:g} Hz)"
    )
    envelope_ax.grid(alpha=0.3)

    spectrogram_ax = axes[1]
    spectrogram = result.spectrogram
    duration_s = len(result.envelope) / ENVELOPE_SAMPLE_RATE
    time_edges = _bin_edges(spectrogram.times_s, 0.0, duration_s)
    frequency_edges = _bin_edges(
        spectrogram.frequencies_hz,
        MIN_MODULATION_HZ,
        MAX_MODULATION_HZ,
    )
    image = spectrogram_ax.pcolormesh(
        time_edges,
        frequency_edges,
        spectrogram.psd_db,
        shading="flat",
        cmap="magma",
        vmin=SPECTROGRAM_FLOOR_DB,
        vmax=0.0,
    )
    _ = spectrogram_ax.set_xlim(0.0, duration_s)
    _ = spectrogram_ax.set_ylim(MIN_MODULATION_HZ, MAX_MODULATION_HZ)
    _ = spectrogram_ax.set_xlabel("time (s)")
    _ = spectrogram_ax.set_ylabel("modulation frequency (Hz)")
    _ = spectrogram_ax.set_title(
        f"Time-resolved envelope PSD ({spectrogram.window_duration_s:g} s window, "
        f"{spectrogram.hop_duration_s:g} s hop, "
        f"{spectrogram.frequency_resolution_hz:g} Hz bins)"
    )
    _ = fig.colorbar(image, ax=spectrogram_ax, label="relative local PSD (dB)")
    if bool(np.all(spectrogram.psd_db == SPECTROGRAM_FLOOR_DB)):
        _ = spectrogram_ax.text(
            0.5,
            0.5,
            "No measurable high-band envelope",
            transform=spectrogram_ax.transAxes,
            color="white",
            ha="center",
            va="center",
        )
    if len(spectrogram.times_s) == 1:
        _ = spectrogram_ax.text(
            0.99,
            0.02,
            "single window; no time localization",
            transform=spectrogram_ax.transAxes,
            color="white",
            ha="right",
            va="bottom",
            fontsize=8,
        )

    spectrum_ax = axes[2]
    _ = spectrum_ax.plot(
        result.frequencies_hz,
        result.spectrum_db,
        color="tab:red",
        lw=1.1,
    )
    _ = spectrum_ax.set_xlim(MIN_MODULATION_HZ, MAX_MODULATION_HZ)
    _ = spectrum_ax.set_ylim(SPECTRUM_FLOOR_DB, 3.0)
    _ = spectrum_ax.set_xlabel("rattle modulation frequency (Hz)")
    _ = spectrum_ax.set_ylabel("relative envelope PSD (dB)")
    title = (
        f"Envelope modulation spectrum (Welch resolution "
        f"{result.frequency_resolution_hz:.2f} Hz)"
    )
    if rpm is not None:
        title += f"; fixed {rpm:g} rpm"
    _ = spectrum_ax.set_title(title)
    spectrum_ax.grid(alpha=0.3)

    if video_fps is not None:
        _mark_video_regions(
            spectrogram_ax=spectrogram_ax,
            spectrum_ax=spectrum_ax,
            video_fps=video_fps,
        )

    for index, peak in enumerate(result.peaks):
        _ = spectrum_ax.axvline(
            peak.frequency_hz,
            color="black",
            lw=0.8,
            ls="--",
            alpha=0.6,
        )
        label = f"{peak.frequency_hz:.1f} Hz"
        if rpm is not None:
            label += f"\n{order_ratio(peak.frequency_hz, rpm):.2f}x"
        _ = spectrum_ax.annotate(
            label,
            xy=(peak.frequency_hz, peak.level_db),
            xytext=(0, -10 - 13 * (index % 2)),
            textcoords="offset points",
            ha="center",
            va="top",
            fontsize=8,
        )

    if not result.peaks:
        _ = spectrum_ax.text(
            0.5,
            0.5,
            "No candidate peaks above the prominence threshold",
            transform=spectrum_ax.transAxes,
            ha="center",
            va="center",
        )

    fig.suptitle(input_name)
    fig.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, format="png")
    plt.close(fig)
    print(f"wrote {output_png}")


def print_report(
    result: ModulationResult,
    *,
    rpm: float | None = None,
    video_fps: float | None = None,
) -> None:
    """Print measured frequencies and optional fixed-speed order ratios."""
    if rpm is not None:
        _validate_rpm(rpm)
    if video_fps is not None:
        _validate_video_fps(video_fps)

    print("\nRattle-envelope modulation peaks:")
    if not result.peaks:
        print("  none above prominence threshold")
    for peak in result.peaks:
        print(f"  {peak.frequency_hz:6.1f} Hz  prominence={peak.prominence_db:5.1f} dB")

    print(f"Frequency resolution: {result.frequency_resolution_hz:.2f} Hz")
    if rpm is not None:
        crank_hz = rpm / 60.0
        print(f"\nEngine speed: {rpm:g} rpm")
        print(f"Crank fundamental: {crank_hz:.1f} Hz")
        print("Orders (fixed-RPM assumption):")
        if not result.peaks:
            print("  none")
        for peak in result.peaks:
            print(
                f"  {peak.frequency_hz:6.1f} Hz  "
                f"{order_ratio(peak.frequency_hz, rpm):5.2f}x"
            )

    if video_fps is not None:
        nyquist_hz = video_fps / 2.0
        print(f"\nVideo: {video_fps:g} fps, Nyquist {nyquist_hz:.2f} Hz")
        print("Video search candidates:")
        if not result.peaks:
            print("  none")
        for peak in result.peaks:
            status = video_observability(peak.frequency_hz, video_fps)
            if status == "at or above Nyquist":
                print(
                    f"  {peak.frequency_hz:6.1f} Hz  {status}; "
                    f"capture >= {4.0 * peak.frequency_hz:.1f} fps"
                )
                continue
            lower_hz, upper_hz = _recommended_motion_band(
                peak.frequency_hz,
                result.frequency_resolution_hz,
                nyquist_hz,
            )
            print(
                f"  {peak.frequency_hz:6.1f} Hz  {status}; "
                f"search {lower_hz:.1f}-{upper_hz:.1f} Hz"
            )


def _mark_video_regions(
    *, spectrogram_ax: Axes, spectrum_ax: Axes, video_fps: float
) -> None:
    four_frame_limit = video_fps / 4.0
    nyquist_hz = video_fps / 2.0
    has_legend_entry = False
    if MIN_MODULATION_HZ < four_frame_limit < MAX_MODULATION_HZ:
        _ = spectrogram_ax.axhline(
            four_frame_limit, color="cyan", lw=0.8, ls=":", alpha=0.8
        )
        _ = spectrum_ax.axvline(
            four_frame_limit,
            color="tab:blue",
            lw=0.8,
            ls=":",
            alpha=0.8,
            label=f"4 frames/cycle ({four_frame_limit:.2f} Hz)",
        )
        has_legend_entry = True
    if nyquist_hz < MAX_MODULATION_HZ:
        shade_start_hz = max(MIN_MODULATION_HZ, nyquist_hz)
        _ = spectrogram_ax.axhspan(
            shade_start_hz, MAX_MODULATION_HZ, color="gray", alpha=0.18
        )
        span_label = (
            f"above video Nyquist ({nyquist_hz:.2f} Hz)"
            if nyquist_hz <= MIN_MODULATION_HZ
            else None
        )
        _ = spectrum_ax.axvspan(
            shade_start_hz,
            MAX_MODULATION_HZ,
            color="gray",
            alpha=0.12,
            label=span_label,
        )
        if MIN_MODULATION_HZ < nyquist_hz:
            _ = spectrogram_ax.axhline(
                nyquist_hz, color="cyan", lw=1.0, ls="--", alpha=0.9
            )
            _ = spectrum_ax.axvline(
                nyquist_hz,
                color="black",
                lw=1.0,
                ls="--",
                alpha=0.8,
                label=f"video Nyquist ({nyquist_hz:.2f} Hz)",
            )
        has_legend_entry = True
    if has_legend_entry:
        _ = spectrum_ax.legend(loc="lower left", fontsize=8)


def _recommended_motion_band(
    frequency_hz: float, resolution_hz: float, nyquist_hz: float
) -> tuple[float, float]:
    half_width_hz = max(2.0, 0.05 * frequency_hz, 2.0 * resolution_hz)
    return (
        max(MIN_MODULATION_HZ, frequency_hz - half_width_hz),
        min(nyquist_hz, frequency_hz + half_width_hz),
    )


def _bin_edges(
    centers: Float64Array, single_start: float, single_end: float
) -> Float64Array:
    if len(centers) == 1:
        return np.array([single_start, single_end], dtype=np.float64)
    midpoints = (centers[:-1] + centers[1:]) / 2.0
    first = float(centers[0] - (midpoints[0] - centers[0]))
    last = float(centers[-1] + (centers[-1] - midpoints[-1]))
    return np.concatenate((np.array([first]), midpoints, np.array([last]))).astype(
        np.float64
    )


def _spectrum(
    envelope: Float64Array,
) -> tuple[Float64Array, Float64Array, float]:
    nperseg = min(len(envelope), round(WELCH_SEGMENT_S * ENVELOPE_SAMPLE_RATE))
    frequencies, power = welch(
        envelope,
        fs=ENVELOPE_SAMPLE_RATE,
        window="hann_periodic",
        nperseg=nperseg,
        noverlap=nperseg // 2,
        detrend="constant",
        scaling="density",
        average="median",
    )
    resolution = ENVELOPE_SAMPLE_RATE / nperseg
    frequency_mask = (frequencies >= MIN_MODULATION_HZ - resolution) & (
        frequencies <= MAX_MODULATION_HZ + resolution
    )
    frequencies = frequencies[frequency_mask].astype(np.float64)
    power = power[frequency_mask].astype(np.float64)
    max_power = float(np.max(power))
    if not math.isfinite(max_power) or max_power <= np.finfo(np.float64).tiny:
        spectrum_db = np.full_like(power, SPECTRUM_FLOOR_DB, dtype=np.float64)
    else:
        floor = max_power * 10.0 ** (SPECTRUM_FLOOR_DB / 10.0)
        spectrum_db = (10.0 * np.log10(np.maximum(power, floor) / max_power)).astype(
            np.float64
        )
    return frequencies, spectrum_db, resolution


def _modulation_spectrogram(envelope: Float64Array) -> ModulationSpectrogram:
    window_samples = min(
        len(envelope), round(SPECTROGRAM_WINDOW_S * ENVELOPE_SAMPLE_RATE)
    )
    hop_samples = min(window_samples, round(SPECTROGRAM_HOP_S * ENVELOPE_SAMPLE_RATE))
    starts = np.arange(
        0,
        len(envelope) - window_samples + 1,
        hop_samples,
        dtype=np.int64,
    )
    final_start = len(envelope) - window_samples
    if int(starts[-1]) != final_start:
        starts = np.append(starts, final_start)
    frames = np.stack([
        envelope[start : start + window_samples] for start in starts.tolist()
    ])
    detrended = frames - np.mean(frames, axis=1, keepdims=True)
    window: Float64Array = hann(window_samples, sym=False).astype(np.float64)
    spectrum = np.fft.rfft(detrended * window, axis=1)
    power = np.square(np.abs(spectrum)) / (
        ENVELOPE_SAMPLE_RATE * float(np.sum(np.square(window)))
    )
    if window_samples % 2 == 0:
        power[:, 1:-1] *= 2.0
    else:
        power[:, 1:] *= 2.0

    frequencies: Float64Array = np.fft.rfftfreq(
        window_samples, 1.0 / ENVELOPE_SAMPLE_RATE
    ).astype(np.float64)
    frequency_mask = (frequencies >= MIN_MODULATION_HZ) & (
        frequencies <= MAX_MODULATION_HZ
    )
    band_frequencies = frequencies[frequency_mask].astype(np.float64)
    band_power = power[:, frequency_mask].T.astype(np.float64)
    psd_db = _relative_db(band_power, SPECTROGRAM_FLOOR_DB)
    times = (starts.astype(np.float64) + window_samples / 2.0) / ENVELOPE_SAMPLE_RATE
    return ModulationSpectrogram(
        times_s=times.astype(np.float64),
        frequencies_hz=band_frequencies,
        psd_db=psd_db,
        window_duration_s=window_samples / ENVELOPE_SAMPLE_RATE,
        hop_duration_s=hop_samples / ENVELOPE_SAMPLE_RATE,
        frequency_resolution_hz=ENVELOPE_SAMPLE_RATE / window_samples,
    )


def _relative_db(power: Float64Array, floor_db: float) -> Float64Array:
    max_power = float(np.max(power))
    if not math.isfinite(max_power) or max_power <= np.finfo(np.float64).tiny:
        return np.full_like(power, floor_db, dtype=np.float64)
    floor = max_power * 10.0 ** (floor_db / 10.0)
    return (10.0 * np.log10(np.maximum(power, floor) / max_power)).astype(np.float64)


def _detect_peaks(
    frequencies: Float64Array,
    spectrum_db: Float64Array,
    resolution_hz: float,
) -> tuple[ModulationPeak, ...]:
    distance = max(1, math.ceil(PEAK_SEPARATION_HZ / resolution_hz))
    indices, properties = find_peaks(
        spectrum_db,
        prominence=PEAK_PROMINENCE_DB,
        distance=distance,
    )
    prominences: Float64Array = properties["prominences"].astype(np.float64)
    in_band = (
        (index, prominence)
        for index, prominence in zip(
            indices.tolist(), prominences.tolist(), strict=True
        )
        if MIN_MODULATION_HZ <= float(frequencies[index]) <= MAX_MODULATION_HZ
    )
    ranked = sorted(
        in_band,
        key=lambda candidate: candidate[1],
        reverse=True,
    )[:MAX_PEAKS]
    selected = sorted(ranked, key=lambda candidate: float(frequencies[candidate[0]]))
    return tuple(
        ModulationPeak(
            frequency_hz=float(frequencies[index]),
            level_db=float(spectrum_db[index]),
            prominence_db=float(prominence),
        )
        for index, prominence in selected
    )


def _validate_input(
    samples: AudioInputArray,
    sample_rate: int,
    crossover_hz: float,
    filter_order: int,
) -> None:
    if samples.ndim != 1:
        msg = "modulation analysis requires mono samples"
        raise ValueError(msg)
    if np.iscomplexobj(samples):
        msg = "audio samples must be real"
        raise ValueError(msg)
    if sample_rate <= 0:
        msg = "sample rate must be positive"
        raise ValueError(msg)
    if sample_rate <= 2.0 * ENVELOPE_CUTOFF_HZ:
        msg = "sample-rate Nyquist frequency must exceed the envelope cutoff"
        raise ValueError(msg)
    if not math.isfinite(crossover_hz) or crossover_hz <= 0.0:
        msg = "crossover frequency must be finite and positive"
        raise ValueError(msg)
    if crossover_hz >= sample_rate / 2.0:
        msg = "sample-rate Nyquist frequency must exceed the crossover"
        raise ValueError(msg)
    if filter_order <= 0:
        msg = "filter order must be positive"
        raise ValueError(msg)
    if len(samples) < round(MIN_DURATION_S * sample_rate):
        msg = f"recording must be at least {MIN_DURATION_S:g} second"
        raise InsufficientAudioError(msg)
    if not bool(np.all(np.isfinite(samples))):
        msg = "audio samples must be finite"
        raise ValueError(msg)


def _validate_rpm(rpm: float) -> None:
    if not math.isfinite(rpm) or rpm <= 0.0:
        msg = "RPM must be finite and positive"
        raise ValueError(msg)


def _validate_video_fps(video_fps: float) -> None:
    if not math.isfinite(video_fps) or video_fps <= 0.0:
        msg = "video frame rate must be finite and positive"
        raise ValueError(msg)
