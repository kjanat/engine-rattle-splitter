"""Rattle-envelope modulation frequency analysis."""

import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import NDArray
from scipy.signal import butter, find_peaks, hilbert, resample_poly, sosfiltfilt, welch

from .audio_io import Float32Array, decode
from .filters import complementary_crossover

type Float64Array = NDArray[np.float64]

DEFAULT_CROSSOVER_HZ = 1800.0
DEFAULT_FILTER_ORDER = 4
ENVELOPE_CUTOFF_HZ = 100.0
ENVELOPE_SAMPLE_RATE = 400
MIN_MODULATION_HZ = 5.0
MAX_MODULATION_HZ = 100.0
WELCH_SEGMENT_S = 8.0
PEAK_PROMINENCE_DB = 6.0
PEAK_SEPARATION_HZ = 1.0
MAX_PEAKS = 6
MIN_DURATION_S = 1.0
SPECTRUM_FLOOR_DB = -120.0


@dataclass(frozen=True)
class ModulationPeak:
    """One prominent frequency in the rattle amplitude envelope."""

    frequency_hz: float
    level_db: float
    prominence_db: float


@dataclass(frozen=True)
class ModulationResult:
    """Envelope, modulation spectrum, and detected candidate frequencies."""

    times_s: Float64Array
    envelope: Float64Array
    frequencies_hz: Float64Array
    spectrum_db: Float64Array
    peaks: tuple[ModulationPeak, ...]
    frequency_resolution_hz: float


class InsufficientAudioError(ValueError):
    """Raised when a clip is too short for meaningful modulation analysis."""


def analyze(
    samples: Float32Array,
    sample_rate: int,
    *,
    crossover_hz: float = DEFAULT_CROSSOVER_HZ,
    filter_order: int = DEFAULT_FILTER_ORDER,
) -> ModulationResult:
    """Measure low-frequency amplitude modulation of the high-band signal."""
    _validate_input(samples, sample_rate, crossover_hz, filter_order)

    pad_samples = min(sample_rate, len(samples) - 1)
    padded: Float32Array = np.pad(samples, pad_samples, mode="reflect").astype(
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

    divisor = math.gcd(sample_rate, ENVELOPE_SAMPLE_RATE)
    envelope = resample_poly(
        envelope,
        ENVELOPE_SAMPLE_RATE // divisor,
        sample_rate // divisor,
    ).astype(np.float64)
    envelope = np.maximum(envelope, 0.0).astype(np.float64)
    times: Float64Array = np.arange(len(envelope), dtype=np.float64) / float(
        ENVELOPE_SAMPLE_RATE
    )

    spectrum_frequencies, full_spectrum_db, resolution = _spectrum(envelope)
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
    )


def order_ratio(frequency_hz: float, rpm: float) -> float:
    """Express a measured modulation frequency as a fixed shaft-speed order."""
    _validate_rpm(rpm)
    return frequency_hz / (rpm / 60.0)


def run(
    input_path: Path,
    sample_rate: int,
    output_png: Path,
    *,
    rpm: float | None = None,
    crossover_hz: float = DEFAULT_CROSSOVER_HZ,
    filter_order: int = DEFAULT_FILTER_ORDER,
) -> ModulationResult:
    """Decode, analyze, report, and render one recording."""
    if rpm is not None:
        _validate_rpm(rpm)
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
    )
    print_report(result, rpm=rpm)
    return result


def render(
    result: ModulationResult,
    *,
    input_name: str,
    output_png: Path,
    crossover_hz: float = DEFAULT_CROSSOVER_HZ,
    rpm: float | None = None,
) -> None:
    """Render the envelope and its low-frequency spectrum."""
    if rpm is not None:
        _validate_rpm(rpm)

    fig: Figure
    axes_arr: NDArray[np.object_]
    fig, axes_arr = plt.subplots(2, 1, figsize=(14, 8), dpi=140, squeeze=False)
    axes: list[Axes] = [axes_arr[index, 0] for index in range(2)]

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

    spectrum_ax = axes[1]
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


def print_report(result: ModulationResult, *, rpm: float | None = None) -> None:
    """Print measured frequencies and optional fixed-speed order ratios."""
    if rpm is not None:
        _validate_rpm(rpm)

    print("\nRattle-envelope modulation peaks:")
    if not result.peaks:
        print("  none above prominence threshold")
    for peak in result.peaks:
        print(f"  {peak.frequency_hz:6.1f} Hz  prominence={peak.prominence_db:5.1f} dB")

    print(f"Frequency resolution: {result.frequency_resolution_hz:.2f} Hz")
    if rpm is None:
        return

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
    samples: Float32Array,
    sample_rate: int,
    crossover_hz: float,
    filter_order: int,
) -> None:
    if samples.ndim != 1:
        msg = "modulation analysis requires mono samples"
        raise ValueError(msg)
    if sample_rate <= 0:
        msg = "sample rate must be positive"
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
