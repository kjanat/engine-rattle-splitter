"""Measured RPM ingestion and time-varying modulation order analysis."""

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from .fault_diagnostics import FrequencyTrack
from .modulation import SPECTROGRAM_FLOOR_DB, ModulationSpectrogram

type Float64Array = NDArray[np.float64]
type BoolArray = NDArray[np.bool_]

DEFAULT_REFERENCE_ORDERS = (0.5, 1.0, 1.5, 2.0, 3.0, 4.0)


@dataclass(frozen=True)
class RpmPoint:
    time_s: float
    rpm: float


@dataclass(frozen=True)
class FixedRpm:
    kind: Literal["fixed"]
    rpm: float


@dataclass(frozen=True)
class RpmTrace:
    kind: Literal["trace"]
    points: tuple[RpmPoint, ...]
    source: Path


type SpeedReference = FixedRpm | RpmTrace


@dataclass(frozen=True)
class OrderFit:
    track_id: int
    rpm_coverage: float
    fitted_order: float | None
    order_mad: float | None
    nearest_reference_order: float | None
    match_fraction: float | None
    frequency_rpm_correlation: float | None
    consistent: bool


@dataclass(frozen=True)
class OrderMap:
    times_s: Float64Array
    orders: Float64Array
    order_resolution: float
    psd_db: Float64Array
    valid: BoolArray


@dataclass(frozen=True)
class OrderAnalysis:
    speed: SpeedReference
    reference_orders: tuple[float, ...]
    fits: tuple[OrderFit, ...]
    order_map: OrderMap


def fixed_rpm(rpm: float) -> FixedRpm:
    _validate_rpm(rpm)
    return FixedRpm(kind="fixed", rpm=rpm)


def load_rpm_trace(path: Path) -> RpmTrace:
    """Parse strict media-relative `time_s,rpm` measurements."""
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["time_s", "rpm"]:
            msg = "RPM trace must contain exactly the columns time_s,rpm"
            raise ValueError(msg)
        points: list[RpmPoint] = []
        for row_number, row in enumerate(reader, start=2):
            try:
                time_s = float(row["time_s"])
                rpm = float(row["rpm"])
            except (KeyError, TypeError, ValueError) as error:
                msg = f"invalid RPM trace row {row_number}"
                raise ValueError(msg) from error
            if not math.isfinite(time_s) or time_s < 0.0:
                msg = f"RPM trace row {row_number} has invalid time_s"
                raise ValueError(msg)
            _validate_rpm(rpm)
            if points and time_s <= points[-1].time_s:
                msg = "RPM trace times must be strictly increasing"
                raise ValueError(msg)
            points.append(RpmPoint(time_s=time_s, rpm=rpm))
    if len(points) < 2:
        msg = "RPM trace requires at least two points"
        raise ValueError(msg)
    return RpmTrace(kind="trace", points=tuple(points), source=path)


def analyze_orders(
    tracks: tuple[FrequencyTrack, ...],
    spectrogram: ModulationSpectrogram,
    speed: SpeedReference,
    *,
    reference_orders: tuple[float, ...] = DEFAULT_REFERENCE_ORDERS,
) -> OrderAnalysis:
    _validate_reference_orders(reference_orders)
    fits = tuple(
        _fit_track(track, speed, reference_orders, spectrogram.frequency_resolution_hz)
        for track in tracks
    )
    return OrderAnalysis(
        speed=speed,
        reference_orders=reference_orders,
        fits=fits,
        order_map=_order_map(spectrogram, speed, reference_orders),
    )


def order_curves(
    times_s: Float64Array,
    speed: SpeedReference,
    reference_orders: tuple[float, ...],
) -> tuple[tuple[float, Float64Array, Float64Array], ...]:
    rpm_values, valid = interpolate_rpm(times_s, speed)
    valid_times = times_s[valid].astype(np.float64)
    valid_rpm = rpm_values[valid]
    return tuple(
        (
            order,
            valid_times,
            (order * valid_rpm / 60.0).astype(np.float64),
        )
        for order in reference_orders
    )


def interpolate_rpm(
    times_s: Float64Array, speed: SpeedReference
) -> tuple[Float64Array, BoolArray]:
    if speed.kind == "fixed":
        return (
            np.full_like(times_s, speed.rpm, dtype=np.float64),
            np.ones_like(times_s, dtype=np.bool_),
        )
    source_times = np.array([point.time_s for point in speed.points], dtype=np.float64)
    source_rpm = np.array([point.rpm for point in speed.points], dtype=np.float64)
    valid = (times_s >= source_times[0]) & (times_s <= source_times[-1])
    values = np.full_like(times_s, 0.0, dtype=np.float64)
    values[valid] = np.interp(times_s[valid], source_times, source_rpm)
    return values, valid.astype(np.bool_)


def _fit_track(
    track: FrequencyTrack,
    speed: SpeedReference,
    reference_orders: tuple[float, ...],
    frequency_bin_hz: float,
) -> OrderFit:
    times = np.array([point.time_s for point in track.points], dtype=np.float64)
    frequencies = np.array(
        [point.frequency_hz for point in track.points], dtype=np.float64
    )
    rpm, valid = interpolate_rpm(times, speed)
    coverage = float(np.mean(valid))
    if int(np.count_nonzero(valid)) < 2:
        return OrderFit(
            track_id=track.track_id,
            rpm_coverage=coverage,
            fitted_order=None,
            order_mad=None,
            nearest_reference_order=None,
            match_fraction=None,
            frequency_rpm_correlation=None,
            consistent=False,
        )
    valid_frequency = frequencies[valid]
    valid_rpm = rpm[valid]
    measured_orders = valid_frequency * 60.0 / valid_rpm
    fitted_order = float(np.median(measured_orders))
    order_mad = float(np.median(np.abs(measured_orders - fitted_order)))
    nearest = min(reference_orders, key=lambda order: abs(order - fitted_order))
    predicted = nearest * valid_rpm / 60.0
    tolerance = np.maximum(frequency_bin_hz, 0.03 * predicted)
    match_fraction = float(np.mean(np.abs(valid_frequency - predicted) <= tolerance))
    correlation = _correlation(valid_frequency, valid_rpm)
    return OrderFit(
        track_id=track.track_id,
        rpm_coverage=coverage,
        fitted_order=fitted_order,
        order_mad=order_mad,
        nearest_reference_order=nearest,
        match_fraction=match_fraction,
        frequency_rpm_correlation=correlation,
        consistent=(
            coverage >= 0.8 and len(valid_frequency) >= 4 and match_fraction >= 0.7
        ),
    )


def _order_map(
    spectrogram: ModulationSpectrogram,
    speed: SpeedReference,
    reference_orders: tuple[float, ...],
) -> OrderMap:
    source_rpm = (
        np.array([speed.rpm], dtype=np.float64)
        if speed.kind == "fixed"
        else np.array([point.rpm for point in speed.points], dtype=np.float64)
    )
    minimum_order = min(
        float(spectrogram.frequencies_hz[0] * 60.0 / np.max(source_rpm)),
        *reference_orders,
    )
    maximum_order = max(
        float(spectrogram.frequencies_hz[-1] * 60.0 / np.min(source_rpm)),
        *reference_orders,
    )
    order_count = min(
        400, max(2, math.ceil((maximum_order - minimum_order) / 0.05) + 1)
    )
    orders = np.linspace(minimum_order, maximum_order, order_count, dtype=np.float64)
    order_resolution = float(orders[1] - orders[0])
    rpm, valid_times = interpolate_rpm(spectrogram.times_s, speed)
    values = np.full(
        (len(orders), len(spectrogram.times_s)),
        SPECTROGRAM_FLOOR_DB,
        dtype=np.float64,
    )
    valid = np.zeros_like(values, dtype=np.bool_)
    for time_index in np.flatnonzero(valid_times).tolist():
        target_frequencies = orders * rpm[time_index] / 60.0
        inside = (target_frequencies >= spectrogram.frequencies_hz[0]) & (
            target_frequencies <= spectrogram.frequencies_hz[-1]
        )
        values[inside, time_index] = np.interp(
            target_frequencies[inside],
            spectrogram.frequencies_hz,
            spectrogram.psd_db[:, time_index],
        )
        valid[inside, time_index] = True
    return OrderMap(
        times_s=spectrogram.times_s,
        orders=orders,
        order_resolution=order_resolution,
        psd_db=values,
        valid=valid,
    )


def _correlation(left: Float64Array, right: Float64Array) -> float | None:
    if len(left) < 3 or float(np.std(left)) == 0.0 or float(np.std(right)) == 0.0:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def _validate_rpm(rpm: float) -> None:
    if not math.isfinite(rpm) or rpm <= 0.0:
        msg = "RPM must be finite and positive"
        raise ValueError(msg)


def _validate_reference_orders(reference_orders: tuple[float, ...]) -> None:
    if not reference_orders or any(
        not math.isfinite(order) or order <= 0.0 for order in reference_orders
    ):
        msg = "reference orders must be finite and positive"
        raise ValueError(msg)
