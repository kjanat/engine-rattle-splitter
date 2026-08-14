"""Cross-method evidence for localizing periodic mechanical rattles."""

import math
from dataclasses import dataclass
from itertools import combinations, pairwise
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy.signal import (
    butter,
    coherence,
    find_peaks,
    hilbert,
    sosfiltfilt,
)

from .audio_io import Float32Array
from .filters import complementary_crossover
from .modulation import (
    DEFAULT_FILTER_ORDER,
    ENVELOPE_CUTOFF_HZ,
    ENVELOPE_SAMPLE_RATE,
    MAX_MODULATION_HZ,
    MIN_MODULATION_HZ,
    ModulationPeak,
    ModulationResult,
    ModulationSpectrogram,
    _detect_peaks,
    _modulation_spectrogram,
    _resample_envelope,
    _spectrum,
)

type Float64Array = NDArray[np.float64]
type EvidenceLabel = Literal["limited", "moderate", "strong"]


@dataclass(frozen=True)
class FrequencyBand:
    name: str
    low_hz: float
    high_hz: float


@dataclass(frozen=True)
class SubbandAnalysis:
    band: FrequencyBand
    carrier_rms: float
    informative: bool
    envelope: Float64Array
    frequencies_hz: Float64Array
    spectrum_db: Float64Array
    peaks: tuple[ModulationPeak, ...]


@dataclass(frozen=True)
class RidgePoint:
    time_s: float
    frequency_hz: float
    level_db: float


@dataclass(frozen=True)
class FrequencyTrack:
    track_id: int
    points: tuple[RidgePoint, ...]
    duration_s: float
    median_frequency_hz: float
    minimum_frequency_hz: float
    maximum_frequency_hz: float
    slope_hz_per_s: float


@dataclass(frozen=True)
class DetectedEvent:
    time_s: float
    strength_z: float
    width_ms: float


@dataclass(frozen=True)
class EventPeriodicity:
    frequency_hz: float
    prominence_db: float
    autocorrelation: float


@dataclass(frozen=True)
class EventAnalysis:
    events: tuple[DetectedEvent, ...]
    periodicities: tuple[EventPeriodicity, ...]
    median_width_ms: float | None
    interval_cv: float | None


@dataclass(frozen=True)
class HarmonicMember:
    frequency_hz: float
    harmonic: int


@dataclass(frozen=True)
class HarmonicFamily:
    base_frequency_hz: float
    members: tuple[HarmonicMember, ...]


@dataclass(frozen=True)
class CandidateEvidence:
    frequency_hz: float
    prominence_db: float
    informative_subbands: int
    supporting_subbands: int
    median_subband_coherence: float | None
    longest_track_s: float
    event_phase_locking: float | None
    event_rate_match: bool
    score: float
    label: EvidenceLabel


@dataclass(frozen=True)
class FaultDiagnostics:
    subbands: tuple[SubbandAnalysis, ...]
    consensus_spectrogram: ModulationSpectrogram | None
    tracks: tuple[FrequencyTrack, ...]
    events: EventAnalysis
    harmonic_families: tuple[HarmonicFamily, ...]
    candidates: tuple[CandidateEvidence, ...]
    warnings: tuple[str, ...]


def analyze_fault_evidence(
    samples: Float32Array,
    sample_rate: int,
    base: ModulationResult,
    *,
    crossover_hz: float,
    filter_order: int = DEFAULT_FILTER_ORDER,
) -> FaultDiagnostics:
    """Corroborate broad-envelope peaks with independent signal views."""
    real_samples = samples.astype(np.float32, copy=False)
    pad_samples = min(sample_rate, len(real_samples) - 1)
    padded: Float32Array = np.pad(real_samples, pad_samples, mode="reflect").astype(
        np.float32, copy=False
    )
    _, broad_high = complementary_crossover(
        padded, sample_rate, crossover_hz, filter_order
    )
    broad_rms = _rms(broad_high[pad_samples:-pad_samples])
    subbands = _analyze_subbands(
        padded,
        sample_rate,
        pad_samples,
        crossover_hz,
        broad_rms,
    )
    informative = tuple(subband for subband in subbands if subband.informative)
    consensus = _consensus_spectrogram(informative)
    track_source = consensus if consensus is not None else base.spectrogram
    tracks = _extract_tracks(
        track_source, tuple(peak.frequency_hz for peak in base.peaks)
    )
    events = _analyze_events(base.envelope)
    coherence_by_frequency = _candidate_coherence(informative, base.peaks)
    candidates = _candidate_evidence(
        base.peaks,
        informative,
        tracks,
        events,
        coherence_by_frequency,
        base.frequency_resolution_hz,
    )
    warnings: list[str] = []
    if len(informative) < 2:
        warnings.append("fewer than two carrier subbands contain useful energy")
    if not events.events:
        warnings.append("no robust high-band burst events were detected")
    warnings.append(
        "evidence scores are uncalibrated rankings, not causal probabilities"
    )
    return FaultDiagnostics(
        subbands=subbands,
        consensus_spectrogram=consensus,
        tracks=tracks,
        events=events,
        harmonic_families=_harmonic_families(base.peaks, base.frequency_resolution_hz),
        candidates=candidates,
        warnings=tuple(warnings),
    )


def _carrier_bands(sample_rate: int, crossover_hz: float) -> tuple[FrequencyBand, ...]:
    upper_limit = min(16_000.0, 0.45 * sample_rate)
    internal_boundaries = tuple(
        boundary
        for boundary in (4_000.0, 8_000.0, 12_000.0)
        if crossover_hz < boundary < upper_limit
    )
    boundaries = (crossover_hz, *internal_boundaries, upper_limit)
    bands: list[FrequencyBand] = []
    for low_hz, high_hz in pairwise(boundaries):
        if high_hz - low_hz < 500.0:
            continue
        bands.append(
            FrequencyBand(
                name=f"{low_hz / 1000:g}-{high_hz / 1000:g} kHz",
                low_hz=low_hz,
                high_hz=high_hz,
            )
        )
    return tuple(bands)


def _analyze_subbands(
    padded: Float32Array,
    sample_rate: int,
    pad_samples: int,
    crossover_hz: float,
    broad_rms: float,
) -> tuple[SubbandAnalysis, ...]:
    analyses: list[SubbandAnalysis] = []
    informative_floor = broad_rms * 10.0 ** (-30.0 / 20.0)
    broad_has_energy = broad_rms > np.finfo(np.float64).tiny
    carrier_frequencies = np.fft.rfftfreq(len(padded), d=1.0 / sample_rate)
    carrier_spectrum = np.fft.rfft(padded.astype(np.float64))
    envelope_sos = butter(
        DEFAULT_FILTER_ORDER,
        ENVELOPE_CUTOFF_HZ,
        btype="low",
        fs=sample_rate,
        output="sos",
    )
    for band in _carrier_bands(sample_rate, crossover_hz):
        carrier_mask = (carrier_frequencies >= band.low_hz) & (
            carrier_frequencies < band.high_hz
        )
        carrier = np.fft.irfft(
            np.where(carrier_mask, carrier_spectrum, 0.0), n=len(padded)
        ).astype(np.float32)
        carrier_rms = _rms(carrier[pad_samples:-pad_samples])
        analytic = np.abs(hilbert(carrier)).astype(np.float64)
        smoothed = sosfiltfilt(envelope_sos, analytic).astype(np.float64)
        envelope = np.maximum(smoothed[pad_samples:-pad_samples], 0.0).astype(
            np.float64
        )
        envelope = np.maximum(_resample_envelope(envelope, sample_rate), 0.0)
        frequencies, spectrum_db, resolution = _spectrum(envelope)
        peaks = _detect_peaks(frequencies, spectrum_db, resolution)
        band_mask = (frequencies >= MIN_MODULATION_HZ) & (
            frequencies <= MAX_MODULATION_HZ
        )
        analyses.append(
            SubbandAnalysis(
                band=band,
                carrier_rms=carrier_rms,
                informative=broad_has_energy and carrier_rms >= informative_floor,
                envelope=envelope,
                frequencies_hz=frequencies[band_mask].astype(np.float64),
                spectrum_db=spectrum_db[band_mask].astype(np.float64),
                peaks=peaks,
            )
        )
    return tuple(analyses)


def _consensus_spectrogram(
    informative: tuple[SubbandAnalysis, ...],
) -> ModulationSpectrogram | None:
    if len(informative) < 2:
        return None
    normalized: list[Float64Array] = []
    for subband in informative:
        scale = max(float(np.median(subband.envelope)), np.finfo(np.float64).eps)
        normalized.append(np.log1p(subband.envelope / scale).astype(np.float64))
    consensus = np.median(np.stack(normalized), axis=0).astype(np.float64)
    return _modulation_spectrogram(consensus)


def _extract_tracks(
    spectrogram: ModulationSpectrogram,
    candidate_frequencies_hz: tuple[float, ...],
) -> tuple[FrequencyTrack, ...]:
    tracks: list[list[RidgePoint]] = []
    resolution = spectrogram.frequency_resolution_hz
    distance = max(1, math.ceil(1.0 / resolution))
    for frame_index, time_s in enumerate(spectrogram.times_s.tolist()):
        column = spectrogram.psd_db[:, frame_index]
        indices, _ = find_peaks(column, prominence=6.0, distance=distance)
        indices = indices[column[indices] >= -25.0]
        ranked = sorted(
            indices.tolist(), key=lambda index: float(column[index]), reverse=True
        )
        selected = [
            index
            for rank, index in enumerate(ranked)
            if rank < 3
            or any(
                abs(float(spectrogram.frequencies_hz[index]) - candidate_hz) <= 3.0
                for candidate_hz in candidate_frequencies_hz
            )
        ][:8]
        used_tracks: set[int] = set()
        for index in selected:
            frequency_hz = float(spectrogram.frequencies_hz[index])
            best_track: int | None = None
            best_delta = math.inf
            for track_index, points in enumerate(tracks):
                if track_index in used_tracks:
                    continue
                elapsed = time_s - points[-1].time_s
                if elapsed <= 0.0 or elapsed > 0.51:
                    continue
                allowed = max(2.0 * resolution, 20.0 * elapsed)
                delta = abs(frequency_hz - points[-1].frequency_hz)
                if delta <= allowed and delta < best_delta:
                    best_track = track_index
                    best_delta = delta
            point = RidgePoint(
                time_s=time_s,
                frequency_hz=frequency_hz,
                level_db=float(column[index]),
            )
            if best_track is None:
                tracks.append([point])
                used_tracks.add(len(tracks) - 1)
            else:
                tracks[best_track].append(point)
                used_tracks.add(best_track)

    retained: list[tuple[float, FrequencyTrack]] = []
    for points in tracks:
        duration_s = points[-1].time_s - points[0].time_s
        if duration_s < 1.0:
            continue
        times = np.array([point.time_s for point in points], dtype=np.float64)
        frequencies = np.array(
            [point.frequency_hz for point in points], dtype=np.float64
        )
        median_frequency_hz = float(np.median(frequencies))
        median_level_db = float(np.median([point.level_db for point in points]))
        near_candidate = any(
            abs(median_frequency_hz - candidate_hz) <= 3.0
            for candidate_hz in candidate_frequencies_hz
        )
        if not near_candidate and median_level_db < -15.0:
            continue
        retained.append((
            duration_s * max(1.0, 30.0 + median_level_db),
            FrequencyTrack(
                track_id=0,
                points=tuple(points),
                duration_s=duration_s,
                median_frequency_hz=median_frequency_hz,
                minimum_frequency_hz=float(np.min(frequencies)),
                maximum_frequency_hz=float(np.max(frequencies)),
                slope_hz_per_s=_linear_slope(times, frequencies),
            ),
        ))
    ranked_tracks = sorted(retained, key=lambda item: item[0], reverse=True)[:12]
    return tuple(
        FrequencyTrack(
            track_id=index,
            points=track.points,
            duration_s=track.duration_s,
            median_frequency_hz=track.median_frequency_hz,
            minimum_frequency_hz=track.minimum_frequency_hz,
            maximum_frequency_hz=track.maximum_frequency_hz,
            slope_hz_per_s=track.slope_hz_per_s,
        )
        for index, (_, track) in enumerate(ranked_tracks, start=1)
    )


def _analyze_events(envelope: Float64Array) -> EventAnalysis:
    scale = max(float(np.median(envelope)), np.finfo(np.float64).eps)
    log_envelope = np.log1p(envelope / scale)
    novelty = np.maximum(np.diff(log_envelope, prepend=log_envelope[0]), 0.0)
    positive_novelty = novelty[novelty > np.finfo(np.float64).eps]
    if len(positive_novelty) == 0:
        return EventAnalysis(
            events=(), periodicities=(), median_width_ms=None, interval_cv=None
        )
    median = float(np.median(positive_novelty))
    mad = float(np.median(np.abs(positive_novelty - median)))
    if mad <= np.finfo(np.float64).eps:
        return EventAnalysis(
            events=(), periodicities=(), median_width_ms=None, interval_cv=None
        )
    z_score = (novelty - median) / (1.4826 * mad)
    indices, _ = find_peaks(
        z_score,
        height=4.0,
        prominence=2.0,
        distance=max(1, round(0.01 * ENVELOPE_SAMPLE_RATE)),
    )
    if len(indices) == 0:
        return EventAnalysis(
            events=(), periodicities=(), median_width_ms=None, interval_cv=None
        )
    width_samples = np.array(
        [
            _event_width_samples(
                log_envelope,
                index,
                indices[position + 1] if position + 1 < len(indices) else len(envelope),
            )
            for position, index in enumerate(indices.tolist())
        ],
        dtype=np.float64,
    )
    events = tuple(
        DetectedEvent(
            time_s=float(index / ENVELOPE_SAMPLE_RATE),
            strength_z=float(z_score[index]),
            width_ms=float(width * 1000.0 / ENVELOPE_SAMPLE_RATE),
        )
        for index, width in zip(indices.tolist(), width_samples.tolist(), strict=True)
    )
    periodicities: tuple[EventPeriodicity, ...] = ()
    if len(events) >= 5:
        impulse_train = np.zeros_like(envelope)
        impulse_train[indices] = z_score[indices]
        frequencies, spectrum_db, resolution = _spectrum(impulse_train)
        peaks = _detect_peaks(frequencies, spectrum_db, resolution)
        periodicities = tuple(
            EventPeriodicity(
                frequency_hz=peak.frequency_hz,
                prominence_db=peak.prominence_db,
                autocorrelation=_autocorrelation_at(impulse_train, peak.frequency_hz),
            )
            for peak in peaks
            if _autocorrelation_at(impulse_train, peak.frequency_hz) >= 0.2
        )
    intervals = np.diff(np.array([event.time_s for event in events]))
    interval_cv = None
    if len(intervals) > 1 and float(np.mean(intervals)) > 0.0:
        interval_cv = float(np.std(intervals) / np.mean(intervals))
    return EventAnalysis(
        events=events,
        periodicities=periodicities,
        median_width_ms=float(np.median(width_samples) * 1000.0 / ENVELOPE_SAMPLE_RATE),
        interval_cv=interval_cv,
    )


def _candidate_coherence(
    informative: tuple[SubbandAnalysis, ...], peaks: tuple[ModulationPeak, ...]
) -> dict[float, float]:
    if len(informative) < 2 or len(informative[0].envelope) < 2_000:
        return {}
    values: dict[float, list[float]] = {peak.frequency_hz: [] for peak in peaks}
    for left, right in combinations(informative, 2):
        frequencies, coherence_values = coherence(
            left.envelope,
            right.envelope,
            fs=ENVELOPE_SAMPLE_RATE,
            nperseg=min(800, len(left.envelope)),
            noverlap=min(400, len(left.envelope) // 2),
        )
        for peak in peaks:
            index = int(np.argmin(np.abs(frequencies - peak.frequency_hz)))
            values[peak.frequency_hz].append(float(coherence_values[index]))
    return {
        frequency_hz: float(np.median(candidate_values))
        for frequency_hz, candidate_values in values.items()
        if candidate_values
    }


def _candidate_evidence(
    peaks: tuple[ModulationPeak, ...],
    informative: tuple[SubbandAnalysis, ...],
    tracks: tuple[FrequencyTrack, ...],
    events: EventAnalysis,
    coherence_by_frequency: dict[float, float],
    resolution_hz: float,
) -> tuple[CandidateEvidence, ...]:
    evidence: list[CandidateEvidence] = []
    event_times = np.array([event.time_s for event in events.events], dtype=np.float64)
    event_weights = np.array(
        [event.strength_z for event in events.events], dtype=np.float64
    )
    for peak in peaks:
        tolerance_hz = max(1.0, 2.0 * resolution_hz)
        supporting = sum(
            any(
                abs(sub_peak.frequency_hz - peak.frequency_hz) <= tolerance_hz
                for sub_peak in subband.peaks
            )
            for subband in informative
        )
        longest_track = max(
            (
                track.duration_s
                for track in tracks
                if track.minimum_frequency_hz - 2.0
                <= peak.frequency_hz
                <= track.maximum_frequency_hz + 2.0
            ),
            default=0.0,
        )
        phase_locking = _phase_locking(event_times, event_weights, peak.frequency_hz)
        event_rate_match = any(
            abs(periodicity.frequency_hz - peak.frequency_hz) <= tolerance_hz
            for periodicity in events.periodicities
        )
        coherence_value = coherence_by_frequency.get(peak.frequency_hz)
        components: list[tuple[float, float]] = [
            (0.30, min(1.0, peak.prominence_db / 30.0)),
            (
                0.20,
                supporting / len(informative) if informative else 0.0,
            ),
            (0.15, min(1.0, longest_track / 3.0)),
        ]
        if coherence_value is not None:
            components.append((0.15, coherence_value))
        if phase_locking is not None:
            components.append((0.20, phase_locking))
        weight_sum = sum(weight for weight, _ in components)
        score = sum(weight * value for weight, value in components) / weight_sum
        independent_support = supporting >= 2 or event_rate_match
        if not independent_support:
            label: EvidenceLabel = "limited"
        elif score >= 0.70:
            label = "strong"
        elif score >= 0.45:
            label = "moderate"
        else:
            label = "limited"
        evidence.append(
            CandidateEvidence(
                frequency_hz=peak.frequency_hz,
                prominence_db=peak.prominence_db,
                informative_subbands=len(informative),
                supporting_subbands=supporting,
                median_subband_coherence=coherence_value,
                longest_track_s=longest_track,
                event_phase_locking=phase_locking,
                event_rate_match=event_rate_match,
                score=score,
                label=label,
            )
        )
    return tuple(sorted(evidence, key=lambda candidate: candidate.score, reverse=True))


def _harmonic_families(
    peaks: tuple[ModulationPeak, ...], resolution_hz: float
) -> tuple[HarmonicFamily, ...]:
    families: list[HarmonicFamily] = []
    used: set[float] = set()
    for base_peak in sorted(peaks, key=lambda peak: peak.frequency_hz):
        if base_peak.frequency_hz in used:
            continue
        members: list[HarmonicMember] = []
        for peak in peaks:
            harmonic = round(peak.frequency_hz / base_peak.frequency_hz)
            if not 1 <= harmonic <= 8:
                continue
            expected = harmonic * base_peak.frequency_hz
            tolerance = max(2.0 * resolution_hz, 0.03 * peak.frequency_hz)
            if abs(peak.frequency_hz - expected) <= tolerance:
                members.append(
                    HarmonicMember(frequency_hz=peak.frequency_hz, harmonic=harmonic)
                )
        if len(members) < 2:
            continue
        families.append(
            HarmonicFamily(
                base_frequency_hz=base_peak.frequency_hz,
                members=tuple(members),
            )
        )
        used.update(member.frequency_hz for member in members)
    return tuple(families)


def _phase_locking(
    event_times: Float64Array, weights: Float64Array, frequency_hz: float
) -> float | None:
    if len(event_times) < 5 or float(np.sum(weights)) <= 0.0:
        return None
    phases = np.exp(2j * np.pi * frequency_hz * event_times)
    return float(abs(np.sum(weights * phases)) / np.sum(weights))


def _event_width_samples(
    log_envelope: Float64Array, onset_index: int, next_onset_index: int
) -> float:
    search_end = min(
        next_onset_index,
        len(log_envelope),
    )
    segment = log_envelope[onset_index:search_end]
    if len(segment) == 0:
        return 1.0
    peak_offset = int(np.argmax(segment))
    peak_value = float(segment[peak_offset])
    baseline = float(np.median(log_envelope))
    half_height = baseline + 0.5 * max(0.0, peak_value - baseline)
    above = segment >= half_height
    left_candidates = np.flatnonzero(~above[: peak_offset + 1])
    left = int(left_candidates[-1] + 1) if len(left_candidates) else 0
    right_candidates = np.flatnonzero(~above[peak_offset:])
    right = (
        peak_offset + int(right_candidates[0])
        if len(right_candidates)
        else len(segment)
    )
    return float(max(1, right - left))


def _autocorrelation_at(signal: Float64Array, frequency_hz: float) -> float:
    lag = round(ENVELOPE_SAMPLE_RATE / frequency_hz)
    if lag <= 0 or lag >= len(signal):
        return 0.0
    left = signal[:-lag]
    right = signal[lag:]
    denominator = math.sqrt(float(np.dot(left, left) * np.dot(right, right)))
    if denominator <= np.finfo(np.float64).tiny:
        return 0.0
    return float(np.dot(left, right) / denominator)


def _linear_slope(times: Float64Array, values: Float64Array) -> float:
    centered_times = times - float(np.mean(times))
    denominator = float(np.dot(centered_times, centered_times))
    if denominator <= np.finfo(np.float64).tiny:
        return 0.0
    centered_values = values - float(np.mean(values))
    return float(np.dot(centered_times, centered_values) / denominator)


def _rms(samples: Float32Array) -> float:
    return float(np.sqrt(float(np.mean(np.square(samples, dtype=np.float64)))))
