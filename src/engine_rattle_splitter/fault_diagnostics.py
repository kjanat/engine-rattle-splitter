"""Cross-method evidence for localizing periodic mechanical rattles."""

import math
from dataclasses import dataclass
from functools import reduce
from itertools import combinations, pairwise
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy.signal import (
    butter,
    coherence,
    find_peaks,
    firwin,
    hilbert,
    kaiserord,
    oaconvolve,
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
    detect_peaks,
    modulation_spectrogram,
    resample_envelope,
    spectrum,
)

type Float64Array = NDArray[np.float64]
type EvidenceLabel = Literal["limited", "moderate", "strong"]

SUBBAND_GUARD_HZ = 100.0
SUBBAND_TRANSITION_HZ = 100.0
SUBBAND_ATTENUATION_DB = 60.0


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
    spectrogram: ModulationSpectrogram


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
class GlobalPeakEvidence:
    frequency_hz: float
    prominence_db: float
    informative_subbands: int
    supporting_subbands: int
    median_subband_coherence: float | None
    event_phase_locking: float | None
    event_rate_match: bool
    score: float
    label: EvidenceLabel


@dataclass(frozen=True)
class SubbandTrackSupport:
    band: FrequencyBand
    persistence: float
    median_contrast_db: float


@dataclass(frozen=True)
class TrackEventSupport:
    event_count: int
    phase_locking: float
    one_cycle_interval_fraction: float
    periodic: bool


@dataclass(frozen=True)
class TrackEvidence:
    track_id: int
    frame_coverage: float
    subbands: tuple[SubbandTrackSupport, ...]
    supporting_subbands: int
    events: TrackEventSupport | None
    score: float
    label: EvidenceLabel


@dataclass(frozen=True)
class FaultDiagnostics:
    subbands: tuple[SubbandAnalysis, ...]
    consensus_spectrogram: ModulationSpectrogram | None
    tracks: tuple[FrequencyTrack, ...]
    track_evidence: tuple[TrackEvidence, ...]
    events: EventAnalysis
    harmonic_families: tuple[HarmonicFamily, ...]
    global_peak_evidence: tuple[GlobalPeakEvidence, ...]
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
    tracks = _extract_tracks(track_source)
    events = _analyze_events(base.envelope)
    track_evidence = _track_evidence(tracks, informative, events, track_source)
    coherence_by_frequency = _candidate_coherence(informative, base.peaks)
    global_peak_evidence = _global_peak_evidence(
        base.peaks,
        informative,
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
        track_evidence=track_evidence,
        events=events,
        harmonic_families=_harmonic_families(base.peaks, base.frequency_resolution_hz),
        global_peak_evidence=global_peak_evidence,
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
    for index, (nominal_low_hz, nominal_high_hz) in enumerate(pairwise(boundaries)):
        low_hz = nominal_low_hz + (SUBBAND_GUARD_HZ if index > 0 else 0.0)
        high_hz = nominal_high_hz - (
            SUBBAND_GUARD_HZ if index < len(boundaries) - 2 else 0.0
        )
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


def _fir_bandpass(
    padded: Float32Array, sample_rate: int, band: FrequencyBand
) -> Float32Array:
    normalized_transition = SUBBAND_TRANSITION_HZ / (sample_rate / 2.0)
    tap_count, beta = kaiserord(SUBBAND_ATTENUATION_DB, normalized_transition)
    tap_count = max(31, tap_count | 1)
    taps = firwin(
        tap_count,
        (band.low_hz, band.high_hz),
        pass_zero=False,
        fs=sample_rate,
        window=("kaiser", beta),
    )
    return oaconvolve(padded.astype(np.float64), taps, mode="same").astype(np.float32)


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
    envelope_sos = butter(
        DEFAULT_FILTER_ORDER,
        ENVELOPE_CUTOFF_HZ,
        btype="low",
        fs=sample_rate,
        output="sos",
    )
    for band in _carrier_bands(sample_rate, crossover_hz):
        carrier = _fir_bandpass(padded, sample_rate, band)
        carrier_rms = _rms(carrier[pad_samples:-pad_samples])
        analytic = np.abs(hilbert(carrier)).astype(np.float64)
        smoothed = sosfiltfilt(envelope_sos, analytic).astype(np.float64)
        envelope = np.maximum(smoothed[pad_samples:-pad_samples], 0.0).astype(
            np.float64
        )
        envelope = np.maximum(resample_envelope(envelope, sample_rate), 0.0)
        frequencies, spectrum_db, resolution = spectrum(envelope)
        peaks = detect_peaks(frequencies, spectrum_db, resolution)
        spectrogram = modulation_spectrogram(envelope)
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
                spectrogram=spectrogram,
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
    return modulation_spectrogram(consensus)


def _extract_tracks(spectrogram: ModulationSpectrogram) -> tuple[FrequencyTrack, ...]:
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
        selected = ranked[:8]
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
                predicted_hz = points[-1].frequency_hz
                if len(points) >= 2:
                    previous_elapsed = points[-1].time_s - points[-2].time_s
                    if previous_elapsed > 0.0:
                        previous_slope = (
                            points[-1].frequency_hz - points[-2].frequency_hz
                        ) / previous_elapsed
                        predicted_hz += previous_slope * elapsed
                delta = abs(frequency_hz - predicted_hz)
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
        if median_level_db < -25.0:
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


def _track_evidence(
    tracks: tuple[FrequencyTrack, ...],
    informative: tuple[SubbandAnalysis, ...],
    events: EventAnalysis,
    spectrogram: ModulationSpectrogram,
) -> tuple[TrackEvidence, ...]:
    evidence: list[TrackEvidence] = []
    for track in tracks:
        expected_frames = max(
            1, round(track.duration_s / spectrogram.hop_duration_s) + 1
        )
        frame_coverage = min(1.0, len(track.points) / expected_frames)
        supports = tuple(
            _subband_track_support(track, subband) for subband in informative
        )
        supporting_subbands = sum(support.persistence >= 0.6 for support in supports)
        event_support = _track_event_support(track, events)
        persistence = sorted(
            (support.persistence for support in supports), reverse=True
        )
        top_two = [*persistence, 0.0, 0.0][:2]
        cross_support = frame_coverage * sum(top_two) / 2.0
        event_score = (
            event_support.phase_locking * event_support.one_cycle_interval_fraction
            if event_support is not None
            else 0.0
        )
        score = 0.8 * cross_support + 0.2 * event_score
        if frame_coverage < 0.7 or supporting_subbands < 2:
            label: EvidenceLabel = "limited"
        elif event_support is not None and event_support.periodic:
            label = "strong"
        else:
            label = "moderate"
        evidence.append(
            TrackEvidence(
                track_id=track.track_id,
                frame_coverage=frame_coverage,
                subbands=supports,
                supporting_subbands=supporting_subbands,
                events=event_support,
                score=score,
                label=label,
            )
        )
    return tuple(evidence)


def _subband_track_support(
    track: FrequencyTrack, subband: SubbandAnalysis
) -> SubbandTrackSupport:
    levels: list[float] = []
    contrasts: list[float] = []
    resolution = subband.spectrogram.frequency_resolution_hz
    half_width_hz = max(1.0, 2.0 * resolution)
    for point in track.points:
        time_index = int(np.argmin(np.abs(subband.spectrogram.times_s - point.time_s)))
        frequency_mask = (
            np.abs(subband.spectrogram.frequencies_hz - point.frequency_hz)
            <= half_width_hz
        )
        column = subband.spectrogram.psd_db[:, time_index]
        level = float(np.max(column[frequency_mask]))
        levels.append(level)
        contrasts.append(level - float(np.median(column)))
    supported = [
        level >= -25.0 and contrast >= 6.0
        for level, contrast in zip(levels, contrasts, strict=True)
    ]
    return SubbandTrackSupport(
        band=subband.band,
        persistence=float(np.mean(supported)),
        median_contrast_db=float(np.median(contrasts)),
    )


def _track_event_support(
    track: FrequencyTrack, events: EventAnalysis
) -> TrackEventSupport | None:
    times = np.array([point.time_s for point in track.points], dtype=np.float64)
    frequencies = np.array(
        [point.frequency_hz for point in track.points], dtype=np.float64
    )
    event_times = np.array(
        [
            event.time_s
            for event in events.events
            if times[0] <= event.time_s <= times[-1]
        ],
        dtype=np.float64,
    )
    if len(event_times) < 5 or event_times[-1] - event_times[0] < 1.0:
        return None
    phase_increments = (
        2.0 * np.pi * 0.5 * (frequencies[:-1] + frequencies[1:]) * np.diff(times)
    )
    phases = np.concatenate((
        np.array([0.0], dtype=np.float64),
        np.cumsum(phase_increments),
    ))
    event_phases = np.interp(event_times, times, phases)
    phase_locking = float(abs(np.mean(np.exp(1j * event_phases))))
    cycles = np.diff(event_phases) / (2.0 * np.pi)
    one_cycle_fraction = float(np.mean(np.abs(cycles - 1.0) <= 0.2))
    stationary_tolerance_hz = max(1.0, 0.05 * track.median_frequency_hz)
    periodicity_match = (
        track.maximum_frequency_hz - track.minimum_frequency_hz
        <= stationary_tolerance_hz
        and any(
            abs(periodicity.frequency_hz - track.median_frequency_hz)
            <= stationary_tolerance_hz
            for periodicity in events.periodicities
        )
    )
    return TrackEventSupport(
        event_count=len(event_phases),
        phase_locking=phase_locking,
        one_cycle_interval_fraction=one_cycle_fraction,
        periodic=(phase_locking >= 0.6 and one_cycle_fraction >= 0.6)
        or periodicity_match,
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
        distance=max(1, round(ENVELOPE_SAMPLE_RATE / MAX_MODULATION_HZ)),
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
    event_span_s = events[-1].time_s - events[0].time_s
    if len(events) >= 5 and event_span_s >= 1.0:
        impulse_train = np.zeros_like(envelope)
        event_indices = np.array(
            [round(event.time_s * ENVELOPE_SAMPLE_RATE) for event in events],
            dtype=np.int64,
        )
        impulse_train[event_indices] = np.array(
            [event.strength_z for event in events], dtype=np.float64
        )
        frequencies, spectrum_db, resolution = spectrum(impulse_train)
        peaks = detect_peaks(frequencies, spectrum_db, resolution)
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
            candidate_value = float(coherence_values[index])
            if math.isfinite(candidate_value):
                values[peak.frequency_hz].append(candidate_value)
    return {
        frequency_hz: float(np.median(candidate_values))
        for frequency_hz, candidate_values in values.items()
        if candidate_values
    }


def _global_peak_evidence(
    peaks: tuple[ModulationPeak, ...],
    informative: tuple[SubbandAnalysis, ...],
    events: EventAnalysis,
    coherence_by_frequency: dict[float, float],
    resolution_hz: float,
) -> tuple[GlobalPeakEvidence, ...]:
    evidence: list[GlobalPeakEvidence] = []
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
        phase_locking = _phase_locking(event_times, event_weights, peak.frequency_hz)
        event_rate_match = any(
            abs(periodicity.frequency_hz - peak.frequency_hz) <= tolerance_hz
            for periodicity in events.periodicities
        )
        coherence_value = coherence_by_frequency.get(peak.frequency_hz)
        score = (
            0.35 * min(1.0, peak.prominence_db / 30.0)
            + 0.25 * (supporting / len(informative) if informative else 0.0)
            + 0.15 * (coherence_value if coherence_value is not None else 0.0)
            + 0.25 * (phase_locking if phase_locking is not None else 0.0)
        )
        if supporting < 2:
            label: EvidenceLabel = "limited"
        elif event_rate_match and score >= 0.70:
            label = "strong"
        elif score >= 0.45:
            label = "moderate"
        else:
            label = "limited"
        evidence.append(
            GlobalPeakEvidence(
                frequency_hz=peak.frequency_hz,
                prominence_db=peak.prominence_db,
                informative_subbands=len(informative),
                supporting_subbands=supporting,
                median_subband_coherence=coherence_value,
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
    proposals: list[tuple[float, HarmonicFamily]] = []
    for anchor in peaks:
        for anchor_harmonic in range(1, 9):
            candidate_base_hz = anchor.frequency_hz / anchor_harmonic
            members: list[HarmonicMember] = []
            residual = 0.0
            for peak in peaks:
                harmonic = round(peak.frequency_hz / candidate_base_hz)
                if not 1 <= harmonic <= 8:
                    continue
                expected_hz = harmonic * candidate_base_hz
                tolerance_hz = max(2.0 * resolution_hz, 0.03 * peak.frequency_hz)
                error_hz = abs(peak.frequency_hz - expected_hz)
                if error_hz <= tolerance_hz:
                    members.append(
                        HarmonicMember(
                            frequency_hz=peak.frequency_hz, harmonic=harmonic
                        )
                    )
                    residual += error_hz
            if len(members) < 2:
                continue
            divisor = reduce(math.gcd, (member.harmonic for member in members))
            normalized_members = tuple(
                HarmonicMember(
                    frequency_hz=member.frequency_hz,
                    harmonic=member.harmonic // divisor,
                )
                for member in members
            )
            refined_base_hz = float(
                np.median([
                    member.frequency_hz / member.harmonic
                    for member in normalized_members
                ])
            )
            proposals.append((
                residual,
                HarmonicFamily(
                    base_frequency_hz=refined_base_hz,
                    members=normalized_members,
                ),
            ))

    families: list[HarmonicFamily] = []
    used: set[float] = set()
    ranked = sorted(
        proposals,
        key=lambda item: (
            -len(item[1].members),
            max(member.harmonic for member in item[1].members),
            item[0],
        ),
    )
    for _, family in ranked:
        frequencies = {member.frequency_hz for member in family.members}
        if frequencies.intersection(used):
            continue
        families.append(family)
        used.update(frequencies)
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
