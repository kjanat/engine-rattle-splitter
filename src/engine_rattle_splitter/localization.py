"""End-to-end motorcycle rattle fault-localization reporting."""

import csv
import json
import math
import subprocess
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Literal

import numpy as np
from scipy.signal import welch

from . import modulation
from .audio_io import decode
from .fault_diagnostics import (
    CandidateEvidence,
    EventAnalysis,
    FaultDiagnostics,
    FrequencyTrack,
    analyze_fault_evidence,
)
from .media import (
    CaptureRate,
    TimelineAlignment,
    VideoMetadata,
    align_audio_tracks,
    explicit_alignment,
    probe_video,
    resolve_capture_rate,
)
from .orders import (
    OrderAnalysis,
    SpeedReference,
    analyze_orders,
    fixed_rpm,
    load_rpm_trace,
    order_curves,
)

type CameraStatus = Literal["well sampled", "marginal", "alias risk", "unknown"]


@dataclass(frozen=True)
class ControlDelta:
    frequency_hz: float
    active_minus_control_db: float


@dataclass(frozen=True)
class CameraTarget:
    track_id: int
    audio_start_s: float
    audio_end_s: float
    video_start_s: float | None
    video_end_s: float | None
    frequency_low_hz: float
    frequency_high_hz: float
    minimum_unaliased_fps_exclusive: float
    recommended_fps: float
    capture_fps: float | None
    status: CameraStatus
    evidence_score: float


@dataclass(frozen=True)
class FaultLocalizationReport:
    input_path: Path
    sample_rate: int
    crossover_hz: float
    filter_order: int
    modulation: modulation.ModulationResult
    diagnostics: FaultDiagnostics
    order_analysis: OrderAnalysis | None
    video_metadata: VideoMetadata | None
    capture_rate: CaptureRate | None
    alignment: TimelineAlignment | None
    control_path: Path | None
    control_deltas: tuple[ControlDelta, ...]
    camera_targets: tuple[CameraTarget, ...]
    warnings: tuple[str, ...]


def run(
    *,
    input_path: Path,
    sample_rate: int,
    output_png: Path,
    crossover_hz: float,
    filter_order: int,
    rpm: float | None = None,
    rpm_trace_path: Path | None = None,
    reference_orders: tuple[float, ...],
    control_path: Path | None = None,
    video_path: Path | None = None,
    capture_fps: float | None = None,
    video_start_offset_s: float | None = None,
    json_output: Path | None = None,
    camera_target_output: Path | None = None,
) -> FaultLocalizationReport:
    """Analyze, corroborate, render, report, and write optional sidecars."""
    _validate_output_paths(
        input_path=input_path,
        control_path=control_path,
        rpm_trace_path=rpm_trace_path,
        video_path=video_path,
        output_png=output_png,
        json_output=json_output,
        camera_target_output=camera_target_output,
    )
    if rpm is not None and rpm_trace_path is not None:
        msg = "fixed RPM and RPM trace are mutually exclusive"
        raise ValueError(msg)
    audio = decode(input_path, sample_rate, channels=1)
    base = modulation.analyze(
        audio,
        sample_rate,
        crossover_hz=crossover_hz,
        filter_order=filter_order,
    )
    diagnostics = analyze_fault_evidence(
        audio,
        sample_rate,
        base,
        crossover_hz=crossover_hz,
        filter_order=filter_order,
    )
    speed = _speed_reference(rpm, rpm_trace_path)
    order_analysis = (
        analyze_orders(
            diagnostics.tracks,
            diagnostics.consensus_spectrogram or base.spectrogram,
            speed,
            reference_orders=reference_orders,
        )
        if speed is not None
        else None
    )
    metadata = probe_video(video_path) if video_path is not None else None
    capture_rate = resolve_capture_rate(metadata, capture_fps)
    warnings = list(diagnostics.warnings)
    if metadata is not None:
        warnings.extend(metadata.warnings)
    if capture_rate is not None and capture_rate.warning is not None:
        warnings.append(capture_rate.warning)
    alignment = _alignment(
        audio,
        video_path,
        sample_rate,
        video_start_offset_s,
        warnings,
    )
    control_deltas = _control_deltas(
        base,
        diagnostics.candidates,
        control_path,
        sample_rate,
        crossover_hz,
        filter_order,
    )
    camera_targets = _camera_targets(
        diagnostics,
        base.spectrogram.frequency_resolution_hz,
        capture_rate,
        alignment,
    )
    report = FaultLocalizationReport(
        input_path=input_path,
        sample_rate=sample_rate,
        crossover_hz=crossover_hz,
        filter_order=filter_order,
        modulation=base,
        diagnostics=diagnostics,
        order_analysis=order_analysis,
        video_metadata=metadata,
        capture_rate=capture_rate,
        alignment=alignment,
        control_path=control_path,
        control_deltas=control_deltas,
        camera_targets=camera_targets,
        warnings=tuple(dict.fromkeys(warnings)),
    )
    curves = (
        order_curves(
            base.spectrogram.times_s,
            order_analysis.speed,
            order_analysis.reference_orders,
        )
        if order_analysis is not None
        else ()
    )
    modulation.render(
        base,
        input_name=input_path.name,
        output_png=output_png,
        crossover_hz=crossover_hz,
        rpm=rpm,
        video_fps=capture_rate.fps if capture_rate is not None else None,
        track_overlays=tuple(
            (
                track.track_id,
                np.array([point.time_s for point in track.points], dtype=np.float64),
                np.array(
                    [point.frequency_hz for point in track.points], dtype=np.float64
                ),
            )
            for track in diagnostics.tracks
        ),
        event_times_s=tuple(event.time_s for event in diagnostics.events.events),
        order_overlays=curves,
        order_map=(
            (
                order_analysis.order_map.times_s,
                order_analysis.order_map.orders,
                order_analysis.order_map.psd_db,
                order_analysis.order_map.valid,
            )
            if order_analysis is not None
            else None
        ),
        subband_spectra=tuple(
            (
                subband.band.name,
                subband.frequencies_hz,
                subband.spectrum_db,
                subband.informative,
            )
            for subband in diagnostics.subbands
        ),
    )
    modulation.print_report(
        base,
        rpm=rpm,
        video_fps=capture_rate.fps if capture_rate is not None else None,
    )
    print_report(report)
    if json_output is not None:
        write_json(json_output, report)
    if camera_target_output is not None:
        write_camera_targets(camera_target_output, camera_targets)
    return report


def print_report(report: FaultLocalizationReport) -> None:
    print("\nCorroborated candidates:")
    if not report.diagnostics.candidates:
        print("  none")
    for candidate in report.diagnostics.candidates:
        coherence_text = (
            f" coherence={candidate.median_subband_coherence:.2f}"
            if candidate.median_subband_coherence is not None
            else ""
        )
        event_text = (
            f" event-lock={candidate.event_phase_locking:.2f}"
            if candidate.event_phase_locking is not None
            else ""
        )
        print(
            f"  {candidate.frequency_hz:6.1f} Hz  {candidate.label:8s} "
            f"score={candidate.score:.2f} subbands="
            f"{candidate.supporting_subbands}/{candidate.informative_subbands}"
            f"{coherence_text}{event_text}"
        )
    if report.diagnostics.harmonic_families:
        print("\nHarmonic relationships:")
        for family in report.diagnostics.harmonic_families:
            members = ", ".join(
                f"{member.harmonic}x={member.frequency_hz:.1f} Hz"
                for member in family.members
            )
            print(f"  base ~{family.base_frequency_hz:.1f} Hz: {members}")
    print(f"\nBurst events: {len(report.diagnostics.events.events)}")
    for periodicity in report.diagnostics.events.periodicities:
        print(
            f"  {periodicity.frequency_hz:.1f} Hz  "
            f"autocorrelation={periodicity.autocorrelation:.2f}"
        )
    if report.diagnostics.tracks:
        print("\nFrequency tracks:")
        for track in report.diagnostics.tracks:
            print(
                f"  track {track.track_id}: {track.points[0].time_s:.2f}-"
                f"{track.points[-1].time_s:.2f} s, "
                f"{track.minimum_frequency_hz:.1f}-{track.maximum_frequency_hz:.1f} Hz, "
                f"slope={track.slope_hz_per_s:+.2f} Hz/s"
            )
    if report.order_analysis is not None:
        print("\nOrder fits:")
        for fit in report.order_analysis.fits:
            if fit.fitted_order is None:
                print(f"  track {fit.track_id}: insufficient RPM coverage")
                continue
            print(
                f"  track {fit.track_id}: {fit.fitted_order:.2f}x "
                f"MAD={fit.order_mad:.2f}x match={fit.match_fraction:.0%} "
                f"consistent={fit.consistent}"
            )
    if report.capture_rate is not None:
        print(
            f"\nCapture rate: {report.capture_rate.fps:g} fps "
            f"({report.capture_rate.provenance})"
        )
    if report.alignment is not None:
        print(
            f"Timeline: video_time = audio_time + {report.alignment.offset_s:+.3f} s "
            f"({report.alignment.provenance})"
        )
    if report.camera_targets:
        print("\nCamera motion targets:")
        for target in report.camera_targets:
            timeline = (
                f"video {target.video_start_s:.2f}-{target.video_end_s:.2f} s"
                if target.video_start_s is not None and target.video_end_s is not None
                else f"audio {target.audio_start_s:.2f}-{target.audio_end_s:.2f} s"
            )
            print(
                f"  track {target.track_id}: {timeline}, "
                f"{target.frequency_low_hz:.1f}-"
                f"{target.frequency_high_hz:.1f} Hz, {target.status}; "
                f"Nyquist >{target.minimum_unaliased_fps_exclusive:.1f} fps, "
                f"recommended {target.recommended_fps:.1f} fps"
            )
    if report.control_deltas:
        print("\nActive minus control modulation:")
        for delta in report.control_deltas:
            print(
                f"  {delta.frequency_hz:6.1f} Hz  "
                f"{delta.active_minus_control_db:+.1f} dB"
            )
    if report.warnings:
        print("\nWarnings:")
        for warning in report.warnings:
            print(f"  - {warning}")


def write_json(path: Path, report: FaultLocalizationReport) -> None:
    payload = _json_payload(report)
    path.parent.mkdir(parents=True, exist_ok=True)
    _ = path.write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(f"wrote {path}")


def write_camera_targets(path: Path, targets: tuple[CameraTarget, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "track_id",
            "audio_start_s",
            "audio_end_s",
            "video_start_s",
            "video_end_s",
            "frequency_low_hz",
            "frequency_high_hz",
            "minimum_unaliased_fps_exclusive",
            "recommended_fps",
            "capture_fps",
            "status",
            "evidence_score",
        ])
        for target in targets:
            writer.writerow([
                target.track_id,
                f"{target.audio_start_s:.6f}",
                f"{target.audio_end_s:.6f}",
                "" if target.video_start_s is None else f"{target.video_start_s:.6f}",
                "" if target.video_end_s is None else f"{target.video_end_s:.6f}",
                f"{target.frequency_low_hz:.6f}",
                f"{target.frequency_high_hz:.6f}",
                f"{target.minimum_unaliased_fps_exclusive:.6f}",
                f"{target.recommended_fps:.6f}",
                "" if target.capture_fps is None else f"{target.capture_fps:.6f}",
                target.status,
                f"{target.evidence_score:.6f}",
            ])
    print(f"wrote {path}")


def _speed_reference(
    rpm: float | None, rpm_trace_path: Path | None
) -> SpeedReference | None:
    if rpm is not None:
        return fixed_rpm(rpm)
    if rpm_trace_path is not None:
        return load_rpm_trace(rpm_trace_path)
    return None


def _alignment(
    source_audio: np.ndarray,
    video_path: Path | None,
    sample_rate: int,
    explicit_offset_s: float | None,
    warnings: list[str],
) -> TimelineAlignment | None:
    if explicit_offset_s is not None:
        return explicit_alignment(explicit_offset_s)
    if video_path is None:
        return None
    try:
        video_audio = decode(video_path, sample_rate, channels=1)
        alignment = align_audio_tracks(source_audio, video_audio, sample_rate)
    except (subprocess.CalledProcessError, ValueError) as error:
        warnings.append(f"automatic audio/video alignment unavailable: {error}")
        return None
    return alignment


def _control_deltas(
    active: modulation.ModulationResult,
    candidates: tuple[CandidateEvidence, ...],
    control_path: Path | None,
    sample_rate: int,
    crossover_hz: float,
    filter_order: int,
) -> tuple[ControlDelta, ...]:
    if control_path is None:
        return ()
    control_audio = decode(control_path, sample_rate, channels=1)
    control = modulation.analyze(
        control_audio,
        sample_rate,
        crossover_hz=crossover_hz,
        filter_order=filter_order,
    )
    active_frequencies, active_power = _fractional_psd(active.envelope)
    control_frequencies, control_power = _fractional_psd(control.envelope)
    deltas: list[ControlDelta] = []
    for candidate in candidates:
        active_value = float(
            np.interp(candidate.frequency_hz, active_frequencies, active_power)
        )
        control_value = float(
            np.interp(candidate.frequency_hz, control_frequencies, control_power)
        )
        deltas.append(
            ControlDelta(
                frequency_hz=candidate.frequency_hz,
                active_minus_control_db=10.0
                * math.log10(
                    (active_value + np.finfo(np.float64).tiny)
                    / (control_value + np.finfo(np.float64).tiny)
                ),
            )
        )
    return tuple(deltas)


def _fractional_psd(envelope: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    scale = max(float(np.median(envelope)), np.finfo(np.float64).eps)
    fractional = (envelope - float(np.median(envelope))) / scale
    frequencies, power = welch(
        fractional,
        fs=modulation.ENVELOPE_SAMPLE_RATE,
        window="hann_periodic",
        nperseg=min(len(fractional), 8 * modulation.ENVELOPE_SAMPLE_RATE),
        noverlap=min(len(fractional) // 2, 4 * modulation.ENVELOPE_SAMPLE_RATE),
        scaling="density",
        average="median",
    )
    return frequencies, power


def _camera_targets(
    diagnostics: FaultDiagnostics,
    frequency_bin_hz: float,
    capture_rate: CaptureRate | None,
    alignment: TimelineAlignment | None,
) -> tuple[CameraTarget, ...]:
    evidence_by_frequency = diagnostics.candidates
    offset_s = alignment.offset_s if alignment is not None else 0.0
    targets: list[CameraTarget] = []
    for track in diagnostics.tracks:
        evidence = _nearest_evidence(track, evidence_by_frequency)
        if evidence is None or evidence.label == "limited":
            continue
        points = track.points
        segment_start = 0
        statuses: list[CameraStatus] = [
            _camera_status(
                point.frequency_hz + frequency_bin_hz,
                capture_rate.fps if capture_rate is not None else None,
            )
            for point in points
        ]
        for index in range(1, len(points) + 1):
            if index < len(points) and statuses[index] == statuses[segment_start]:
                continue
            segment = points[segment_start:index]
            low_hz = min(point.frequency_hz for point in segment)
            high_hz = max(point.frequency_hz for point in segment) + frequency_bin_hz
            start_s = segment[0].time_s - modulation.SPECTROGRAM_HOP_S / 2.0
            end_s = segment[-1].time_s + modulation.SPECTROGRAM_HOP_S / 2.0
            targets.append(
                CameraTarget(
                    track_id=track.track_id,
                    audio_start_s=start_s,
                    audio_end_s=end_s,
                    video_start_s=(
                        start_s + offset_s if alignment is not None else None
                    ),
                    video_end_s=(end_s + offset_s if alignment is not None else None),
                    frequency_low_hz=low_hz,
                    frequency_high_hz=high_hz,
                    minimum_unaliased_fps_exclusive=2.0 * high_hz,
                    recommended_fps=4.0 * high_hz,
                    capture_fps=capture_rate.fps if capture_rate is not None else None,
                    status=statuses[segment_start],
                    evidence_score=evidence.score if evidence is not None else 0.0,
                )
            )
            segment_start = index
    return tuple(targets)


def _camera_status(
    upper_frequency_hz: float, capture_fps: float | None
) -> CameraStatus:
    if capture_fps is None:
        return "unknown"
    if capture_fps >= 4.0 * upper_frequency_hz:
        return "well sampled"
    if capture_fps > 2.0 * upper_frequency_hz:
        return "marginal"
    return "alias risk"


def _nearest_evidence(
    track: FrequencyTrack, candidates: tuple[CandidateEvidence, ...]
) -> CandidateEvidence | None:
    supported = tuple(
        candidate for candidate in candidates if candidate.label != "limited"
    )
    if not supported:
        return None
    nearest = min(
        supported,
        key=lambda candidate: (
            _distance_to_track(candidate.frequency_hz, track),
            -candidate.score,
        ),
    )
    tolerance_hz = max(2.0, 0.05 * nearest.frequency_hz)
    return (
        nearest
        if _distance_to_track(nearest.frequency_hz, track) <= tolerance_hz
        else None
    )


def _distance_to_track(frequency_hz: float, track: FrequencyTrack) -> float:
    if len(track.points) == 1:
        return abs(frequency_hz - track.points[0].frequency_hz)
    return min(
        _distance_to_interval(
            frequency_hz,
            left.frequency_hz,
            right.frequency_hz,
        )
        for left, right in pairwise(track.points)
    )


def _distance_to_interval(value: float, first: float, second: float) -> float:
    lower, upper = sorted((first, second))
    if lower <= value <= upper:
        return 0.0
    return min(abs(value - lower), abs(value - upper))


def _validate_output_paths(
    *,
    input_path: Path,
    control_path: Path | None,
    rpm_trace_path: Path | None,
    video_path: Path | None,
    output_png: Path,
    json_output: Path | None,
    camera_target_output: Path | None,
) -> None:
    sources = {
        path.resolve()
        for path in (input_path, control_path, rpm_trace_path, video_path)
        if path is not None
    }
    outputs = [
        path.resolve()
        for path in (output_png, json_output, camera_target_output)
        if path is not None
    ]
    if len(outputs) != len(set(outputs)):
        msg = "plot, JSON, and camera-target outputs must use distinct paths"
        raise ValueError(msg)
    collisions = sources.intersection(outputs)
    if collisions:
        collision = min(collisions, key=str)
        msg = f"output path would overwrite an input: {collision}"
        raise ValueError(msg)


def _json_payload(report: FaultLocalizationReport) -> dict[str, object]:
    return {
        "schema_version": 1,
        "input": {
            "path": str(report.input_path),
            "sample_rate_hz": report.sample_rate,
            "crossover_hz": report.crossover_hz,
            "filter_order": report.filter_order,
        },
        "global_peaks": [
            {
                "frequency_hz": peak.frequency_hz,
                "prominence_db": peak.prominence_db,
            }
            for peak in report.modulation.peaks
        ],
        "candidates": [
            {
                "frequency_hz": candidate.frequency_hz,
                "prominence_db": candidate.prominence_db,
                "supporting_subbands": candidate.supporting_subbands,
                "informative_subbands": candidate.informative_subbands,
                "median_subband_coherence": candidate.median_subband_coherence,
                "longest_track_s": candidate.longest_track_s,
                "event_phase_locking": candidate.event_phase_locking,
                "event_rate_match": candidate.event_rate_match,
                "evidence_score": candidate.score,
                "evidence_label": candidate.label,
            }
            for candidate in report.diagnostics.candidates
        ],
        "subbands": [
            {
                "name": subband.band.name,
                "low_hz": subband.band.low_hz,
                "high_hz": subband.band.high_hz,
                "carrier_rms": subband.carrier_rms,
                "informative": subband.informative,
                "peak_frequencies_hz": [peak.frequency_hz for peak in subband.peaks],
            }
            for subband in report.diagnostics.subbands
        ],
        "tracks": [
            {
                "track_id": track.track_id,
                "duration_s": track.duration_s,
                "median_frequency_hz": track.median_frequency_hz,
                "frequency_span_hz": [
                    track.minimum_frequency_hz,
                    track.maximum_frequency_hz,
                ],
                "slope_hz_per_s": track.slope_hz_per_s,
                "points": [
                    {
                        "time_s": point.time_s,
                        "frequency_hz": point.frequency_hz,
                        "level_db": point.level_db,
                    }
                    for point in track.points
                ],
            }
            for track in report.diagnostics.tracks
        ],
        "events": _events_json(report.diagnostics.events),
        "harmonic_families": [
            {
                "base_frequency_hz": family.base_frequency_hz,
                "members": [
                    {
                        "harmonic": member.harmonic,
                        "frequency_hz": member.frequency_hz,
                    }
                    for member in family.members
                ],
            }
            for family in report.diagnostics.harmonic_families
        ],
        "orders": _orders_json(report.order_analysis),
        "video": _video_json(
            report.video_metadata, report.capture_rate, report.alignment
        ),
        "camera_targets": [
            {
                "track_id": target.track_id,
                "audio_start_s": target.audio_start_s,
                "audio_end_s": target.audio_end_s,
                "video_start_s": target.video_start_s,
                "video_end_s": target.video_end_s,
                "frequency_low_hz": target.frequency_low_hz,
                "frequency_high_hz": target.frequency_high_hz,
                "minimum_unaliased_fps_exclusive": target.minimum_unaliased_fps_exclusive,
                "recommended_fps": target.recommended_fps,
                "capture_fps": target.capture_fps,
                "status": target.status,
                "evidence_score": target.evidence_score,
            }
            for target in report.camera_targets
        ],
        "control": {
            "path": str(report.control_path)
            if report.control_path is not None
            else None,
            "candidate_deltas_db": [
                {
                    "frequency_hz": delta.frequency_hz,
                    "active_minus_control_db": delta.active_minus_control_db,
                }
                for delta in report.control_deltas
            ],
        },
        "warnings": list(report.warnings),
    }


def _orders_json(analysis: OrderAnalysis | None) -> object:
    if analysis is None:
        return None
    speed: dict[str, object]
    if analysis.speed.kind == "fixed":
        speed = {"kind": "fixed", "rpm": analysis.speed.rpm}
    else:
        speed = {
            "kind": "trace",
            "source": str(analysis.speed.source),
            "points": [
                {"time_s": point.time_s, "rpm": point.rpm}
                for point in analysis.speed.points
            ],
        }
    return {
        "speed_reference": speed,
        "reference_orders": list(analysis.reference_orders),
        "track_fits": [
            {
                "track_id": fit.track_id,
                "rpm_coverage": fit.rpm_coverage,
                "fitted_order": fit.fitted_order,
                "order_mad": fit.order_mad,
                "nearest_reference_order": fit.nearest_reference_order,
                "match_fraction": fit.match_fraction,
                "frequency_rpm_correlation": fit.frequency_rpm_correlation,
                "consistent": fit.consistent,
            }
            for fit in analysis.fits
        ],
        "order_map": {
            "times_s": analysis.order_map.times_s.tolist(),
            "orders": analysis.order_map.orders.tolist(),
            "psd_db": analysis.order_map.psd_db.tolist(),
            "valid": analysis.order_map.valid.tolist(),
        },
    }


def _events_json(events: EventAnalysis) -> dict[str, object]:
    return {
        "count": len(events.events),
        "median_width_ms": events.median_width_ms,
        "interval_cv": events.interval_cv,
        "measurements": [
            {
                "time_s": event.time_s,
                "width_ms": event.width_ms,
                "strength_z": event.strength_z,
            }
            for event in events.events
        ],
        "periodicities": [
            {
                "frequency_hz": periodicity.frequency_hz,
                "prominence_db": periodicity.prominence_db,
                "autocorrelation": periodicity.autocorrelation,
            }
            for periodicity in events.periodicities
        ],
    }


def _video_json(
    metadata: VideoMetadata | None,
    capture_rate: CaptureRate | None,
    alignment: TimelineAlignment | None,
) -> object:
    if metadata is None and capture_rate is None and alignment is None:
        return None
    return {
        "metadata": (
            {
                "path": str(metadata.path),
                "average_fps": metadata.average_fps,
                "nominal_fps": metadata.nominal_fps,
                "time_base_s": metadata.time_base_s,
                "duration_s": metadata.duration_s,
                "frame_count": metadata.frame_count,
            }
            if metadata is not None
            else None
        ),
        "capture_rate": (
            {
                "fps": capture_rate.fps,
                "provenance": capture_rate.provenance,
            }
            if capture_rate is not None
            else None
        ),
        "alignment": (
            {
                "offset_s": alignment.offset_s,
                "provenance": alignment.provenance,
                "correlation": alignment.correlation,
                "convention": "video_time_s = audio_time_s + offset_s",
            }
            if alignment is not None
            else None
        ),
    }
