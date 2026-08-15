"""End-to-end motorcycle rattle fault-localization reporting."""

import csv
import json
import math
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from scipy.signal import spectrogram as signal_spectrogram

from . import modulation
from .audio_io import decode
from .fault_diagnostics import (
    EventAnalysis,
    FaultDiagnostics,
    FrequencyTrack,
    TrackEvidence,
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
type DiagnosticPriority = Literal["exclude", "candidate", "corroborated"]


@dataclass(frozen=True)
class TrackControlContrast:
    track_id: int
    median_active_minus_control_db: float
    compared_points: int


@dataclass(frozen=True)
class TrackDiagnostic:
    track_id: int
    signal_label: Literal["limited", "moderate", "strong"]
    audio_evidence_score: float
    order_consistent: bool | None
    control: TrackControlContrast | None
    priority: DiagnosticPriority


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
    audio_evidence_score: float
    priority: DiagnosticPriority


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
    track_diagnostics: tuple[TrackDiagnostic, ...]
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
    warnings = list(diagnostics.warnings)
    metadata = _probe_metadata(video_path, warnings)
    capture_rate = resolve_capture_rate(metadata, capture_fps)
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
    control_contrasts = _track_control_contrasts(
        base,
        diagnostics.tracks,
        control_path,
        sample_rate,
        crossover_hz,
        filter_order,
    )
    track_diagnostics = _diagnose_tracks(
        diagnostics.track_evidence,
        order_analysis,
        control_contrasts,
    )
    track_spectrogram = diagnostics.consensus_spectrogram or base.spectrogram
    physical_capture_fps = (
        capture_rate.fps
        if capture_rate is not None and capture_rate.provenance == "physical override"
        else None
    )
    camera_targets = _camera_targets(
        diagnostics.tracks,
        track_diagnostics,
        track_spectrogram,
        len(base.envelope) / modulation.ENVELOPE_SAMPLE_RATE,
        physical_capture_fps,
        alignment,
        video_duration_s=metadata.duration_s if metadata is not None else None,
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
        track_diagnostics=track_diagnostics,
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
        video_fps=physical_capture_fps,
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
                order_analysis.order_map.order_resolution,
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
        video_fps=physical_capture_fps,
    )
    print_report(report)
    if json_output is not None:
        write_json(json_output, report)
    if camera_target_output is not None:
        write_camera_targets(camera_target_output, camera_targets)
    return report


def print_report(report: FaultLocalizationReport) -> None:
    print("\nGlobal-peak audio evidence:")
    if not report.diagnostics.global_peak_evidence:
        print("  none")
    for candidate in report.diagnostics.global_peak_evidence:
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
        track_evidence = {
            evidence.track_id: evidence
            for evidence in report.diagnostics.track_evidence
        }
        track_diagnostics = {
            diagnostic.track_id: diagnostic for diagnostic in report.track_diagnostics
        }
        for track in report.diagnostics.tracks:
            evidence = track_evidence[track.track_id]
            diagnostic = track_diagnostics[track.track_id]
            print(
                f"  track {track.track_id}: {track.points[0].time_s:.2f}-"
                f"{track.points[-1].time_s:.2f} s, "
                f"{track.minimum_frequency_hz:.1f}-{track.maximum_frequency_hz:.1f} Hz, "
                f"slope={track.slope_hz_per_s:+.2f} Hz/s, "
                f"signal={evidence.label}, priority={diagnostic.priority}"
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
        rate_name = (
            "Capture rate"
            if report.capture_rate.provenance == "physical override"
            else "Container playback rate"
        )
        print(
            f"\n{rate_name}: {report.capture_rate.fps:g} fps "
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
    controls = tuple(
        diagnostic.control
        for diagnostic in report.track_diagnostics
        if diagnostic.control is not None
    )
    if controls:
        print("\nTrack-local active minus control modulation:")
        for contrast in controls:
            print(
                f"  track {contrast.track_id}: "
                f"{contrast.median_active_minus_control_db:+.1f} dB"
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
            "audio_evidence_score",
            "priority",
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
                f"{target.audio_evidence_score:.6f}",
                target.priority,
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


def _probe_metadata(
    video_path: Path | None, warnings: list[str]
) -> VideoMetadata | None:
    if video_path is None:
        return None
    try:
        return probe_video(video_path)
    except (
        subprocess.CalledProcessError,
        subprocess.TimeoutExpired,
        ValueError,
    ) as error:
        warnings.append(f"video metadata unavailable: {error}")
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
    except (
        subprocess.CalledProcessError,
        subprocess.TimeoutExpired,
        ValueError,
    ) as error:
        warnings.append(f"automatic audio/video alignment unavailable: {error}")
        return None
    return alignment


def _track_control_contrasts(
    active: modulation.ModulationResult,
    tracks: tuple[FrequencyTrack, ...],
    control_path: Path | None,
    sample_rate: int,
    crossover_hz: float,
    filter_order: int,
) -> tuple[TrackControlContrast, ...]:
    if control_path is None:
        return ()
    control_audio = decode(control_path, sample_rate, channels=1)
    control = modulation.analyze(
        control_audio,
        sample_rate,
        crossover_hz=crossover_hz,
        filter_order=filter_order,
    )
    active_times, active_frequencies, active_power = _fractional_power_map(
        active.envelope
    )
    _, control_frequencies, control_power = _fractional_power_map(control.envelope)
    active_tube_power = _tube_power_map(active_frequencies, active_power)
    control_tube_power = _tube_power_map(control_frequencies, control_power)
    control_reference = np.percentile(control_tube_power, 90.0, axis=1)
    contrasts: list[TrackControlContrast] = []
    for track in tracks:
        point_deltas: list[float] = []
        for point in track.points:
            active_time_index = int(np.argmin(np.abs(active_times - point.time_s)))
            active_value = float(
                np.interp(
                    point.frequency_hz,
                    active_frequencies,
                    active_tube_power[:, active_time_index],
                )
            )
            control_value = float(
                np.interp(
                    point.frequency_hz,
                    control_frequencies,
                    control_reference,
                )
            )
            point_deltas.append(
                10.0
                * math.log10(
                    (active_value + np.finfo(np.float64).tiny)
                    / (control_value + np.finfo(np.float64).tiny)
                )
            )
        contrasts.append(
            TrackControlContrast(
                track_id=track.track_id,
                median_active_minus_control_db=float(np.median(point_deltas)),
                compared_points=len(point_deltas),
            )
        )
    return tuple(contrasts)


def _fractional_power_map(
    envelope: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    median = float(np.median(envelope))
    scale = max(median, np.finfo(np.float64).eps)
    fractional = (envelope - median) / scale
    window_samples = min(
        len(fractional),
        round(modulation.SPECTROGRAM_WINDOW_S * modulation.ENVELOPE_SAMPLE_RATE),
    )
    hop_samples = min(
        window_samples,
        round(modulation.SPECTROGRAM_HOP_S * modulation.ENVELOPE_SAMPLE_RATE),
    )
    frequencies, times, power = signal_spectrogram(
        fractional,
        fs=modulation.ENVELOPE_SAMPLE_RATE,
        window="hann",
        nperseg=window_samples,
        noverlap=window_samples - hop_samples,
        detrend="constant",
        scaling="density",
        mode="psd",
    )
    band = (frequencies >= modulation.MIN_MODULATION_HZ) & (
        frequencies <= modulation.MAX_MODULATION_HZ
    )
    return times, frequencies[band], power[band, :]


def _tube_power_map(frequencies: np.ndarray, power: np.ndarray) -> np.ndarray:
    resolution_hz = (
        float(frequencies[1] - frequencies[0]) if len(frequencies) > 1 else 1.0
    )
    half_width_hz = max(1.0, 2.0 * resolution_hz)
    tube_power = np.empty_like(power, dtype=np.float64)
    for frequency_index, frequency_hz in enumerate(frequencies.tolist()):
        inside = np.abs(frequencies - frequency_hz) <= half_width_hz
        tube_power[frequency_index, :] = (
            np.sum(power[inside, :], axis=0) * resolution_hz
        )
    return tube_power


def _diagnose_tracks(
    evidence: tuple[TrackEvidence, ...],
    order_analysis: OrderAnalysis | None,
    controls: tuple[TrackControlContrast, ...],
) -> tuple[TrackDiagnostic, ...]:
    track_ids = tuple(item.track_id for item in evidence)
    if len(track_ids) != len(set(track_ids)):
        msg = "track evidence IDs must be unique"
        raise ValueError(msg)
    if order_analysis is not None and {
        fit.track_id for fit in order_analysis.fits
    } != set(track_ids):
        msg = "order-fit IDs must match track evidence IDs"
        raise ValueError(msg)
    control_ids = tuple(control.track_id for control in controls)
    if len(control_ids) != len(set(control_ids)) or not set(control_ids).issubset(
        track_ids
    ):
        msg = "control IDs must be unique known track IDs"
        raise ValueError(msg)
    order_by_track = (
        {fit.track_id: fit for fit in order_analysis.fits}
        if order_analysis is not None and order_analysis.speed.kind == "trace"
        else {}
    )
    control_by_track = {control.track_id: control for control in controls}
    diagnostics: list[TrackDiagnostic] = []
    for track_evidence in evidence:
        fit = order_by_track.get(track_evidence.track_id)
        order_consistent = fit.consistent if fit is not None else None
        control = control_by_track.get(track_evidence.track_id)
        if track_evidence.label == "limited":
            priority: DiagnosticPriority = "exclude"
        elif track_evidence.label == "strong" or (
            order_consistent is True
            and control is not None
            and control.median_active_minus_control_db > 6.0
        ):
            priority = "corroborated"
        else:
            priority = "candidate"
        diagnostics.append(
            TrackDiagnostic(
                track_id=track_evidence.track_id,
                signal_label=track_evidence.label,
                audio_evidence_score=track_evidence.score,
                order_consistent=order_consistent,
                control=control,
                priority=priority,
            )
        )
    return tuple(diagnostics)


def _camera_targets(
    tracks: tuple[FrequencyTrack, ...],
    diagnostics: tuple[TrackDiagnostic, ...],
    spectrogram: modulation.ModulationSpectrogram,
    audio_duration_s: float,
    physical_capture_fps: float | None,
    alignment: TimelineAlignment | None,
    video_duration_s: float | None = None,
) -> tuple[CameraTarget, ...]:
    diagnostic_by_track = {
        diagnostic.track_id: diagnostic for diagnostic in diagnostics
    }
    offset_s = alignment.offset_s if alignment is not None else 0.0
    targets: list[CameraTarget] = []
    frequency_half_width_hz = spectrogram.frequency_resolution_hz / 2.0
    time_half_width_s = spectrogram.window_duration_s / 2.0
    for track in tracks:
        diagnostic = diagnostic_by_track[track.track_id]
        if diagnostic.priority == "exclude":
            continue
        points = track.points
        segment_start = 0
        statuses: list[CameraStatus] = [
            _camera_status(
                point.frequency_hz + frequency_half_width_hz,
                physical_capture_fps,
            )
            for point in points
        ]
        for index in range(1, len(points) + 1):
            if index < len(points) and statuses[index] == statuses[segment_start]:
                continue
            segment = points[segment_start:index]
            low_hz = max(
                modulation.MIN_MODULATION_HZ,
                min(point.frequency_hz for point in segment) - frequency_half_width_hz,
            )
            high_hz = min(
                modulation.MAX_MODULATION_HZ,
                max(point.frequency_hz for point in segment) + frequency_half_width_hz,
            )
            start_s = max(0.0, segment[0].time_s - time_half_width_s)
            end_s = min(audio_duration_s, segment[-1].time_s + time_half_width_s)
            video_start_s: float | None = None
            video_end_s: float | None = None
            if alignment is not None:
                aligned_interval = _intersect_video_interval(
                    start_s,
                    end_s,
                    offset_s,
                    video_duration_s,
                )
                if aligned_interval is None:
                    segment_start = index
                    continue
                start_s, end_s, video_start_s, video_end_s = aligned_interval
            targets.append(
                CameraTarget(
                    track_id=track.track_id,
                    audio_start_s=start_s,
                    audio_end_s=end_s,
                    video_start_s=video_start_s,
                    video_end_s=video_end_s,
                    frequency_low_hz=low_hz,
                    frequency_high_hz=high_hz,
                    minimum_unaliased_fps_exclusive=2.0 * high_hz,
                    recommended_fps=4.0 * high_hz,
                    capture_fps=physical_capture_fps,
                    status=statuses[segment_start],
                    audio_evidence_score=diagnostic.audio_evidence_score,
                    priority=diagnostic.priority,
                )
            )
            segment_start = index
    return tuple(targets)


def _intersect_video_interval(
    audio_start_s: float,
    audio_end_s: float,
    offset_s: float,
    video_duration_s: float | None,
) -> tuple[float, float, float, float] | None:
    video_start_s = max(0.0, audio_start_s + offset_s)
    video_end_s = audio_end_s + offset_s
    if video_duration_s is not None:
        video_end_s = min(video_end_s, video_duration_s)
    if video_end_s <= video_start_s:
        return None
    return (
        video_start_s - offset_s,
        video_end_s - offset_s,
        video_start_s,
        video_end_s,
    )


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
    track_evidence = {
        evidence.track_id: evidence for evidence in report.diagnostics.track_evidence
    }
    track_diagnostics = {
        diagnostic.track_id: diagnostic for diagnostic in report.track_diagnostics
    }
    return {
        "schema_version": 2,
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
        "global_peak_evidence": [
            {
                "frequency_hz": candidate.frequency_hz,
                "prominence_db": candidate.prominence_db,
                "supporting_subbands": candidate.supporting_subbands,
                "informative_subbands": candidate.informative_subbands,
                "median_subband_coherence": _finite_optional(
                    candidate.median_subband_coherence
                ),
                "event_phase_locking": _finite_optional(candidate.event_phase_locking),
                "event_rate_match": candidate.event_rate_match,
                "audio_evidence_score": _finite_optional(candidate.score),
                "signal_label": candidate.label,
            }
            for candidate in report.diagnostics.global_peak_evidence
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
                "slope_hz_per_s": _finite_optional(track.slope_hz_per_s),
                "signal_evidence": _track_evidence_json(track_evidence[track.track_id]),
                "diagnostic": _track_diagnostic_json(track_diagnostics[track.track_id]),
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
                "audio_evidence_score": _finite_optional(target.audio_evidence_score),
                "priority": target.priority,
            }
            for target in report.camera_targets
        ],
        "control": {
            "path": str(report.control_path)
            if report.control_path is not None
            else None,
            "track_contrasts": [
                {
                    "track_id": diagnostic.control.track_id,
                    "median_active_minus_control_db": _finite_optional(
                        diagnostic.control.median_active_minus_control_db
                    ),
                    "compared_points": diagnostic.control.compared_points,
                }
                for diagnostic in report.track_diagnostics
                if diagnostic.control is not None
            ],
        },
        "warnings": list(report.warnings),
    }


def _orders_json(analysis: OrderAnalysis | None) -> dict[str, object] | None:
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
                "fitted_order": _finite_optional(fit.fitted_order),
                "order_mad": _finite_optional(fit.order_mad),
                "nearest_reference_order": _finite_optional(
                    fit.nearest_reference_order
                ),
                "match_fraction": _finite_optional(fit.match_fraction),
                "frequency_rpm_correlation": _finite_optional(
                    fit.frequency_rpm_correlation
                ),
                "consistent": fit.consistent,
            }
            for fit in analysis.fits
        ],
        "order_map": {
            "times_s": analysis.order_map.times_s.tolist(),
            "orders": analysis.order_map.orders.tolist(),
            "order_resolution": analysis.order_map.order_resolution,
            "median_psd_db_per_order": _order_medians(analysis),
            "valid_fraction_per_order": np.round(
                np.mean(analysis.order_map.valid, axis=1), 4
            ).tolist(),
        },
    }


def _events_json(events: EventAnalysis) -> dict[str, object]:
    return {
        "count": len(events.events),
        "median_width_ms": _finite_optional(events.median_width_ms),
        "interval_cv": _finite_optional(events.interval_cv),
        "measurements": [
            {
                "time_s": event.time_s,
                "width_ms": event.width_ms,
                "strength_z": _finite_optional(event.strength_z),
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
) -> dict[str, object] | None:
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
                "correlation": _finite_optional(alignment.correlation),
                "convention": "video_time_s = audio_time_s + offset_s",
            }
            if alignment is not None
            else None
        ),
    }


def _track_evidence_json(evidence: TrackEvidence) -> dict[str, object]:
    return {
        "frame_coverage": _finite_optional(evidence.frame_coverage),
        "supporting_subbands": evidence.supporting_subbands,
        "audio_evidence_score": _finite_optional(evidence.score),
        "signal_label": evidence.label,
        "subbands": [
            {
                "name": support.band.name,
                "persistence": _finite_optional(support.persistence),
                "median_contrast_db": _finite_optional(support.median_contrast_db),
            }
            for support in evidence.subbands
        ],
        "events": (
            {
                "count": evidence.events.event_count,
                "phase_locking": _finite_optional(evidence.events.phase_locking),
                "one_cycle_interval_fraction": _finite_optional(
                    evidence.events.one_cycle_interval_fraction
                ),
                "periodic": evidence.events.periodic,
            }
            if evidence.events is not None
            else None
        ),
    }


def _track_diagnostic_json(diagnostic: TrackDiagnostic) -> dict[str, object]:
    return {
        "priority": diagnostic.priority,
        "order_consistent": diagnostic.order_consistent,
        "control_active_minus_db": (
            _finite_optional(diagnostic.control.median_active_minus_control_db)
            if diagnostic.control is not None
            else None
        ),
    }


def _order_medians(analysis: OrderAnalysis) -> list[float]:
    masked = np.ma.masked_where(~analysis.order_map.valid, analysis.order_map.psd_db)
    medians = np.ma.median(masked, axis=1).filled(modulation.SPECTROGRAM_FLOOR_DB)
    return np.round(medians, 1).tolist()


def _finite_optional(value: float | None) -> float | None:
    if value is None or not math.isfinite(value):
        return None
    return value
