"""Tests for localization report synthesis."""

import csv
import json
import subprocess
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from typing import Literal
from unittest.mock import patch

import numpy as np

from engine_rattle_splitter.fault_diagnostics import (
    FaultDiagnostics,
    FrequencyBand,
    FrequencyTrack,
    GlobalPeakEvidence,
    RidgePoint,
    SubbandTrackSupport,
    TrackEvidence,
    analyze_fault_evidence,
)
from engine_rattle_splitter.localization import (
    FaultLocalizationReport,
    TrackControlContrast,
    TrackDiagnostic,
    _camera_targets,
    _diagnose_tracks,
    _probe_metadata,
    _track_control_contrasts,
    write_camera_targets,
    write_json,
)
from engine_rattle_splitter.localization import run as run_localization
from engine_rattle_splitter.media import TimelineAlignment
from engine_rattle_splitter.modulation import (
    ModulationResult,
    ModulationSpectrogram,
    analyze,
)
from engine_rattle_splitter.orders import RpmPoint, RpmTrace, analyze_orders, fixed_rpm

SAMPLE_RATE = 12_000


def _track(track_id: int, frequency_hz: float) -> FrequencyTrack:
    points = (
        RidgePoint(time_s=1.0, frequency_hz=frequency_hz, level_db=-3.0),
        RidgePoint(time_s=2.0, frequency_hz=frequency_hz + 0.5, level_db=-4.0),
    )
    return FrequencyTrack(
        track_id=track_id,
        points=points,
        duration_s=1.0,
        median_frequency_hz=frequency_hz + 0.25,
        minimum_frequency_hz=frequency_hz,
        maximum_frequency_hz=frequency_hz + 0.5,
        slope_hz_per_s=0.5,
    )


def _track_evidence(
    track_id: int,
    label: Literal["limited", "moderate", "strong"] = "moderate",
) -> TrackEvidence:
    support = SubbandTrackSupport(
        band=FrequencyBand(name="2-4 kHz", low_hz=2_000.0, high_hz=4_000.0),
        persistence=0.9,
        median_contrast_db=20.0,
    )
    return TrackEvidence(
        track_id=track_id,
        frame_coverage=1.0,
        subbands=(support, support),
        supporting_subbands=2,
        events=None,
        score=0.72,
        label=label,
    )


def _diagnostic(
    track_id: int,
    priority: Literal["exclude", "candidate", "corroborated"] = "candidate",
) -> TrackDiagnostic:
    return TrackDiagnostic(
        track_id=track_id,
        signal_label="moderate" if priority != "exclude" else "limited",
        audio_evidence_score=0.72,
        order_consistent=None,
        control=None,
        priority=priority,
    )


def _spectrogram() -> ModulationSpectrogram:
    return ModulationSpectrogram(
        times_s=np.array([1.0, 2.0], dtype=np.float64),
        frequencies_hz=np.arange(5.0, 100.5, 0.5, dtype=np.float64),
        psd_db=np.zeros((191, 2), dtype=np.float64),
        window_duration_s=2.0,
        hop_duration_s=0.25,
        frequency_resolution_hz=0.5,
    )


def _chirped_analysis() -> tuple[np.ndarray, ModulationResult, FaultDiagnostics]:
    duration_s = 12.0
    times = np.arange(round(SAMPLE_RATE * duration_s), dtype=np.float64) / SAMPLE_RATE
    frequencies = 20.0 + 20.0 * times / duration_s
    phase = 2.0 * np.pi * np.cumsum(frequencies) / SAMPLE_RATE
    amplitude = 1.0 + 0.8 * np.sin(phase)
    samples = (
        amplitude
        * (
            np.sin(2.0 * np.pi * 3_000.0 * times)
            + np.sin(2.0 * np.pi * 5_000.0 * times)
        )
    ).astype(np.float32)
    base = replace(analyze(samples, SAMPLE_RATE), peaks=())
    diagnostics = analyze_fault_evidence(
        samples, SAMPLE_RATE, base, crossover_hz=1800.0
    )
    return samples, base, diagnostics


class LocalizationTests(unittest.TestCase):
    def test_camera_targets_use_window_support_and_physical_rate(self) -> None:
        tracks = (_track(1, 20.0), _track(2, 80.0))
        targets = _camera_targets(
            tracks,
            (_diagnostic(1), _diagnostic(2, "exclude")),
            _spectrogram(),
            audio_duration_s=4.0,
            physical_capture_fps=120.0,
            alignment=TimelineAlignment(
                offset_s=0.25, provenance="explicit", correlation=None
            ),
        )

        self.assertEqual(len(targets), 1)
        self.assertEqual(targets[0].audio_start_s, 0.0)
        self.assertEqual(targets[0].audio_end_s, 3.0)
        self.assertEqual(targets[0].video_start_s, 0.25)
        self.assertEqual(targets[0].status, "well sampled")

    def test_metadata_only_camera_status_is_unknown(self) -> None:
        targets = _camera_targets(
            (_track(1, 20.0),),
            (_diagnostic(1),),
            _spectrogram(),
            audio_duration_s=4.0,
            physical_capture_fps=None,
            alignment=None,
        )

        self.assertEqual(targets[0].status, "unknown")
        self.assertIsNone(targets[0].capture_fps)

    def test_camera_targets_intersect_video_timeline(self) -> None:
        common = (
            (_track(1, 20.0),),
            (_diagnostic(1),),
            _spectrogram(),
        )
        outside = _camera_targets(
            *common,
            audio_duration_s=4.0,
            physical_capture_fps=None,
            alignment=TimelineAlignment(
                offset_s=-10.0, provenance="explicit", correlation=None
            ),
        )
        partial = _camera_targets(
            *common,
            audio_duration_s=4.0,
            physical_capture_fps=None,
            alignment=TimelineAlignment(
                offset_s=-1.0, provenance="explicit", correlation=None
            ),
            video_duration_s=1.5,
        )

        self.assertEqual(outside, ())
        self.assertEqual(partial[0].audio_start_s, 1.0)
        self.assertEqual(partial[0].audio_end_s, 2.5)
        self.assertEqual(partial[0].video_start_s, 0.0)
        self.assertEqual(partial[0].video_end_s, 1.5)

    def test_track_diagnostics_reject_unknown_control_id(self) -> None:
        control = TrackControlContrast(
            track_id=2,
            median_active_minus_control_db=10.0,
            compared_points=1,
        )

        with self.assertRaisesRegex(ValueError, "known track IDs"):
            _ = _diagnose_tracks((_track_evidence(1),), None, (control,))

    def test_video_probe_timeout_degrades_to_warning(self) -> None:
        warnings: list[str] = []
        with patch(
            "engine_rattle_splitter.localization.probe_video",
            side_effect=subprocess.TimeoutExpired("ffprobe", 30),
        ):
            metadata = _probe_metadata(Path("video.mp4"), warnings)

        self.assertIsNone(metadata)
        self.assertTrue(any("video metadata unavailable" in item for item in warnings))

    def test_chirped_track_needs_no_global_peak_for_camera_target(self) -> None:
        _, base, diagnostics = _chirped_analysis()
        diagnostics = replace(diagnostics, global_peak_evidence=())
        track_diagnostics = _diagnose_tracks(diagnostics.track_evidence, None, ())
        track_spectrogram = diagnostics.consensus_spectrogram or base.spectrogram

        targets = _camera_targets(
            diagnostics.tracks,
            track_diagnostics,
            track_spectrogram,
            audio_duration_s=12.0,
            physical_capture_fps=None,
            alignment=None,
        )

        self.assertTrue(targets)
        self.assertTrue(any(item.priority == "candidate" for item in track_diagnostics))

    def test_chirped_order_fit_is_track_keyed(self) -> None:
        _, base, diagnostics = _chirped_analysis()
        speed = RpmTrace(
            kind="trace",
            points=(
                RpmPoint(time_s=0.0, rpm=600.0),
                RpmPoint(time_s=12.0, rpm=1_200.0),
            ),
            source=Path("rpm.csv"),
        )
        orders = analyze_orders(
            diagnostics.tracks,
            diagnostics.consensus_spectrogram or base.spectrogram,
            speed,
        )
        track_diagnostics = _diagnose_tracks(diagnostics.track_evidence, orders, ())

        self.assertTrue(any(fit.consistent for fit in orders.fits))
        self.assertTrue(
            any(item.order_consistent is True for item in track_diagnostics)
        )
        consistent_fit = next(fit for fit in orders.fits if fit.consistent)
        control = TrackControlContrast(
            track_id=consistent_fit.track_id,
            median_active_minus_control_db=10.0,
            compared_points=10,
        )
        corroborated = _diagnose_tracks(diagnostics.track_evidence, orders, (control,))
        self.assertEqual(
            next(
                item.priority
                for item in corroborated
                if item.track_id == consistent_fit.track_id
            ),
            "corroborated",
        )

        fixed_orders = analyze_orders(
            diagnostics.tracks,
            diagnostics.consensus_spectrogram or base.spectrogram,
            fixed_rpm(900.0),
        )
        fixed_diagnostics = _diagnose_tracks(
            diagnostics.track_evidence, fixed_orders, (control,)
        )
        self.assertNotEqual(
            next(
                item.priority
                for item in fixed_diagnostics
                if item.track_id == consistent_fit.track_id
            ),
            "corroborated",
        )

    def test_track_control_uses_moving_ridge_power(self) -> None:
        times = np.arange(SAMPLE_RATE * 8, dtype=np.float64) / SAMPLE_RATE
        carrier = np.sin(2.0 * np.pi * 4_000.0 * times)
        modulation_hz = 20.0 + 10.0 * times / 8.0
        phase = 2.0 * np.pi * np.cumsum(modulation_hz) / SAMPLE_RATE
        active_samples = ((1.0 + 0.8 * np.sin(phase)) * carrier).astype(np.float32)
        control_samples = ((1.0 + 0.05 * np.sin(phase)) * carrier).astype(np.float32)
        active = analyze(active_samples, SAMPLE_RATE)
        track_times = np.arange(1.0, 7.25, 0.25)
        track_frequencies = 20.0 + 10.0 * track_times / 8.0
        track = FrequencyTrack(
            track_id=1,
            points=tuple(
                RidgePoint(
                    time_s=float(time_s),
                    frequency_hz=float(frequency_hz),
                    level_db=0.0,
                )
                for time_s, frequency_hz in zip(
                    track_times, track_frequencies, strict=True
                )
            ),
            duration_s=6.0,
            median_frequency_hz=float(np.median(track_frequencies)),
            minimum_frequency_hz=float(np.min(track_frequencies)),
            maximum_frequency_hz=float(np.max(track_frequencies)),
            slope_hz_per_s=1.25,
        )
        with patch(
            "engine_rattle_splitter.localization.decode", return_value=control_samples
        ):
            contrasts = _track_control_contrasts(
                active,
                (track,),
                Path("control.wav"),
                SAMPLE_RATE,
                1800.0,
                4,
            )

        self.assertEqual(len(contrasts), 1)
        self.assertGreater(contrasts[0].median_active_minus_control_db, 20.0)

    def test_rejects_output_collisions_before_decoding(self) -> None:
        with self.assertRaisesRegex(ValueError, "overwrite an input"):
            _ = run_localization(
                input_path=Path("input.wav"),
                sample_rate=SAMPLE_RATE,
                output_png=Path("input.wav"),
                crossover_hz=1800.0,
                filter_order=4,
                reference_orders=(1.0, 2.0),
            )
        with self.assertRaisesRegex(ValueError, "distinct paths"):
            _ = run_localization(
                input_path=Path("input.wav"),
                sample_rate=SAMPLE_RATE,
                output_png=Path("report.out"),
                crossover_hz=1800.0,
                filter_order=4,
                reference_orders=(1.0, 2.0),
                json_output=Path("report.out"),
            )

    def test_camera_csv_and_json_publish_track_contract(self) -> None:
        _, base, diagnostics = _chirped_analysis()
        track_diagnostics = _diagnose_tracks(diagnostics.track_evidence, None, ())
        track_spectrogram = diagnostics.consensus_spectrogram or base.spectrogram
        targets = _camera_targets(
            diagnostics.tracks,
            track_diagnostics,
            track_spectrogram,
            audio_duration_s=12.0,
            physical_capture_fps=None,
            alignment=None,
        )
        nonfinite_evidence = replace(diagnostics.track_evidence[0], score=float("nan"))
        diagnostics = replace(
            diagnostics,
            track_evidence=(
                nonfinite_evidence,
                *diagnostics.track_evidence[1:],
            ),
            global_peak_evidence=(
                GlobalPeakEvidence(
                    frequency_hz=25.0,
                    prominence_db=20.0,
                    informative_subbands=2,
                    supporting_subbands=2,
                    median_subband_coherence=float("nan"),
                    event_phase_locking=None,
                    event_rate_match=False,
                    score=float("nan"),
                    label="moderate",
                ),
            ),
        )
        track_diagnostics = (
            replace(track_diagnostics[0], audio_evidence_score=float("nan")),
            *track_diagnostics[1:],
        )
        report = FaultLocalizationReport(
            input_path=Path("input.wav"),
            sample_rate=SAMPLE_RATE,
            crossover_hz=1800.0,
            filter_order=4,
            modulation=base,
            diagnostics=diagnostics,
            order_analysis=None,
            video_metadata=None,
            capture_rate=None,
            alignment=None,
            control_path=None,
            track_diagnostics=track_diagnostics,
            camera_targets=targets,
            warnings=diagnostics.warnings,
        )
        with tempfile.TemporaryDirectory() as directory:
            csv_path = Path(directory) / "camera.csv"
            json_path = Path(directory) / "report.json"
            write_camera_targets(csv_path, targets)
            write_json(json_path, report)
            with csv_path.open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            payload = json.loads(json_path.read_text(encoding="utf-8"))

        self.assertEqual(payload["schema_version"], 2)
        for section in ("orders", "video", "camera_targets", "control"):
            self.assertIn(section, payload)
        self.assertIn("diagnostic", payload["tracks"][0])
        self.assertIsNone(
            payload["tracks"][0]["signal_evidence"]["audio_evidence_score"]
        )
        self.assertIsNone(payload["global_peak_evidence"][0]["audio_evidence_score"])
        self.assertIn("audio_evidence_score", rows[0])
        self.assertEqual(rows[0]["status"], "unknown")


if __name__ == "__main__":
    unittest.main()
