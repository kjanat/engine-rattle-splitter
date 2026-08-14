"""Tests for localization report synthesis."""

import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from engine_rattle_splitter.fault_diagnostics import (
    CandidateEvidence,
    EventAnalysis,
    FaultDiagnostics,
    FrequencyTrack,
    RidgePoint,
    analyze_fault_evidence,
)
from engine_rattle_splitter.localization import (
    _camera_targets,
    _control_deltas,
    write_camera_targets,
)
from engine_rattle_splitter.localization import run as run_localization
from engine_rattle_splitter.media import CaptureRate, TimelineAlignment
from engine_rattle_splitter.modulation import analyze
from engine_rattle_splitter.orders import RpmPoint, RpmTrace, analyze_orders

SAMPLE_RATE = 12_000


def _candidate(frequency_hz: float, label: str = "strong") -> CandidateEvidence:
    if label not in {"limited", "moderate", "strong"}:
        raise ValueError(label)
    evidence_label = (
        "strong"
        if label == "strong"
        else "moderate"
        if label == "moderate"
        else "limited"
    )
    return CandidateEvidence(
        frequency_hz=frequency_hz,
        prominence_db=20.0,
        informative_subbands=3,
        supporting_subbands=3,
        median_subband_coherence=0.8,
        longest_track_s=2.0,
        event_phase_locking=None,
        event_rate_match=False,
        score=0.8,
        label=evidence_label,
    )


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


class LocalizationTests(unittest.TestCase):
    def test_camera_targets_use_alignment_and_capture_rate(self) -> None:
        diagnostics = FaultDiagnostics(
            subbands=(),
            consensus_spectrogram=None,
            tracks=(_track(1, 20.0), _track(2, 80.0)),
            events=EventAnalysis(
                events=(), periodicities=(), median_width_ms=None, interval_cv=None
            ),
            harmonic_families=(),
            candidates=(_candidate(20.0), _candidate(80.0, "limited")),
            warnings=(),
        )

        targets = _camera_targets(
            diagnostics,
            frequency_bin_hz=0.5,
            capture_rate=CaptureRate(
                fps=120.0, provenance="physical override", warning=None
            ),
            alignment=TimelineAlignment(
                offset_s=0.25, provenance="explicit", correlation=None
            ),
        )

        self.assertEqual(len(targets), 1)
        video_start_s = targets[0].video_start_s
        if video_start_s is None:
            self.fail("explicit alignment must produce video time")
        self.assertAlmostEqual(video_start_s - targets[0].audio_start_s, 0.25)
        self.assertEqual(targets[0].status, "well sampled")

    def test_camera_target_matches_candidate_inside_chirped_track(self) -> None:
        track = FrequencyTrack(
            track_id=1,
            points=(
                RidgePoint(time_s=1.0, frequency_hz=21.5, level_db=-3.0),
                RidgePoint(time_s=1.33, frequency_hz=26.5, level_db=-3.0),
                RidgePoint(time_s=1.66, frequency_hz=30.0, level_db=-3.0),
                RidgePoint(time_s=2.0, frequency_hz=38.5, level_db=-3.0),
            ),
            duration_s=1.0,
            median_frequency_hz=30.0,
            minimum_frequency_hz=21.5,
            maximum_frequency_hz=38.5,
            slope_hz_per_s=17.0,
        )
        diagnostics = FaultDiagnostics(
            subbands=(),
            consensus_spectrogram=None,
            tracks=(track,),
            events=EventAnalysis(
                events=(), periodicities=(), median_width_ms=None, interval_cv=None
            ),
            harmonic_families=(),
            candidates=(_candidate(26.5, "limited"), _candidate(28.25)),
            warnings=(),
        )

        targets = _camera_targets(diagnostics, 0.5, None, None)

        self.assertEqual(len(targets), 1)

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

    def test_chirped_order_remains_a_camera_target(self) -> None:
        duration_s = 12.0
        times = (
            np.arange(round(SAMPLE_RATE * duration_s), dtype=np.float64) / SAMPLE_RATE
        )
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
        base = analyze(samples, SAMPLE_RATE)
        diagnostics = analyze_fault_evidence(
            samples, SAMPLE_RATE, base, crossover_hz=1800.0
        )
        speed = RpmTrace(
            kind="trace",
            points=(
                RpmPoint(time_s=0.0, rpm=600.0),
                RpmPoint(time_s=duration_s, rpm=1_200.0),
            ),
            source=Path("rpm.csv"),
        )
        orders = analyze_orders(
            diagnostics.tracks,
            diagnostics.consensus_spectrogram or base.spectrogram,
            speed,
        )

        targets = _camera_targets(
            diagnostics, base.spectrogram.frequency_resolution_hz, None, None
        )

        self.assertTrue(targets)
        self.assertTrue(any(fit.consistent for fit in orders.fits))
        fitted = next(fit for fit in orders.fits if fit.consistent)
        self.assertAlmostEqual(fitted.fitted_order or 0.0, 2.0, delta=0.1)

    def test_control_delta_uses_fractional_envelope_power(self) -> None:
        times = np.arange(SAMPLE_RATE * 8, dtype=np.float64) / SAMPLE_RATE
        carrier = np.sin(2.0 * np.pi * 4_000.0 * times)
        active_samples = (
            (1.0 + 0.8 * np.sin(2.0 * np.pi * 25.0 * times)) * carrier
        ).astype(np.float32)
        control_samples = (
            (1.0 + 0.05 * np.sin(2.0 * np.pi * 25.0 * times)) * carrier
        ).astype(np.float32)
        active = analyze(active_samples, SAMPLE_RATE)
        with patch(
            "engine_rattle_splitter.localization.decode", return_value=control_samples
        ):
            deltas = _control_deltas(
                active,
                (_candidate(25.0),),
                Path("control.wav"),
                SAMPLE_RATE,
                1800.0,
                4,
            )

        self.assertEqual(len(deltas), 1)
        self.assertGreater(deltas[0].active_minus_control_db, 20.0)

    def test_camera_csv_has_declared_timeline_fields(self) -> None:
        diagnostics = FaultDiagnostics(
            subbands=(),
            consensus_spectrogram=None,
            tracks=(_track(1, 20.0),),
            events=EventAnalysis(
                events=(), periodicities=(), median_width_ms=None, interval_cv=None
            ),
            harmonic_families=(),
            candidates=(_candidate(20.0),),
            warnings=(),
        )
        targets = _camera_targets(diagnostics, 0.5, None, None)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "camera.csv"
            write_camera_targets(path, targets)
            with path.open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))

        self.assertEqual(len(rows), 1)
        self.assertIn("audio_start_s", rows[0])
        self.assertIn("video_start_s", rows[0])
        self.assertEqual(rows[0]["video_start_s"], "")
        self.assertEqual(rows[0]["status"], "unknown")


if __name__ == "__main__":
    unittest.main()
