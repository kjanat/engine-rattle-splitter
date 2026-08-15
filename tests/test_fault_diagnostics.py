"""Tests for cross-method rattle evidence."""

import json
import unittest
from itertools import pairwise

import numpy as np

from engine_rattle_splitter.fault_diagnostics import (
    DetectedEvent,
    EventAnalysis,
    EventPeriodicity,
    FrequencyTrack,
    RidgePoint,
    _analyze_events,
    _carrier_bands,
    _event_width_samples,
    _harmonic_families,
    _track_event_support,
    analyze_fault_evidence,
)
from engine_rattle_splitter.localization import _events_json
from engine_rattle_splitter.modulation import ModulationPeak, analyze

SAMPLE_RATE = 12_000
DURATION_S = 12.0


def _impact_train() -> np.ndarray:
    sample_count = round(SAMPLE_RATE * DURATION_S)
    times = np.arange(sample_count, dtype=np.float64) / SAMPLE_RATE
    signal = 1.5 * np.sin(2.0 * np.pi * 300.0 * times)
    rng = np.random.default_rng(42)
    burst_times = np.arange(round(0.018 * SAMPLE_RATE), dtype=np.float64) / SAMPLE_RATE
    burst = np.exp(-220.0 * burst_times) * (
        rng.normal(size=len(burst_times))
        + np.sin(2.0 * np.pi * 3_500.0 * burst_times)
        + np.sin(2.0 * np.pi * 5_000.0 * burst_times)
    )
    for start in range(2 * SAMPLE_RATE, 10 * SAMPLE_RATE, round(SAMPLE_RATE / 25.0)):
        end = min(start + len(burst), sample_count)
        signal[start:end] += burst[: end - start]
    return signal.astype(np.float32)


class FaultDiagnosticsTests(unittest.TestCase):
    def test_broadband_impacts_have_independent_support(self) -> None:
        samples = _impact_train()
        base = analyze(samples, SAMPLE_RATE)

        diagnostics = analyze_fault_evidence(
            samples,
            SAMPLE_RATE,
            base,
            crossover_hz=1800.0,
        )

        candidate = min(
            diagnostics.global_peak_evidence,
            key=lambda item: abs(item.frequency_hz - 25.0),
        )
        self.assertGreaterEqual(candidate.supporting_subbands, 2)
        self.assertTrue(
            candidate.event_rate_match or candidate.event_phase_locking is not None
        )
        self.assertTrue(diagnostics.harmonic_families)
        self.assertLessEqual(len(diagnostics.tracks), 12)
        self.assertTrue(
            any(evidence.label == "strong" for evidence in diagnostics.track_evidence)
        )
        self.assertIn(
            "evidence scores are uncalibrated rankings, not causal probabilities",
            diagnostics.warnings,
        )

    def test_silence_produces_no_false_evidence(self) -> None:
        samples = np.zeros(SAMPLE_RATE * 2, dtype=np.float32)
        base = analyze(samples, SAMPLE_RATE)

        diagnostics = analyze_fault_evidence(
            samples,
            SAMPLE_RATE,
            base,
            crossover_hz=1800.0,
        )

        self.assertEqual(diagnostics.global_peak_evidence, ())
        self.assertEqual(diagnostics.tracks, ())
        self.assertEqual(diagnostics.events.events, ())
        self.assertIn(
            "fewer than two carrier subbands contain useful energy",
            diagnostics.warnings,
        )
        self.assertIn(
            "no robust high-band burst events were detected",
            diagnostics.warnings,
        )

    def test_stationary_tone_beating_has_limited_evidence(self) -> None:
        times = (
            np.arange(round(SAMPLE_RATE * DURATION_S), dtype=np.float64) / SAMPLE_RATE
        )
        for left_hz, right_hz in ((3_500.0, 3_525.0), (3_990.0, 4_015.0)):
            with self.subTest(left_hz=left_hz):
                samples = (
                    np.sin(2.0 * np.pi * left_hz * times)
                    + np.sin(2.0 * np.pi * right_hz * times)
                ).astype(np.float32)
                base = analyze(samples, SAMPLE_RATE)
                diagnostics = analyze_fault_evidence(
                    samples,
                    SAMPLE_RATE,
                    base,
                    crossover_hz=1800.0,
                )
                candidate = min(
                    diagnostics.global_peak_evidence,
                    key=lambda item: abs(item.frequency_hz - 25.0),
                )
                self.assertEqual(candidate.label, "limited")

    def test_subbands_never_cross_requested_crossover(self) -> None:
        for crossover_hz in (5_000.0, 9_000.0, 13_000.0):
            with self.subTest(crossover_hz=crossover_hz):
                bands = _carrier_bands(48_000, crossover_hz)
                self.assertTrue(all(band.low_hz >= crossover_hz for band in bands))
                self.assertTrue(
                    all(left.high_hz <= right.low_hz for left, right in pairwise(bands))
                )

    def test_single_impulse_does_not_create_periodic_events(self) -> None:
        samples = np.zeros(round(SAMPLE_RATE * DURATION_S), dtype=np.float32)
        samples[round(6.0 * SAMPLE_RATE)] = 1.0
        base = analyze(samples, SAMPLE_RATE)

        diagnostics = analyze_fault_evidence(
            samples, SAMPLE_RATE, base, crossover_hz=1800.0
        )

        self.assertEqual(diagnostics.events.periodicities, ())
        self.assertFalse(
            any(
                evidence.events is not None and evidence.events.periodic
                for evidence in diagnostics.track_evidence
            )
        )

    def test_harmonics_infer_missing_fundamental(self) -> None:
        peaks = tuple(
            ModulationPeak(frequency_hz=frequency, level_db=0.0, prominence_db=20.0)
            for frequency in (22.8, 34.2, 45.6)
        )

        families = _harmonic_families(peaks, resolution_hz=0.1)

        self.assertTrue(families)
        self.assertAlmostEqual(families[0].base_frequency_hz, 11.4, delta=0.2)
        self.assertEqual(
            tuple(member.harmonic for member in families[0].members), (2, 3, 4)
        )

    def test_event_corroboration_supports_100_hz_limit(self) -> None:
        track_times = np.arange(0.0, 1.26, 0.25)
        track = FrequencyTrack(
            track_id=1,
            points=tuple(
                RidgePoint(time_s=float(time_s), frequency_hz=100.0, level_db=0.0)
                for time_s in track_times.tolist()
            ),
            duration_s=1.25,
            median_frequency_hz=100.0,
            minimum_frequency_hz=100.0,
            maximum_frequency_hz=100.0,
            slope_hz_per_s=0.0,
        )
        events = EventAnalysis(
            events=tuple(
                DetectedEvent(time_s=float(time_s), strength_z=8.0, width_ms=2.5)
                for time_s in np.arange(0.0, 1.251, 0.01).tolist()
            ),
            periodicities=(
                EventPeriodicity(
                    frequency_hz=100.0,
                    prominence_db=20.0,
                    autocorrelation=0.9,
                ),
            ),
            median_width_ms=2.5,
            interval_cv=0.0,
        )

        support = _track_event_support(track, events)
        subharmonic_track = FrequencyTrack(
            track_id=2,
            points=tuple(
                RidgePoint(time_s=float(time_s), frequency_hz=25.0, level_db=0.0)
                for time_s in track_times.tolist()
            ),
            duration_s=1.25,
            median_frequency_hz=25.0,
            minimum_frequency_hz=25.0,
            maximum_frequency_hz=25.0,
            slope_hz_per_s=0.0,
        )
        subharmonic_support = _track_event_support(subharmonic_track, events)

        self.assertIsNotNone(support)
        if support is not None:
            self.assertTrue(support.periodic)
        self.assertIsNotNone(subharmonic_support)
        if subharmonic_support is not None:
            self.assertFalse(subharmonic_support.periodic)

    def test_event_detection_preserves_near_limit_train(self) -> None:
        rng = np.random.default_rng(13)
        envelope = 1.0 + 0.001 * rng.normal(size=1_200)
        starts = np.arange(20, 1_180, 4)
        envelope[starts] += 1.0

        events = _analyze_events(envelope.astype(np.float64))

        self.assertEqual(len(events.events), len(starts))

    def test_event_width_tracks_envelope_decay(self) -> None:
        short = np.zeros(400, dtype=np.float64)
        long = np.zeros(400, dtype=np.float64)
        short[20:] = np.exp(-np.arange(380, dtype=np.float64) / 8.0)
        long[20:] = np.exp(-np.arange(380, dtype=np.float64) / 40.0)

        short_width = _event_width_samples(short, 20, len(short))
        long_width = _event_width_samples(long, 20, len(long))

        self.assertGreater(long_width, short_width * 3.0)

    def test_long_event_width_and_measurements_are_not_censored(self) -> None:
        envelope = np.zeros(3_000, dtype=np.float64)
        envelope[20:] = np.exp(-np.arange(2_980, dtype=np.float64) / 800.0)

        width_samples = _event_width_samples(envelope, 20, len(envelope))
        payload = _events_json(
            EventAnalysis(
                events=(DetectedEvent(time_s=0.05, strength_z=8.0, width_ms=1_200.0),),
                periodicities=(),
                median_width_ms=1_200.0,
                interval_cv=None,
            )
        )

        self.assertGreater(width_samples, 400.0)
        serialized = json.dumps(payload)
        self.assertIn('"width_ms": 1200.0', serialized)
        self.assertIn('"strength_z": 8.0', serialized)


if __name__ == "__main__":
    unittest.main()
