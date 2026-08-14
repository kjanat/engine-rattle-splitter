"""Tests for cross-method rattle evidence."""

import json
import unittest
from itertools import pairwise

import numpy as np

from engine_rattle_splitter.fault_diagnostics import (
    DetectedEvent,
    EventAnalysis,
    _carrier_bands,
    _event_width_samples,
    analyze_fault_evidence,
)
from engine_rattle_splitter.localization import _events_json
from engine_rattle_splitter.modulation import analyze

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
            diagnostics.candidates,
            key=lambda item: abs(item.frequency_hz - 25.0),
        )
        self.assertGreaterEqual(candidate.supporting_subbands, 2)
        self.assertTrue(
            candidate.event_rate_match or candidate.event_phase_locking is not None
        )
        self.assertTrue(diagnostics.harmonic_families)
        self.assertLessEqual(len(diagnostics.tracks), 12)

    def test_silence_produces_no_false_evidence(self) -> None:
        samples = np.zeros(SAMPLE_RATE * 2, dtype=np.float32)
        base = analyze(samples, SAMPLE_RATE)

        diagnostics = analyze_fault_evidence(
            samples,
            SAMPLE_RATE,
            base,
            crossover_hz=1800.0,
        )

        self.assertEqual(diagnostics.candidates, ())
        self.assertEqual(diagnostics.tracks, ())
        self.assertEqual(diagnostics.events.events, ())

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
                    diagnostics.candidates,
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
