"""Tests for rattle-envelope modulation analysis."""

import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from engine_rattle_splitter.audio_io import Float32Array
from engine_rattle_splitter.cli import _run_modulation
from engine_rattle_splitter.modulation import (
    InsufficientAudioError,
    _resample_envelope,
    analyze,
    order_ratio,
    render,
)

SAMPLE_RATE = 12_000
DURATION_S = 12.0
CARRIER_HZ = 4_000.0


def _am_signal(modulations: tuple[tuple[float, float], ...]) -> Float32Array:
    times = np.arange(round(SAMPLE_RATE * DURATION_S), dtype=np.float64) / SAMPLE_RATE
    amplitude = np.ones_like(times)
    for frequency_hz, depth in modulations:
        amplitude += depth * np.sin(2.0 * np.pi * frequency_hz * times)
    carrier = amplitude * np.sin(2.0 * np.pi * CARRIER_HZ * times)
    engine = 2.0 * np.sin(2.0 * np.pi * 300.0 * times)
    return (carrier + engine).astype(np.float32)


class ModulationTests(unittest.TestCase):
    def test_detects_known_modulation_above_strong_engine_tone(self) -> None:
        result = analyze(_am_signal(((27.4, 0.8),)), SAMPLE_RATE)

        error = min(abs(peak.frequency_hz - 27.4) for peak in result.peaks)
        self.assertLessEqual(error, result.frequency_resolution_hz)

    def test_detects_multiple_modulation_frequencies(self) -> None:
        result = analyze(_am_signal(((23.0, 0.45), (47.0, 0.35))), SAMPLE_RATE)
        frequencies = tuple(peak.frequency_hz for peak in result.peaks)

        self.assertTrue(any(abs(frequency - 23.0) <= 0.15 for frequency in frequencies))
        self.assertTrue(any(abs(frequency - 47.0) <= 0.15 for frequency in frequencies))

    def test_detects_modulation_at_search_band_edges(self) -> None:
        result = analyze(_am_signal(((5.0, 0.45), (100.0, 0.45))), SAMPLE_RATE)
        frequencies = tuple(peak.frequency_hz for peak in result.peaks)

        self.assertTrue(any(abs(frequency - 5.0) <= 0.15 for frequency in frequencies))
        self.assertTrue(
            any(abs(frequency - 100.0) <= 0.15 for frequency in frequencies)
        )

    def test_two_stationary_tones_produce_beating_component(self) -> None:
        times = (
            np.arange(round(SAMPLE_RATE * DURATION_S), dtype=np.float64) / SAMPLE_RATE
        )
        samples = (
            np.sin(2.0 * np.pi * CARRIER_HZ * times)
            + np.sin(2.0 * np.pi * (CARRIER_HZ + 27.4) * times)
        ).astype(np.float32)

        result = analyze(samples, SAMPLE_RATE)

        self.assertTrue(
            any(abs(peak.frequency_hz - 27.4) <= 0.15 for peak in result.peaks)
        )

    def test_silence_has_finite_result_without_peaks(self) -> None:
        samples = np.zeros(round(SAMPLE_RATE * 2.0), dtype=np.float32)
        result = analyze(samples, SAMPLE_RATE)

        self.assertEqual(result.peaks, ())
        self.assertTrue(bool(np.all(np.isfinite(result.envelope))))
        self.assertTrue(bool(np.all(np.isfinite(result.spectrum_db))))

    def test_resampling_preserves_constant_envelope_edges(self) -> None:
        envelope = np.ones(201, dtype=np.float64)

        resampled = _resample_envelope(envelope, sample_rate=201)

        self.assertEqual(len(resampled), 400)
        self.assertTrue(bool(np.allclose(resampled, 1.0, atol=0.001)))

    def test_order_ratio_uses_fixed_crank_frequency(self) -> None:
        self.assertAlmostEqual(order_ratio(27.4, 1800.0), 27.4 / 30.0)

    def test_invalid_rpm_is_rejected(self) -> None:
        for rpm in (0.0, -1.0, math.nan, math.inf):
            with self.subTest(rpm=rpm), self.assertRaises(ValueError):
                _ = order_ratio(30.0, rpm)

    def test_invalid_audio_is_rejected(self) -> None:
        short = np.zeros(SAMPLE_RATE - 1, dtype=np.float32)
        non_finite = np.zeros(SAMPLE_RATE, dtype=np.float32)
        non_finite[0] = math.nan
        complex_samples = np.zeros(SAMPLE_RATE, dtype=np.complex64)
        outside_float32 = np.full(
            SAMPLE_RATE,
            np.float64(np.finfo(np.float32).max) * 2.0,
            dtype=np.float64,
        )

        with self.assertRaises(InsufficientAudioError):
            _ = analyze(short, SAMPLE_RATE)
        with self.assertRaises(ValueError):
            _ = analyze(non_finite, SAMPLE_RATE)
        with self.assertRaisesRegex(ValueError, "samples must be real"):
            _ = analyze(complex_samples, SAMPLE_RATE)
        with self.assertRaisesRegex(ValueError, "after float32 conversion"):
            _ = analyze(outside_float32, SAMPLE_RATE)
        with self.assertRaisesRegex(ValueError, "envelope cutoff"):
            _ = analyze(np.zeros(200, dtype=np.float32), 200, crossover_hz=50.0)
        with self.assertRaises(ValueError):
            _ = analyze(np.zeros(SAMPLE_RATE, dtype=np.float32), 3_600)

    def test_render_writes_png_with_rpm_annotations(self) -> None:
        result = analyze(_am_signal(((30.0, 0.8),)), SAMPLE_RATE)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "modulation.png"
            render(
                result,
                input_name="synthetic.wav",
                output_png=output,
                rpm=1800.0,
            )

            self.assertTrue(output.is_file())
            self.assertGreater(output.stat().st_size, 0)

    def test_site_worker_removes_stale_plot_when_clip_is_too_short(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "modulation.png"
            output.write_bytes(b"stale")
            with patch(
                "engine_rattle_splitter.cli.modulation.run",
                side_effect=InsufficientAudioError("too short"),
            ):
                _run_modulation(Path("short.wav"), 48_000, output, 1800.0, 4)

            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
