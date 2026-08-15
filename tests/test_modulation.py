"""Tests for rattle-envelope modulation analysis."""

import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from engine_rattle_splitter.audio_io import Float32Array
from engine_rattle_splitter.cli import (
    DEFAULT_VIDEO_FPS,
    Args,
    _run_modulation,
    build_parser,
)
from engine_rattle_splitter.modulation import (
    InsufficientAudioError,
    _bin_edges,
    _validate_video_fps,
    analyze,
    modulation_spectrogram,
    order_ratio,
    render,
    resample_envelope,
    video_observability,
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


def _changing_modulation_signal() -> Float32Array:
    times = np.arange(round(SAMPLE_RATE * DURATION_S), dtype=np.float64) / SAMPLE_RATE
    frequencies = np.where(times < 4.0, 20.0, np.where(times < 8.0, 35.0, 50.0))
    phase = 2.0 * np.pi * np.cumsum(frequencies) / SAMPLE_RATE
    amplitude = 1.0 + 0.8 * np.sin(phase)
    carrier = amplitude * np.sin(2.0 * np.pi * CARRIER_HZ * times)
    engine = 2.0 * np.sin(2.0 * np.pi * 300.0 * times)
    return (carrier + engine).astype(np.float32)


def _impact_train_signal() -> Float32Array:
    sample_count = round(SAMPLE_RATE * DURATION_S)
    times = np.arange(sample_count, dtype=np.float64) / SAMPLE_RATE
    signal = 1.5 * np.sin(2.0 * np.pi * 300.0 * times)
    rng = np.random.default_rng(42)
    burst_times = np.arange(round(0.018 * SAMPLE_RATE), dtype=np.float64) / SAMPLE_RATE
    burst = np.exp(-220.0 * burst_times) * (
        rng.normal(size=len(burst_times))
        + 0.8 * np.sin(2.0 * np.pi * 3_500.0 * burst_times)
        + 0.6 * np.sin(2.0 * np.pi * 5_200.0 * burst_times)
    )
    period = round(SAMPLE_RATE / 25.0)
    for start in range(2 * SAMPLE_RATE, 10 * SAMPLE_RATE, period):
        end = min(start + len(burst), sample_count)
        signal[start:end] += burst[: end - start]
    return signal.astype(np.float32)


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

    def test_spectrogram_localizes_changing_modulation(self) -> None:
        result = analyze(_changing_modulation_signal(), SAMPLE_RATE)

        for time_s, expected_hz in ((2.0, 20.0), (6.0, 35.0), (10.0, 50.0)):
            time_index = int(np.argmin(np.abs(result.spectrogram.times_s - time_s)))
            frequency_index = int(np.argmax(result.spectrogram.psd_db[:, time_index]))
            measured_hz = float(result.spectrogram.frequencies_hz[frequency_index])
            with self.subTest(time_s=time_s):
                self.assertAlmostEqual(measured_hz, expected_hz, delta=0.5)

    def test_repeated_broadband_impacts_recover_event_rate(self) -> None:
        result = analyze(_impact_train_signal(), SAMPLE_RATE)

        self.assertTrue(
            any(abs(peak.frequency_hz - 25.0) <= 0.15 for peak in result.peaks)
        )
        frequency_index = int(
            np.argmin(np.abs(result.spectrogram.frequencies_hz - 25.0))
        )
        active_index = int(np.argmin(np.abs(result.spectrogram.times_s - 6.0)))
        inactive_index = int(np.argmin(np.abs(result.spectrogram.times_s - 1.0)))
        self.assertGreater(
            float(result.spectrogram.psd_db[frequency_index, active_index]),
            float(result.spectrogram.psd_db[frequency_index, inactive_index]) + 12.0,
        )

    def test_silence_has_finite_result_without_peaks(self) -> None:
        samples = np.zeros(round(SAMPLE_RATE * 2.0), dtype=np.float32)
        result = analyze(samples, SAMPLE_RATE)

        self.assertEqual(result.peaks, ())
        self.assertTrue(bool(np.all(np.isfinite(result.envelope))))
        self.assertTrue(bool(np.all(np.isfinite(result.spectrum_db))))
        self.assertTrue(bool(np.all(np.isfinite(result.spectrogram.psd_db))))

    def test_one_second_clip_has_one_local_window(self) -> None:
        times = np.arange(SAMPLE_RATE, dtype=np.float64) / SAMPLE_RATE
        samples = (
            (1.0 + 0.8 * np.sin(2.0 * np.pi * 30.0 * times))
            * np.sin(2.0 * np.pi * CARRIER_HZ * times)
        ).astype(np.float32)

        result = analyze(samples, SAMPLE_RATE)

        self.assertEqual(len(result.spectrogram.times_s), 1)
        self.assertEqual(result.spectrogram.frequency_resolution_hz, 1.0)

    def test_spectrogram_includes_end_aligned_window(self) -> None:
        envelope = np.zeros(round(2.1 * 400), dtype=np.float64)
        tail_times = np.arange(40, dtype=np.float64) / 400.0
        envelope[-40:] = 1.0 + np.sin(2.0 * np.pi * 20.0 * tail_times)

        spectrogram = modulation_spectrogram(envelope)

        self.assertEqual(len(spectrogram.times_s), 2)
        self.assertAlmostEqual(float(spectrogram.times_s[-1]), 1.1)
        frequency_index = int(np.argmin(np.abs(spectrogram.frequencies_hz - 20.0)))
        self.assertGreater(
            float(spectrogram.psd_db[frequency_index, -1]),
            float(spectrogram.psd_db[frequency_index, 0]) + 20.0,
        )

    def test_nonuniform_spectrogram_times_have_explicit_edges(self) -> None:
        centers = np.array([1.0, 1.0025], dtype=np.float64)

        edges = _bin_edges(centers, 0.0, 2.0025)

        np.testing.assert_allclose(edges, [0.99875, 1.00125, 1.00375])

    def test_resampling_preserves_constant_envelope_edges(self) -> None:
        envelope = np.ones(201, dtype=np.float64)

        resampled = resample_envelope(envelope, sample_rate=201)

        self.assertEqual(len(resampled), 400)
        self.assertTrue(bool(np.allclose(resampled, 1.0, atol=0.001)))

    def test_order_ratio_uses_fixed_crank_frequency(self) -> None:
        self.assertAlmostEqual(order_ratio(27.4, 1800.0), 27.4 / 30.0)

    def test_video_observability_boundaries(self) -> None:
        self.assertEqual(video_observability(30.0, 120.0), "well sampled")
        self.assertEqual(video_observability(30.1, 120.0), "marginal")
        self.assertEqual(video_observability(59.9, 120.0), "marginal")
        self.assertEqual(video_observability(60.0, 120.0), "at or above Nyquist")

    def test_invalid_video_inputs_are_rejected(self) -> None:
        for video_fps in (0.0, -1.0, math.nan, math.inf):
            with self.subTest(video_fps=video_fps), self.assertRaises(ValueError):
                _validate_video_fps(video_fps)
        for frequency_hz in (0.0, -1.0, math.nan, math.inf):
            with self.subTest(frequency_hz=frequency_hz), self.assertRaises(ValueError):
                _ = video_observability(frequency_hz, 120.0)

    def test_site_uses_supported_video_fps_default(self) -> None:
        args = build_parser().parse_args(["site"], namespace=Args())

        self.assertEqual(args.video_fps, DEFAULT_VIDEO_FPS)

    def test_fixed_and_traced_rpm_are_mutually_exclusive(self) -> None:
        with self.assertRaises(SystemExit):
            _ = build_parser().parse_args([
                "modulation",
                "recording.wav",
                "--rpm",
                "1800",
                "--rpm-trace",
                "rpm.csv",
            ])

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
                video_fps=119.88,
            )

            self.assertTrue(output.is_file())
            self.assertGreater(output.stat().st_size, 0)

    def test_site_worker_removes_stale_plot_when_clip_is_too_short(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "modulation.png"
            output.write_bytes(b"stale")
            with patch(
                "engine_rattle_splitter.cli.localization.run",
                side_effect=InsufficientAudioError("too short"),
            ):
                _run_modulation(Path("short.wav"), 48_000, output, 1800.0, 4)

            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
