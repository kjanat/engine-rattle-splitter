"""Tests for measured RPM and modulation order analysis."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from engine_rattle_splitter.fault_diagnostics import FrequencyTrack, RidgePoint
from engine_rattle_splitter.modulation import ModulationSpectrogram
from engine_rattle_splitter.orders import (
    RpmPoint,
    RpmTrace,
    analyze_orders,
    fixed_rpm,
    load_rpm_trace,
)


class OrderTests(unittest.TestCase):
    def test_loads_strict_rpm_csv(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rpm.csv"
            path.write_text("time_s,rpm\n0,1000\n1,1200\n", encoding="utf-8")

            trace = load_rpm_trace(path)

            self.assertEqual(len(trace.points), 2)
            self.assertEqual(trace.points[-1].rpm, 1200.0)

    def test_rejects_non_increasing_rpm_times(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rpm.csv"
            path.write_text("time_s,rpm\n0,1000\n0,1200\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "strictly increasing"):
                _ = load_rpm_trace(path)

    def test_rejects_other_invalid_rpm_csv_shapes(self) -> None:
        cases = (
            ("time,rpm\n0,1000\n1,1200\n", "exactly the columns"),
            ("time_s,rpm\n0,0\n1,1200\n", "finite and positive"),
            ("time_s,rpm\n0,nope\n1,1200\n", "invalid RPM trace row"),
            ("time_s,rpm\n0,1000\n", "at least two points"),
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rpm.csv"
            for contents, message in cases:
                with self.subTest(message=message):
                    path.write_text(contents, encoding="utf-8")
                    with self.assertRaisesRegex(ValueError, message):
                        _ = load_rpm_trace(path)

    def test_fits_known_two_x_order(self) -> None:
        times = np.arange(0.0, 5.25, 0.25, dtype=np.float64)
        rpm = 1_000.0 + 200.0 * times
        frequencies = 2.0 * rpm / 60.0
        track = FrequencyTrack(
            track_id=1,
            points=tuple(
                RidgePoint(
                    time_s=float(time), frequency_hz=float(frequency), level_db=0.0
                )
                for time, frequency in zip(times, frequencies, strict=True)
            ),
            duration_s=5.0,
            median_frequency_hz=float(np.median(frequencies)),
            minimum_frequency_hz=float(np.min(frequencies)),
            maximum_frequency_hz=float(np.max(frequencies)),
            slope_hz_per_s=float((frequencies[-1] - frequencies[0]) / 5.0),
        )
        trace = RpmTrace(
            kind="trace",
            points=tuple(
                RpmPoint(time_s=float(time), rpm=float(value))
                for time, value in zip(times, rpm, strict=True)
            ),
            source=Path("rpm.csv"),
        )
        spectrogram = ModulationSpectrogram(
            times_s=times,
            frequencies_hz=np.arange(5.0, 100.5, 0.5, dtype=np.float64),
            psd_db=np.zeros((191, len(times)), dtype=np.float64),
            window_duration_s=2.0,
            hop_duration_s=0.25,
            frequency_resolution_hz=0.5,
        )

        analysis = analyze_orders((track,), spectrogram, trace)

        fit = analysis.fits[0]
        self.assertAlmostEqual(fit.fitted_order or 0.0, 2.0, places=6)
        self.assertEqual(fit.nearest_reference_order, 2.0)
        self.assertTrue(fit.consistent)
        ridge = np.argmax(analysis.order_map.psd_db, axis=0)
        self.assertTrue(
            bool(np.all(analysis.order_map.valid[ridge, np.arange(len(times))]))
        )

    def test_order_map_covers_measurable_low_and_high_orders(self) -> None:
        times = np.arange(0.0, 2.25, 0.25, dtype=np.float64)
        spectrogram = ModulationSpectrogram(
            times_s=times,
            frequencies_hz=np.arange(5.0, 100.5, 0.5, dtype=np.float64),
            psd_db=np.zeros((191, len(times)), dtype=np.float64),
            window_duration_s=2.0,
            hop_duration_s=0.25,
            frequency_resolution_hz=0.5,
        )

        high_rpm = analyze_orders((), spectrogram, fixed_rpm(6_000.0))
        low_rpm = analyze_orders((), spectrogram, fixed_rpm(60.0))

        self.assertLessEqual(float(high_rpm.order_map.orders[0]), 0.05)
        self.assertGreaterEqual(float(low_rpm.order_map.orders[-1]), 100.0)
        self.assertLessEqual(len(low_rpm.order_map.orders), 400)
        self.assertAlmostEqual(
            low_rpm.order_map.order_resolution,
            float(low_rpm.order_map.orders[1] - low_rpm.order_map.orders[0]),
        )


if __name__ == "__main__":
    unittest.main()
