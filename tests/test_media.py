"""Tests for video metadata and timeline alignment."""

import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from engine_rattle_splitter.media import (
    align_audio_tracks,
    probe_video,
    resolve_capture_rate,
)


class MediaTests(unittest.TestCase):
    def test_parses_rational_video_metadata(self) -> None:
        output = (
            "avg_frame_rate=30000/1001\n"
            "r_frame_rate=30/1\n"
            "time_base=1/90000\n"
            "duration=10.5\n"
            "nb_frames=315\n"
        )
        with patch(
            "engine_rattle_splitter.media.subprocess.run",
            return_value=SimpleNamespace(stdout=output),
        ):
            metadata = probe_video(Path("bike.mp4"))

        self.assertAlmostEqual(metadata.average_fps, 30_000 / 1_001)
        self.assertTrue(metadata.warnings)
        override = resolve_capture_rate(metadata, 119.88)
        self.assertIsNotNone(override)
        if override is not None:
            self.assertEqual(override.provenance, "physical override")
            self.assertEqual(override.fps, 119.88)
        assumed = resolve_capture_rate(metadata, None)
        self.assertIsNotNone(assumed)
        if assumed is not None:
            self.assertEqual(assumed.provenance, "container assumption")
            self.assertIsNotNone(assumed.warning)

    def test_ignores_unparsable_optional_video_metadata(self) -> None:
        output = (
            "avg_frame_rate=30/1\n"
            "r_frame_rate=30/1\n"
            "time_base=1/90000\n"
            "duration=unknown\n"
            "nb_frames=unknown\n"
        )
        with patch(
            "engine_rattle_splitter.media.subprocess.run",
            return_value=SimpleNamespace(stdout=output),
        ):
            metadata = probe_video(Path("bike.mp4"))

        self.assertIsNone(metadata.duration_s)
        self.assertIsNone(metadata.frame_count)

    def test_audio_alignment_uses_documented_offset_sign(self) -> None:
        rng = np.random.default_rng(7)
        source = rng.normal(size=1_000).astype(np.float32)
        video = np.concatenate((np.zeros(100, dtype=np.float32), source))

        alignment = align_audio_tracks(source, video, sample_rate=1_000)

        self.assertAlmostEqual(alignment.offset_s, 0.1, places=3)
        self.assertGreater(alignment.correlation or 0.0, 0.9)

    def test_audio_alignment_accepts_inverted_and_excerpt_audio(self) -> None:
        rng = np.random.default_rng(8)
        source = rng.normal(size=3_000).astype(np.float32)

        inverted = align_audio_tracks(source, -source, sample_rate=1_000)
        excerpt = align_audio_tracks(source, source[1_000:2_500], sample_rate=1_000)

        self.assertAlmostEqual(inverted.offset_s, 0.0, places=3)
        self.assertAlmostEqual(excerpt.offset_s, -1.0, places=3)

    def test_audio_alignment_rejects_unrelated_audio(self) -> None:
        rng = np.random.default_rng(9)
        source = rng.normal(size=3_000).astype(np.float32)
        unrelated = rng.normal(size=3_000).astype(np.float32)

        with self.assertRaisesRegex(ValueError, "no reliable"):
            _ = align_audio_tracks(source, unrelated, sample_rate=1_000)

    def test_audio_alignment_rejects_ambiguous_repetition(self) -> None:
        rng = np.random.default_rng(10)
        pattern = rng.normal(size=100).astype(np.float32)
        repeated = np.tile(pattern, 30)

        with self.assertRaisesRegex(ValueError, "ambiguous"):
            _ = align_audio_tracks(repeated, repeated, sample_rate=1_000)

    def test_audio_alignment_accepts_short_partial_overlap(self) -> None:
        rng = np.random.default_rng(11)
        source = rng.normal(size=1_500).astype(np.float32)
        video = np.concatenate((np.zeros(200, dtype=np.float32), source[:1_300]))

        alignment = align_audio_tracks(source, video, sample_rate=1_000)

        self.assertAlmostEqual(alignment.offset_s, 0.2, places=3)

    def test_audio_alignment_resamples_production_rate(self) -> None:
        rng = np.random.default_rng(12)
        source = rng.normal(size=96_000).astype(np.float32)
        video = np.concatenate((np.zeros(4_800, dtype=np.float32), source))

        alignment = align_audio_tracks(source, video, sample_rate=48_000)

        self.assertAlmostEqual(alignment.offset_s, 0.1, places=3)


if __name__ == "__main__":
    unittest.main()
