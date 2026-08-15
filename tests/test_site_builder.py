"""Tests for modulation artifact placement on the generated site."""

import unittest
from pathlib import Path

from engine_rattle_splitter.site_builder import Artifact, _render_sections


class SiteBuilderTests(unittest.TestCase):
    def test_modulation_has_dedicated_section(self) -> None:
        artifacts = [
            Artifact(path=Path("modulation.png"), kind="image", size_bytes=1),
            Artifact(path=Path("fault-report.json"), kind="file", size_bytes=1),
            Artifact(path=Path("camera-targets.csv"), kind="file", size_bytes=1),
        ]

        rendered = _render_sections(artifacts)

        self.assertIn("<h2>Rattle Modulation</h2>", rendered)
        self.assertIn("combines carrier-subband agreement", rendered)
        self.assertIn("configured video sampling limits", rendered)
        self.assertIn("Download machine-readable evidence", rendered)
        self.assertIn("Download camera targets", rendered)
        self.assertNotIn("<h2>Downloads</h2>", rendered)
        self.assertNotIn("13 s mark", rendered)

    def test_sidecars_remain_downloadable_without_plot(self) -> None:
        artifacts = [
            Artifact(path=Path("fault-report.json"), kind="file", size_bytes=1),
            Artifact(path=Path("camera-targets.csv"), kind="file", size_bytes=1),
        ]

        rendered = _render_sections(artifacts)

        self.assertIn("<h2>Downloads</h2>", rendered)
        self.assertIn("fault-report.json", rendered)
        self.assertIn("camera-targets.csv", rendered)


if __name__ == "__main__":
    unittest.main()
