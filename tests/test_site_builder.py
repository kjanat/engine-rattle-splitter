"""Tests for modulation artifact placement on the generated site."""

import unittest
from pathlib import Path

from engine_rattle_splitter.site_builder import Artifact, _render_sections


class SiteBuilderTests(unittest.TestCase):
    def test_modulation_has_dedicated_section(self) -> None:
        artifact = Artifact(path=Path("modulation.png"), kind="image", size_bytes=1)

        rendered = _render_sections([artifact])

        self.assertIn("<h2>Rattle Modulation</h2>", rendered)
        self.assertIn("shows when modulation components occur", rendered)
        self.assertIn("configured video sampling limits", rendered)
        self.assertNotIn("13 s mark", rendered)


if __name__ == "__main__":
    unittest.main()
