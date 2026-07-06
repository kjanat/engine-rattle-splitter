# Repository Guidelines

## Project Structure & Module Organization

This is a small Python audio-processing package managed with `uv`. Source code
uses the `src/` layout. The CLI lives in `src/engine_rattle_splitter/cli.py`;
reusable code is under `src/engine_rattle_splitter/`:

- `audio_io.py`: ffmpeg-backed decode/encode and WAV writing.
- `filters.py`: complementary Butterworth crossover logic.
- `pipeline.py`: high-level split orchestration.
- `analysis.py` and `spectrogram.py`: diagnostic plot generation.

Committed source recordings live in `recordings/`. Static site source files live
in `web/`. Generated stems, plots, and deployable site output go under
`artifacts/` and should not be treated as source.

## Build, Test, and Development Commands

- `uv sync --frozen`: install the locked Python 3.14 environment.
- `uv run engine-rattle-splitter separate [INPUT] -o artifacts/stems`: split an
  audio file into engine and rattle stems.
- `uv run engine-rattle-splitter analyze [INPUT] --split-at 13.0`: compare
  features before and after a transition.
- `uv run engine-rattle-splitter spectrogram [INPUT] -o artifacts/spectrogram.png`:
  render a diagnostic spectrogram.
- `uv run engine-rattle-splitter site`: reproduce the GitHub Pages build under
  `artifacts/site`.
- `uv run ruff check .`: run lint checks.
- `uv run basedpyright`: run static type checking.
- `uv build`: build the package artifacts.

Install `ffmpeg` locally; audio decoding and MP3 encoding shell out to it.

## Coding Style & Naming Conventions

Use typed Python with 4-space indentation. Prefer `pathlib.Path`, explicit
return types, and small functions that keep DSP, I/O, and CLI wiring separate.
Use `snake_case` for functions and variables, `UPPER_CASE` for constants, and
short module docstrings that explain behavior. Keep CLI defaults centralized in
`src/engine_rattle_splitter/cli.py`.

## Testing Guidelines

There is no dedicated automated test suite yet. Before submitting changes, run
`ruff`, `basedpyright`, and at least one CLI smoke test against the bundled
`.m4a` input. When adding tests, place them under `tests/` as `test_*.py`,
prefer small synthetic NumPy arrays for filter behavior, and avoid depending on
committed generated audio.

## Commit & Pull Request Guidelines

Recent commits use short imperative subjects, sometimes scoped (`ci:`) and
sometimes with compact rationale in parentheses. Keep commits focused and avoid
unrelated formatting churn. Pull requests should describe the behavior change,
list verification commands, link relevant issues, and include before/after plots
or audio notes when DSP output changes.

## Security & Configuration Tips

Do not commit local virtual environments, build artifacts, or regenerated media
unless explicitly needed. Keep GitHub Actions pinned when editing workflows,
matching the existing workflow style.
