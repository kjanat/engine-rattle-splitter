# engine-rattle-splitter

Splits an audio recording into a low-band stem (engine combustion drone)
and a high-band stem (mechanical rattles / knocks / dangling parts) via
a complementary Butterworth crossover. Reconstruction is bit-exact:
`engine.wav + rattles.wav == input`.

Works on **any audio file ffmpeg can decode** — wav, mp3, m4a, flac,
ogg, opus, mp4 audio tracks, etc. The included
`Shitty motor (goede recording).m4a` is just the example used to tune
the defaults and lives under `recordings/`.

## Usage

```bash
uv run engine-rattle-splitter separate     [INPUT] [-o DIR] [--crossover HZ] [--order N]
uv run engine-rattle-splitter analyze      [INPUT] [--split-at SECONDS] [-o OUT.png]
uv run engine-rattle-splitter spectrogram  [INPUT] [-o OUT.png]
```

Run any subcommand with `--help` for full option descriptions.

```bash
# any input format works
uv run engine-rattle-splitter separate recordings/ride.flac -o artifacts/stems --crossover 2000
uv run engine-rattle-splitter analyze  recordings/ride.mp3 --split-at 5.2
uv run engine-rattle-splitter spectrogram ~/audio/clip.opus

# no INPUT → falls back to the bundled recording
uv run engine-rattle-splitter separate
```

## Tuning per recording

Two parameters depend on what you feed in. Defaults are tuned for the
bundled file and likely need adjusting for other recordings:

- **`--crossover` (default 1800 Hz)** — the frequency where engine
  combustion content ends and rattle content begins. For the bundled
  recording the engine lives below 500 Hz and rattles dominate above
  2 kHz, so 1800 Hz cleanly separates them. Different motors (smaller,
  quieter, higher-RPM) split at different bands. Run `analyze` on a
  rattle-free segment first if unsure.
- **`--split-at` (default 13.0 s)** — `analyze` only. The time
  dividing the two halves it compares (e.g. rattling vs. not rattling).
  Meaningless if your recording has no such transition; pass the actual
  boundary in your file, or skip `analyze` entirely.

## Listen / look

Stems and plots are regenerated from the bundled m4a on every CI run
and hosted at <https://kjanat.github.io/engine-rattle-splitter/>. The
spectrogram, analysis plots, and playable mp3 stems live there — they
are not committed to this repo (CI is the source of truth).

![spectrogram](https://kjanat.github.io/engine-rattle-splitter/spectrogram.png)

Frame features and per-octave-band energy contrast across the 13 s
mark of the bundled recording — quantifies what "rattling" looks like
statistically (spectral flux +57%, energy doubles above 2 kHz).

![analysis](https://kjanat.github.io/engine-rattle-splitter/analysis.png)

Same analysis run on the rattles stem validates the split — rattle
energy is 2× louder before 13 s, with no engine-band content leaking
through.

![rattles analysis](https://kjanat.github.io/engine-rattle-splitter/rattles_analysis.png)
