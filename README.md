# engine-sound-splitter

Splits an audio recording into a low-band stem (engine combustion drone)
and a high-band stem (mechanical rattles / knocks / dangling parts) via
a complementary Butterworth crossover. Reconstruction is bit-exact:
`engine.wav + rattles.wav == input`.

Works on **any audio file ffmpeg can decode** — wav, mp3, m4a, flac,
ogg, opus, mp4 audio tracks, etc. The included
`Shitty motor (goede recording).m4a` is just the example used to tune
the defaults.

## Usage

```bash
uv run python main.py separate     [INPUT] [-o DIR] [--crossover HZ] [--order N]
uv run python main.py analyze      [INPUT] [--split-at SECONDS] [-o OUT.png]
uv run python main.py spectrogram  [INPUT] [-o OUT.png]
```

Run any subcommand with `--help` for full option descriptions.

```bash
# any input format works
uv run python main.py separate ride.flac -o ./stems --crossover 2000
uv run python main.py analyze  recording.mp3 --split-at 5.2
uv run python main.py spectrogram ~/audio/clip.opus -o clip.png

# no INPUT → falls back to the bundled recording
uv run python main.py separate
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

## Spectrogram

![spectrogram](spectrogram.png)

## Analysis

Frame features and per-octave-band energy contrast across the 13s mark
of the bundled recording — quantifies what "rattling" looks like
statistically (spectral flux +57%, energy doubles above 2 kHz).

![analysis](analysis.png)

Same analysis run on `rattles.wav` validates the split — rattle energy
is 2× louder before 13s, with no engine-band content leaking through.

![rattles analysis](rattles_analysis.png)

## Audio

- [engine.mp3](engine.mp3)
- [rattles.mp3](rattles.mp3)

Both mp3s are re-encoded automatically every time `separate` runs, so
they always reflect the current crossover output.
