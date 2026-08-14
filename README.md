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
uv run engine-rattle-splitter modulation   [INPUT] [--rpm RPM] [--video-fps FPS] [--crossover HZ] [--filter-order N] [-o OUT.png]
```

Run any subcommand with `--help` for full option descriptions.

```bash
# any input format works
uv run engine-rattle-splitter separate recordings/ride.flac -o artifacts/stems --crossover 2000
uv run engine-rattle-splitter analyze  recordings/ride.mp3 --split-at 5.2
uv run engine-rattle-splitter spectrogram ~/audio/clip.opus
uv run engine-rattle-splitter modulation recordings/steady-idle.wav --rpm 1800 --video-fps 119.88

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

## Rattle modulation

`modulation` extracts the amplitude envelope of the high-band rattle signal
and shows both a time-resolved 5–100 Hz modulation map and prominent global
components. They may expose repeated impacts or bursts, but a broadband
envelope can also contain beating between stationary tones. Treat the peaks as
camera-frequency candidates, not proof of a mechanical source or the 2–16 kHz
acoustic resonances excited by an impact.

When `--rpm` is supplied, each measured peak is also expressed as a fixed
shaft-speed order: `order = frequency / (RPM / 60)`. RPM only changes this
interpretation; it never changes the measured frequencies. Use it only for a
near-steady-RPM clip. Acceleration and deceleration require synchronized,
time-varying RPM data for proper order tracking.

The CLI does not inspect video metadata. `modulation` adds presentation-only
video guidance only when `--video-fps FPS` is explicitly supplied. Frequencies
at or below one quarter of that frame rate have at least four frames per cycle;
frequencies between that and the Nyquist limit (`FPS / 2`) are marginal, and
frequencies at or above Nyquist cannot be matched unambiguously in ordinary
frame-to-frame video. Detection remains entirely audio-derived.

Only `site` defaults to 119.88 fps for the bundled EOS R7 experiment; override
it with `site --video-fps FPS` for another capture rate.

## Listen / look

Stems and plots are regenerated from the bundled m4a on every CI run and hosted
[here][pages]. The site page is rendered from the files present in the CI output
directory; adding or renaming generated artifacts changes the page without
editing a content manifest. Generated outputs are not committed to this repo.

![spectrogram](https://kjanat.github.io/engine-rattle-splitter/spectrogram.png)

Frame features and per-octave-band energy contrast across the 13 s
mark of the bundled recording — quantifies what "rattling" looks like
statistically (spectral flux +57%, energy doubles above 2 kHz).

![analysis](https://kjanat.github.io/engine-rattle-splitter/analysis.png)

Same analysis run on the rattles stem validates the split — rattle
energy is 2× louder before 13 s, with no engine-band content leaking
through.

![rattles analysis](https://kjanat.github.io/engine-rattle-splitter/rattles_analysis.png)

[pages]: https://kjanat.github.io/engine-rattle-splitter/

<!-- markdownlint-disable-file MD013 MD059 -->
<!-- rumdl-disable-file        MD013 MD059 -->
