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
uv run engine-rattle-splitter modulation   [INPUT] [--rpm RPM | --rpm-trace CSV] [--video VIDEO] [--capture-fps FPS] [--json PATH] [--camera-target CSV] [-o OUT.png]
```

Run any subcommand with `--help` for full option descriptions.

```bash
# any input format works
uv run engine-rattle-splitter separate recordings/ride.flac -o artifacts/stems --crossover 2000
uv run engine-rattle-splitter analyze  recordings/ride.mp3 --split-at 5.2
uv run engine-rattle-splitter spectrogram ~/audio/clip.opus
uv run engine-rattle-splitter modulation recordings/steady-idle.wav --rpm 1800 --video-fps 119.88
uv run engine-rattle-splitter modulation recordings/pull.wav --rpm-trace rpm.csv --video pull.mp4 --capture-fps 240 --json artifacts/report.json --camera-target artifacts/camera.csv

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

`modulation` builds a fault-localization report from the amplitude envelope of
the high-band rattle signal. The plot and terminal report combine:

- a time-resolved 5-100 Hz modulation map and global Welch spectrum;
- independent 1.8-4, 4-8, 8-12, and 12-16 kHz carrier-subband envelopes;
- cross-subband coherence and a median consensus map;
- transient event times, widths, cadence, and phase locking;
- frequency ridge tracks and harmonic-family grouping;
- optional measured RPM order fits, control-recording contrast, and video
  timeline/capture guidance.

Broadband envelope peaks can arise from repeated impacts, changing load, or
beating between stationary tones. The `limited`, `moderate`, and `strong`
labels rank corroboration across these signal views. Their 0-1 scores are not
calibrated probabilities, do not establish causality, and do not identify the
2-16 kHz acoustic resonance as a physical motion frequency.

When `--rpm` is supplied, each measured peak is also expressed as a fixed
shaft-speed order: `order = frequency / (RPM / 60)`. Use it only for a
near-steady-RPM clip. For changing speed, pass `--rpm-trace CSV`; the file must
contain exactly `time_s,rpm`, with strictly increasing media-relative times and
positive RPM values. The report then fits each frequency track against RPM and
adds an RPM-normalized order map. `--orders 0.5,1,1.5,2,3,4` controls the
reference curves. RPM changes interpretation, never measured frequencies.

Use `--control AUDIO` with a comparable rattle-free recording to report each
candidate's active-minus-control modulation power. Matching operating speed,
microphone position, gain, and duration makes this comparison meaningful.

Without `--video` or `--capture-fps`, standalone `modulation` makes no frame-rate
assumption. `--video VIDEO` reads average/nominal frame rate, time base,
duration, and frame count with ffprobe, then aligns the video's audio track to
the analyzed audio by correlation. Use `--video-start-offset SECONDS` to supply
the mapping explicitly: `video_time = audio_time + offset`.

Container frame rate does not prove physical capture frame rate, especially for
slow-motion playback files. `--capture-fps FPS` (alias `--video-fps`) is the
authoritative physical-rate override. Frequencies at or below one quarter of
that rate have at least four frames per cycle; frequencies below `FPS / 2` but
above that boundary are marginal; frequencies at or above Nyquist are
ambiguous in ordinary frame-to-frame video. Detection remains audio-derived.

Only `site` defaults to 119.88 fps for the bundled EOS R7 experiment; override
it with `site --video-fps FPS` for another capture rate.

`--json PATH` writes the complete evidence model, tracks, event measurements,
order grid, video provenance/alignment, control deltas, and warnings as strict
JSON. `--camera-target CSV` writes corroborated time/frequency intervals for
motion magnification, including audio and aligned video times, minimum
unaliased FPS, a conservative four-frames-per-cycle recommendation, and alias
risk. The generated site publishes both files for the bundled recording.

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
