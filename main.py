"""CLI entry — wires together the engine_sound_splitter pipelines.

Subcommands:
  separate     decode → crossover filter → write engine.wav + rattles.wav
  analyze      compute frame features and contrast before/after a time mark
  spectrogram  render a log-frequency dB spectrogram PNG
"""

import argparse
import sys
from collections.abc import Callable
from pathlib import Path

from engine_sound_splitter import analysis, pipeline, spectrogram

DEFAULT_INPUT = Path("Shitty motor (goede recording).m4a")
DEFAULT_OUTPUT_DIR = Path()
DEFAULT_SAMPLE_RATE = 48_000
DEFAULT_CROSSOVER_HZ = 1800.0
DEFAULT_CROSSOVER_ORDER = 4
DEFAULT_SPLIT_AT = 13.0
DEFAULT_ANALYSIS_PNG = Path("analysis.png")
DEFAULT_SPECTROGRAM_PNG = Path("spectrogram.png")


def _no_op(_: "Args") -> int:
    return 0


class Args(argparse.Namespace):
    """Typed view of parsed CLI args. Class-level defaults satisfy the
    type checker; argparse overwrites them on parse_args()."""

    cmd: str = ""
    sample_rate: int = DEFAULT_SAMPLE_RATE
    input: Path = DEFAULT_INPUT
    output_dir: Path = DEFAULT_OUTPUT_DIR
    crossover: float = DEFAULT_CROSSOVER_HZ
    order: int = DEFAULT_CROSSOVER_ORDER
    split_at: float = DEFAULT_SPLIT_AT
    output: Path = DEFAULT_ANALYSIS_PNG
    func: Callable[["Args"], int] = _no_op


def cmd_separate(args: Args) -> int:
    if not args.input.exists():
        print(f"missing: {args.input}", file=sys.stderr)
        return 1
    stats = pipeline.split(
        input_path=args.input,
        output_dir=args.output_dir,
        sample_rate=args.sample_rate,
        crossover_hz=args.crossover,
        order=args.order,
    )
    print(f"input         : {args.input}")
    print(f"duration      : {stats['duration_s']:.2f} s")
    print(f"crossover     : {args.crossover} Hz, Butterworth order {args.order}")
    print(f"input    RMS  : {stats['input_rms']:.4f}")
    print(
        f"engine   RMS  : {stats['engine_rms']:.4f}  -> {stats['engine_wav']} / {stats['engine_mp3']}"
    )
    print(
        f"rattles  RMS  : {stats['rattles_rms']:.4f}  -> {stats['rattles_wav']} / {stats['rattles_mp3']}"
    )
    return 0


def cmd_analyze(args: Args) -> int:
    if not args.input.exists():
        print(f"missing: {args.input}", file=sys.stderr)
        return 1
    analysis.run(
        input_path=args.input,
        sample_rate=args.sample_rate,
        split_at=args.split_at,
        output_png=args.output,
    )
    return 0


def cmd_spectrogram(args: Args) -> int:
    if not args.input.exists():
        print(f"missing: {args.input}", file=sys.stderr)
        return 1
    spectrogram.run(
        input_path=args.input,
        sample_rate=args.sample_rate,
        output_png=args.output,
    )
    return 0


INPUT_HELP = (
    "audio input file — any format ffmpeg can decode "
    "(wav, mp3, m4a, flac, ogg, opus, mp4 audio, ...). "
    "Default: the bundled motorcycle recording."
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="engine-sound-splitter",
        description=(
            "Split engine recordings into a low-band stem (combustion drone) "
            "and a high-band stem (mechanical rattles) via complementary "
            "Butterworth crossover. Works on any audio file ffmpeg can decode."
        ),
        epilog=(
            "Examples:\n"
            "  python main.py separate ride.flac -o ./stems --crossover 2000\n"
            "  python main.py analyze  ride.flac --split-at 5.2\n"
            "  python main.py spectrogram ride.flac -o ride.png"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _ = parser.add_argument(
        "--sample-rate",
        type=int,
        default=DEFAULT_SAMPLE_RATE,
        metavar="HZ",
        help=(
            "decode/resample rate; ffmpeg resamples the input to this "
            "(default: %(default)s)"
        ),
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    sep = sub.add_parser(
        "separate",
        help="split into engine + rattles via crossover",
        description=(
            "Decode the input, apply a zero-phase forward-backward Butterworth "
            "low-pass at --crossover Hz, derive the high band by subtraction "
            "(low + high == input bit-exact). Writes engine.wav, rattles.wav, "
            "and re-encoded engine.mp3 / rattles.mp3 into --output-dir."
        ),
    )
    _ = sep.add_argument(
        "input", type=Path, nargs="?", default=DEFAULT_INPUT, help=INPUT_HELP
    )
    _ = sep.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        metavar="DIR",
        help=(
            "directory to write engine.{wav,mp3} + rattles.{wav,mp3} "
            "(default: current directory)"
        ),
    )
    _ = sep.add_argument(
        "--crossover",
        type=float,
        default=DEFAULT_CROSSOVER_HZ,
        metavar="HZ",
        help=(
            "crossover frequency in Hz — frequencies below go to the engine "
            "stem, above go to the rattles stem. %(default)s is tuned for the "
            "bundled recording; different engines split at different bands. "
            "Use `analyze` on a clean segment to find the right value."
        ),
    )
    _ = sep.add_argument(
        "--order",
        type=int,
        default=DEFAULT_CROSSOVER_ORDER,
        metavar="N",
        help=(
            "Butterworth filter order; forward-backward filtering doubles the "
            "effective slope to ~12N dB/octave at the crossover "
            "(default: %(default)s → ~48 dB/oct)"
        ),
    )
    sep.set_defaults(func=cmd_separate)

    an = sub.add_parser(
        "analyze",
        help="contrast frame features before/after a time mark",
        description=(
            "Compute RMS, crest factor, spectral centroid/flatness/flux, and "
            "per-octave-band energy on either side of --split-at. Useful for "
            "quantifying what changes at a known transition (e.g. when "
            "rattling stops) and for picking a `separate --crossover` value."
        ),
    )
    _ = an.add_argument(
        "input", type=Path, nargs="?", default=DEFAULT_INPUT, help=INPUT_HELP
    )
    _ = an.add_argument(
        "--split-at",
        type=float,
        default=DEFAULT_SPLIT_AT,
        metavar="SECONDS",
        help=(
            "time in seconds dividing the two halves to compare. "
            "Only meaningful when your recording has a known transition; "
            "default %(default)s matches the bundled recording — override per file."
        ),
    )
    _ = an.add_argument(
        "-o",
        "--output",
        type=Path,
        default=DEFAULT_ANALYSIS_PNG,
        metavar="PNG",
        help="PNG output path (default: %(default)s)",
    )
    an.set_defaults(func=cmd_analyze)

    sg = sub.add_parser(
        "spectrogram",
        help="render a log-frequency dB spectrogram PNG",
        description=(
            "Render a log-frequency, dB-magnitude STFT spectrogram of the "
            "input — diagnostic view of where energy lives over time."
        ),
    )
    _ = sg.add_argument(
        "input", type=Path, nargs="?", default=DEFAULT_INPUT, help=INPUT_HELP
    )
    _ = sg.add_argument(
        "-o",
        "--output",
        type=Path,
        default=DEFAULT_SPECTROGRAM_PNG,
        metavar="PNG",
        help="PNG output path (default: %(default)s)",
    )
    sg.set_defaults(func=cmd_spectrogram)

    return parser


def main() -> int:
    args = build_parser().parse_args(namespace=Args())
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
