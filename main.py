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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="engine-sound-splitter")
    _ = parser.add_argument(
        "--sample-rate",
        type=int,
        default=DEFAULT_SAMPLE_RATE,
        help="decode sample rate (default: %(default)s)",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    sep = sub.add_parser("separate", help="split into engine + rattles via crossover")
    _ = sep.add_argument("input", type=Path, nargs="?", default=DEFAULT_INPUT)
    _ = sep.add_argument("-o", "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    _ = sep.add_argument(
        "--crossover",
        type=float,
        default=DEFAULT_CROSSOVER_HZ,
        help="crossover frequency in Hz (default: %(default)s)",
    )
    _ = sep.add_argument(
        "--order",
        type=int,
        default=DEFAULT_CROSSOVER_ORDER,
        help="Butterworth filter order (default: %(default)s)",
    )
    sep.set_defaults(func=cmd_separate)

    an = sub.add_parser(
        "analyze", help="contrast frame features before/after a time mark"
    )
    _ = an.add_argument("input", type=Path, nargs="?", default=DEFAULT_INPUT)
    _ = an.add_argument(
        "--split-at",
        type=float,
        default=DEFAULT_SPLIT_AT,
        help="time in seconds dividing the two halves (default: %(default)s)",
    )
    _ = an.add_argument("-o", "--output", type=Path, default=DEFAULT_ANALYSIS_PNG)
    an.set_defaults(func=cmd_analyze)

    sg = sub.add_parser("spectrogram", help="render a log-frequency dB spectrogram PNG")
    _ = sg.add_argument("input", type=Path, nargs="?", default=DEFAULT_INPUT)
    _ = sg.add_argument("-o", "--output", type=Path, default=DEFAULT_SPECTROGRAM_PNG)
    sg.set_defaults(func=cmd_spectrogram)

    return parser


def main() -> int:
    args = build_parser().parse_args(namespace=Args())
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
