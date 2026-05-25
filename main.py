"""CLI entry — wires together the engine_sound_splitter pipelines.

Subcommands:
  separate  decode → crossover filter → write engine.wav + rattles.wav
  analyze   compute frame features and contrast before/after a time mark
"""

import argparse
import sys
from pathlib import Path

from engine_sound_splitter import analysis, pipeline

DEFAULT_INPUT = Path("Shitty motor (goede recording).m4a")
DEFAULT_OUTPUT_DIR = Path(".")
DEFAULT_SAMPLE_RATE = 48_000
DEFAULT_CROSSOVER_HZ = 1800.0
DEFAULT_CROSSOVER_ORDER = 4
DEFAULT_SPLIT_AT = 13.0
DEFAULT_ANALYSIS_PNG = Path("analysis.png")


def cmd_separate(args: argparse.Namespace) -> int:
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
    print(f"engine   RMS  : {stats['engine_rms']:.4f}  -> {stats['engine_path']}")
    print(f"rattles  RMS  : {stats['rattles_rms']:.4f}  -> {stats['rattles_path']}")
    return 0


def cmd_analyze(args: argparse.Namespace) -> int:
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="engine-sound-splitter")
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=DEFAULT_SAMPLE_RATE,
        help="decode sample rate (default: %(default)s)",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    sep = sub.add_parser("separate", help="split into engine + rattles via crossover")
    sep.add_argument("input", type=Path, nargs="?", default=DEFAULT_INPUT)
    sep.add_argument("-o", "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    sep.add_argument(
        "--crossover",
        type=float,
        default=DEFAULT_CROSSOVER_HZ,
        help="crossover frequency in Hz (default: %(default)s)",
    )
    sep.add_argument(
        "--order",
        type=int,
        default=DEFAULT_CROSSOVER_ORDER,
        help="Butterworth filter order (default: %(default)s)",
    )
    sep.set_defaults(func=cmd_separate)

    an = sub.add_parser(
        "analyze", help="contrast frame features before/after a time mark"
    )
    an.add_argument("input", type=Path, nargs="?", default=DEFAULT_INPUT)
    an.add_argument(
        "--split-at",
        type=float,
        default=DEFAULT_SPLIT_AT,
        help="time in seconds dividing the two halves (default: %(default)s)",
    )
    an.add_argument("-o", "--output", type=Path, default=DEFAULT_ANALYSIS_PNG)
    an.set_defaults(func=cmd_analyze)

    return parser


def main() -> int:
    args = build_parser().parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
