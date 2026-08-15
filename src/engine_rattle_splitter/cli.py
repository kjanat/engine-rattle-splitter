"""CLI entry — wires together the engine_rattle_splitter pipelines.

Subcommands:
  separate     decode → crossover filter → write engine.wav + rattles.wav
  analyze      compute frame features and contrast before/after a time mark
  spectrogram  render a log-frequency dB spectrogram PNG
  modulation   measure high-band envelope modulation components
"""

import argparse
import math
import multiprocessing
import os
import shutil
import sys
from collections.abc import Callable
from concurrent.futures import Future, ProcessPoolExecutor
from pathlib import Path
from typing import Self

from engine_rattle_splitter import (
    analysis,
    localization,
    modulation,
    moments,
    orders,
    pipeline,
    site_builder,
    spectrogram,
)

DEFAULT_INPUT = Path("recordings/Shitty motor (goede recording).m4a")
DEFAULT_OUTPUT_DIR = Path("artifacts/stems")
DEFAULT_SAMPLE_RATE = 48_000
DEFAULT_CROSSOVER_HZ = 1800.0
DEFAULT_CROSSOVER_ORDER = 4
DEFAULT_SPLIT_AT = 13.0
DEFAULT_ANALYSIS_PNG = Path("artifacts/analysis.png")
DEFAULT_SPECTROGRAM_PNG = Path("artifacts/spectrogram.png")
DEFAULT_MODULATION_PNG = Path("artifacts/modulation.png")
DEFAULT_FAULT_REPORT_JSON = "fault-report.json"
DEFAULT_CAMERA_TARGETS_CSV = "camera-targets.csv"
DEFAULT_REFERENCE_ORDERS = orders.DEFAULT_REFERENCE_ORDERS
DEFAULT_VIDEO_FPS = 119.88
DEFAULT_SITE_DIR = Path("artifacts/site")
DEFAULT_RECORDINGS_DIR = Path("recordings")
DEFAULT_STYLESHEET = Path("web/site.css")
DEFAULT_FAVICON = Path("web/favicon.svg")


class Args(argparse.Namespace):
    """Typed view of parsed CLI args. Class-level defaults satisfy the
    type checker; argparse overwrites or sets them on parse_args()."""

    cmd: str = ""
    sample_rate: int = DEFAULT_SAMPLE_RATE
    input: Path = DEFAULT_INPUT
    output_dir: Path = DEFAULT_OUTPUT_DIR
    crossover: float = DEFAULT_CROSSOVER_HZ
    order: int = DEFAULT_CROSSOVER_ORDER
    split_at: float = DEFAULT_SPLIT_AT
    rpm: float | None = None
    rpm_trace: Path | None = None
    orders: tuple[float, ...] = DEFAULT_REFERENCE_ORDERS
    video_fps: float | None = None
    video: Path | None = None
    video_start_offset: float | None = None
    control: Path | None = None
    json_output: Path | None = None
    camera_target: Path | None = None
    output: Path = DEFAULT_ANALYSIS_PNG
    stylesheet: Path = DEFAULT_STYLESHEET
    favicon: Path = DEFAULT_FAVICON
    func: Callable[[Self], int] | None = None


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


def cmd_modulation(args: Args) -> int:
    if not args.input.exists():
        print(f"missing: {args.input}", file=sys.stderr)
        return 1
    for optional_path in (args.control, args.rpm_trace, args.video):
        if optional_path is not None and not optional_path.exists():
            print(f"missing: {optional_path}", file=sys.stderr)
            return 1
    _ = localization.run(
        input_path=args.input,
        sample_rate=args.sample_rate,
        output_png=args.output,
        rpm=args.rpm,
        rpm_trace_path=args.rpm_trace,
        reference_orders=args.orders,
        control_path=args.control,
        video_path=args.video,
        capture_fps=args.video_fps,
        video_start_offset_s=args.video_start_offset,
        json_output=args.json_output,
        camera_target_output=args.camera_target,
        crossover_hz=args.crossover,
        filter_order=args.order,
    )
    return 0


def cmd_site(args: Args) -> int:
    """Build the full static site under args.output_dir in one process.

    Independent audio/plot jobs run concurrently; the rattles-stem analysis
    waits only for the split output it actually needs.
    """
    if not args.input.exists():
        print(f"missing: {args.input}", file=sys.stderr)
        return 1
    if not args.stylesheet.exists():
        print(f"missing: {args.stylesheet}", file=sys.stderr)
        return 1
    if not args.favicon.exists():
        print(f"missing: {args.favicon}", file=sys.stderr)
        return 1

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    recordings = _recordings(DEFAULT_RECORDINGS_DIR)
    default_spectrogram = out / "spectrogram.png"
    recording_jobs = [
        recording for recording in recordings if not _same_path(recording, args.input)
    ]

    with ProcessPoolExecutor(
        max_workers=_site_worker_count(recording_jobs),
        mp_context=multiprocessing.get_context("fork"),
    ) as pool:
        split_future = pool.submit(
            _run_split,
            args.input,
            out,
            args.sample_rate,
            args.crossover,
            args.order,
        )
        default_spectrogram_future = pool.submit(
            _run_spectrogram,
            args.input,
            args.sample_rate,
            default_spectrogram,
        )
        futures: list[Future[None]] = [
            pool.submit(
                _run_analysis,
                args.input,
                args.sample_rate,
                args.split_at,
                out / "analysis.png",
            ),
            pool.submit(_run_moments, out, args.sample_rate),
        ]
        futures.extend(
            pool.submit(
                _run_spectrogram,
                recording,
                args.sample_rate,
                out / _recording_spectrogram_name(recording),
            )
            for recording in recording_jobs
        )

        split_future.result()
        futures.append(
            pool.submit(
                _run_analysis,
                out / "rattles.wav",
                args.sample_rate,
                args.split_at,
                out / "rattles_analysis.png",
            )
        )
        default_spectrogram_future.result()
        _copy_default_recording_spectrogram(
            input_path=args.input,
            recordings=recordings,
            source=default_spectrogram,
            output_dir=out,
        )
        _finish(futures)
        pool.submit(
            _run_modulation,
            args.input,
            args.sample_rate,
            out / "modulation.png",
            args.crossover,
            args.order,
            args.video_fps,
        ).result()

    input_copy = out / args.input.name
    _ = shutil.copy(args.input, input_copy)
    print(f"wrote {input_copy}")
    (out / "engine.wav").unlink(missing_ok=True)
    (out / "rattles.wav").unlink(missing_ok=True)

    _ = site_builder.build(
        stylesheet_path=args.stylesheet,
        favicon_path=args.favicon,
        output_dir=out,
    )

    print(f"site -> {out}")
    return 0


def _run_split(
    input_path: Path,
    output_dir: Path,
    sample_rate: int,
    crossover_hz: float,
    order: int,
) -> None:
    _ = pipeline.split(
        input_path=input_path,
        output_dir=output_dir,
        sample_rate=sample_rate,
        crossover_hz=crossover_hz,
        order=order,
    )


def _run_spectrogram(input_path: Path, sample_rate: int, output_png: Path) -> None:
    spectrogram.run(
        input_path=input_path, sample_rate=sample_rate, output_png=output_png
    )


def _run_analysis(
    input_path: Path,
    sample_rate: int,
    split_at: float,
    output_png: Path,
) -> None:
    analysis.run(
        input_path=input_path,
        sample_rate=sample_rate,
        split_at=split_at,
        output_png=output_png,
    )


def _run_modulation(
    input_path: Path,
    sample_rate: int,
    output_png: Path,
    crossover_hz: float,
    filter_order: int,
    video_fps: float | None = None,
) -> None:
    json_output = output_png.with_name(DEFAULT_FAULT_REPORT_JSON)
    camera_target_output = output_png.with_name(DEFAULT_CAMERA_TARGETS_CSV)
    output_png.unlink(missing_ok=True)
    json_output.unlink(missing_ok=True)
    camera_target_output.unlink(missing_ok=True)
    try:
        _ = localization.run(
            input_path=input_path,
            sample_rate=sample_rate,
            output_png=output_png,
            crossover_hz=crossover_hz,
            filter_order=filter_order,
            reference_orders=Args.orders,
            capture_fps=video_fps,
            json_output=json_output,
            camera_target_output=camera_target_output,
        )
    except modulation.InsufficientAudioError as error:
        print(f"skipped modulation: {error}")


def _run_moments(output_dir: Path, sample_rate: int) -> None:
    moments.build_default(output_dir, sample_rate)


def _finish(futures: list[Future[None]]) -> None:
    for future in futures:
        future.result()


def _site_worker_count(recording_jobs: list[Path]) -> int:
    job_count = 5 + len(recording_jobs)
    cpu_count = os.cpu_count() or 2
    return min(max(2, job_count), max(2, cpu_count), 6)


def _copy_default_recording_spectrogram(
    *,
    input_path: Path,
    recordings: list[Path],
    source: Path,
    output_dir: Path,
) -> None:
    for recording in recordings:
        if _same_path(recording, input_path):
            target = output_dir / _recording_spectrogram_name(recording)
            _ = shutil.copy(source, target)
            print(f"wrote {target}")
            return


def _same_path(left: Path, right: Path) -> bool:
    return left.resolve() == right.resolve()


def _recordings(path: Path) -> list[Path]:
    if not path.exists():
        return []
    return sorted(
        child
        for child in path.iterdir()
        if child.is_file() and child.suffix.lower() in site_builder.AUDIO_EXTENSIONS
    )


def _recording_spectrogram_name(recording: Path) -> str:
    return f"recording_spectrogram_{_slug(recording.stem)}.png"


def _slug(value: str) -> str:
    chars: list[str] = []
    for char in value.lower():
        if char.isalnum():
            chars.append(char)
        elif chars and chars[-1] != "_":
            chars.append("_")
    return "".join(chars).strip("_") or "recording"


def _positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0.0:
        msg = "must be finite and greater than zero"
        raise argparse.ArgumentTypeError(msg)
    return parsed


def _finite_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        msg = "must be finite"
        raise argparse.ArgumentTypeError(msg)
    return parsed


def _order_list(value: str) -> tuple[float, ...]:
    try:
        orders = tuple(float(item) for item in value.split(","))
    except ValueError as error:
        msg = "must be comma-separated numbers"
        raise argparse.ArgumentTypeError(msg) from error
    if not orders or any(not math.isfinite(order) or order <= 0.0 for order in orders):
        msg = "orders must be finite and positive"
        raise argparse.ArgumentTypeError(msg)
    return orders


INPUT_HELP = (
    "audio input file — any format ffmpeg can decode "
    "(wav, mp3, m4a, flac, ogg, opus, mp4 audio, ...). "
    "Default: the bundled motorcycle recording."
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="engine-rattle-splitter",
        description=(
            "Split engine recordings into a low-band stem (combustion drone) "
            "and a high-band stem (mechanical rattles) via complementary "
            "Butterworth crossover. Works on any audio file ffmpeg can decode."
        ),
        epilog=(
            "Examples:\n"
            "  engine-rattle-splitter separate recordings/ride.flac -o artifacts/stems --crossover 2000\n"
            "  engine-rattle-splitter analyze  recordings/ride.flac --split-at 5.2\n"
            "  engine-rattle-splitter spectrogram recordings/ride.flac\n"
            "  engine-rattle-splitter modulation recordings/steady.wav --rpm 1800"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _ = parser.add_argument(
        "--sample-rate",
        type=int,
        default=DEFAULT_SAMPLE_RATE,
        metavar="HZ",
        help=(
            "decode/resample rate; ffmpeg resamples the input to this (default: %(default)s)"
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
            "(default: %(default)s)"
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

    mod = sub.add_parser(
        "modulation",
        help="measure high-band envelope modulation frequencies",
        description=(
            "Extract the analytic amplitude envelope of the complementary "
            "high-band rattle signal, then report prominent 5-100 Hz envelope "
            "modulation frequencies. Components may reflect repeated events "
            "or beating between tones. With --rpm, also express each measured "
            "frequency as a fixed shaft-speed order; RPM never changes the measurement."
        ),
    )
    _ = mod.add_argument(
        "input", type=Path, nargs="?", default=DEFAULT_INPUT, help=INPUT_HELP
    )
    rpm_group = mod.add_mutually_exclusive_group()
    _ = rpm_group.add_argument(
        "--rpm",
        type=_positive_float,
        default=None,
        metavar="RPM",
        help="fixed engine speed used only to express peaks as shaft orders",
    )
    _ = rpm_group.add_argument(
        "--rpm-trace",
        type=Path,
        default=None,
        metavar="CSV",
        help="time-varying media-relative RPM measurements (time_s,rpm)",
    )
    _ = mod.add_argument(
        "--orders",
        type=_order_list,
        default=DEFAULT_REFERENCE_ORDERS,
        metavar="LIST",
        help="comma-separated order references (default: 0.5,1,1.5,2,3,4)",
    )
    _ = mod.add_argument(
        "--capture-fps",
        "--video-fps",
        dest="video_fps",
        type=_positive_float,
        default=None,
        metavar="FPS",
        help="physical capture FPS override used for camera guidance",
    )
    _ = mod.add_argument(
        "--video",
        type=Path,
        default=None,
        metavar="VIDEO",
        help="probe video timing and align its audio track with the input",
    )
    _ = mod.add_argument(
        "--video-start-offset",
        type=_finite_float,
        default=None,
        metavar="SECONDS",
        help="explicit timeline mapping: video_time = audio_time + offset",
    )
    _ = mod.add_argument(
        "--control",
        type=Path,
        default=None,
        metavar="AUDIO",
        help="comparable rattle-free recording for candidate contrast",
    )
    _ = mod.add_argument(
        "--json",
        dest="json_output",
        type=Path,
        default=None,
        metavar="PATH",
        help="write machine-readable diagnostic summary",
    )
    _ = mod.add_argument(
        "--camera-target",
        type=Path,
        default=None,
        metavar="CSV",
        help="write time-varying motion-magnification targets",
    )
    _ = mod.add_argument(
        "--crossover",
        type=float,
        default=DEFAULT_CROSSOVER_HZ,
        metavar="HZ",
        help="rattle-band crossover frequency in Hz (default: %(default)s)",
    )
    _ = mod.add_argument(
        "--filter-order",
        dest="order",
        type=int,
        default=DEFAULT_CROSSOVER_ORDER,
        metavar="N",
        help="Butterworth crossover filter order (default: %(default)s)",
    )
    _ = mod.add_argument(
        "-o",
        "--output",
        type=Path,
        default=DEFAULT_MODULATION_PNG,
        metavar="PNG",
        help="PNG output path (default: %(default)s)",
    )
    mod.set_defaults(func=cmd_modulation)

    st = sub.add_parser(
        "site",
        help="build the full GitHub Pages site (stems + plots + html) into a dir",
        description=(
            "Run separate + spectrogram + analyze + analyze-on-rattles in a "
            "single Python process and assemble a self-contained static site "
            "in --output-dir, ready for actions/upload-pages-artifact. One "
            "process instead of four kills the cold-start / numpy / matplotlib "
            "import overhead — this is the CI build step."
        ),
    )
    _ = st.add_argument(
        "input", type=Path, nargs="?", default=DEFAULT_INPUT, help=INPUT_HELP
    )
    _ = st.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=DEFAULT_SITE_DIR,
        metavar="DIR",
        help="site output directory (default: %(default)s)",
    )
    _ = st.add_argument(
        "--stylesheet",
        type=Path,
        default=DEFAULT_STYLESHEET,
        metavar="CSS",
        help="path to site stylesheet (default: %(default)s)",
    )
    _ = st.add_argument(
        "--favicon",
        type=Path,
        default=DEFAULT_FAVICON,
        metavar="SVG",
        help="path to site favicon (default: %(default)s)",
    )
    _ = st.add_argument(
        "--crossover",
        type=float,
        default=DEFAULT_CROSSOVER_HZ,
        metavar="HZ",
        help="crossover frequency in Hz (default: %(default)s)",
    )
    _ = st.add_argument(
        "--order",
        type=int,
        default=DEFAULT_CROSSOVER_ORDER,
        metavar="N",
        help="Butterworth filter order (default: %(default)s)",
    )
    _ = st.add_argument(
        "--split-at",
        type=float,
        default=DEFAULT_SPLIT_AT,
        metavar="SECONDS",
        help="time in seconds for analyze split (default: %(default)s)",
    )
    _ = st.add_argument(
        "--video-fps",
        type=_positive_float,
        default=DEFAULT_VIDEO_FPS,
        metavar="FPS",
        help="video sampling reference shown on modulation plot (default: %(default)s)",
    )
    st.set_defaults(func=cmd_site)

    return parser


def main() -> int:
    args = build_parser().parse_args(namespace=Args())
    if args.func is None:
        return 2
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
