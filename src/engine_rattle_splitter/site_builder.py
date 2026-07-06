"""Static site renderer for the artifacts produced by a build."""

import html
import os
import shutil
import subprocess
import tomllib
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast
from urllib.parse import quote

type ArtifactKind = Literal["audio", "image", "file"]

AUDIO_EXTENSIONS = {
    ".aac",
    ".aiff",
    ".flac",
    ".m4a",
    ".mp3",
    ".oga",
    ".ogg",
    ".opus",
    ".wav",
    ".weba",
}
IMAGE_EXTENSIONS = {".avif", ".gif", ".jpeg", ".jpg", ".png", ".svg", ".webp"}
DEFAULT_PYPROJECT = Path("pyproject.toml")
SITE_LEDE = (
    "Splits a recording into low-band (combustion drone) and high-band "
    "(mechanical rattles) stems via complementary Butterworth crossover."
)
AUDIO_ORDER = (
    "Shitty motor (goede recording).m4a",
    "engine.mp3",
    "rattles.mp3",
)
IMAGE_ORDER = ("spectrogram.png", "analysis.png", "rattles_analysis.png")
RECORDING_SPECTROGRAM_PREFIX = "recording_spectrogram_"


@dataclass(frozen=True)
class Artifact:
    path: Path
    kind: ArtifactKind
    size_bytes: int


@dataclass(frozen=True)
class ProjectMetadata:
    title: str
    description: str
    source_url: str | None


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>{title}</title>
    {favicon_link}
    <link rel="stylesheet" href="{stylesheet}">
  </head>
  <body>
    <main>
      <header class="site-header">
        <div>
          <h1>{title}</h1>
          <p class="lede">{lede}</p>
        </div>
        {source_link}
      </header>
{sections}
{footer}
    </main>
  </body>
</html>
"""


def build(
    *,
    output_dir: Path,
    stylesheet_path: Path,
    favicon_path: Path | None,
    pyproject_path: Path = DEFAULT_PYPROJECT,
) -> Path:
    """Render index.html from the files that exist in output_dir."""
    output_dir.mkdir(parents=True, exist_ok=True)
    stylesheet_out = _copy_required(stylesheet_path, output_dir)
    favicon_out = _copy_optional(favicon_path, output_dir)
    metadata = _load_project_metadata(pyproject_path)

    excluded = {"index.html", _relative_name(stylesheet_out, output_dir)}
    if favicon_out is not None:
        excluded.add(_relative_name(favicon_out, output_dir))

    artifacts = discover(output_dir, exclude=excluded)
    html_out = output_dir / "index.html"
    _ = html_out.write_text(
        _render_html(
            metadata=metadata,
            artifacts=artifacts,
            stylesheet=stylesheet_out.relative_to(output_dir),
            favicon=favicon_out.relative_to(output_dir) if favicon_out else None,
        ),
        encoding="utf-8",
    )
    print(f"wrote {html_out}")
    return html_out


def discover(output_dir: Path, *, exclude: Iterable[str] = ()) -> list[Artifact]:
    """Return renderable artifacts found under output_dir."""
    excluded = set(exclude)
    artifacts: list[Artifact] = []
    for path in output_dir.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(output_dir)
        if relative.as_posix() in excluded or any(
            part.startswith(".") for part in relative.parts
        ):
            continue
        kind = _kind_for(relative)
        artifacts.append(
            Artifact(path=relative, kind=kind, size_bytes=path.stat().st_size)
        )
    return sorted(
        artifacts, key=lambda artifact: (artifact.kind, artifact.path.as_posix())
    )


def _kind_for(path: Path) -> ArtifactKind:
    suffix = path.suffix.lower()
    if suffix in AUDIO_EXTENSIONS:
        return "audio"
    if suffix in IMAGE_EXTENSIONS:
        return "image"
    return "file"


def _copy_required(source: Path, output_dir: Path) -> Path:
    if not source.exists():
        msg = f"missing: {source}"
        raise FileNotFoundError(msg)
    target = output_dir / source.name
    _ = shutil.copy(source, target)
    print(f"wrote {target}")
    return target


def _copy_optional(source: Path | None, output_dir: Path) -> Path | None:
    if source is None or not source.exists():
        return None
    target = output_dir / source.name
    _ = shutil.copy(source, target)
    print(f"wrote {target}")
    return target


def _load_project_metadata(pyproject_path: Path) -> ProjectMetadata:
    project: Mapping[str, object] = {}
    if pyproject_path.exists():
        data = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
        project = _as_mapping(data.get("project"))

    title = _string_value(project.get("name"), fallback="Generated audio")
    description = _string_value(
        project.get("description"),
        fallback="Generated audio and diagnostic plots.",
    )
    return ProjectMetadata(
        title=title,
        description=description,
        source_url=_source_url(),
    )


def _source_url() -> str | None:
    repository = os.environ.get("GITHUB_REPOSITORY")
    if repository:
        server = os.environ.get("GITHUB_SERVER_URL", "https://github.com").rstrip("/")
        return f"{server}/{repository}"

    return _git_origin_url()


def _git_origin_url() -> str | None:
    result = subprocess.run(
        ["git", "config", "--get", "remote.origin.url"],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None
    return _normalize_git_url(result.stdout.strip())


def _normalize_git_url(value: str) -> str | None:
    if not value:
        return None
    if value.startswith("git@github.com:"):
        value = f"https://github.com/{value.removeprefix('git@github.com:')}"
    value = value.removesuffix(".git")
    return value


def _render_html(
    *,
    metadata: ProjectMetadata,
    artifacts: list[Artifact],
    stylesheet: Path,
    favicon: Path | None,
) -> str:
    return HTML_TEMPLATE.format(
        title=_e(metadata.title),
        lede=_e(SITE_LEDE),
        source_link=_render_source_link(metadata.source_url),
        favicon_link=_render_favicon_link(favicon),
        stylesheet=_asset_url(stylesheet),
        sections=_render_sections(artifacts),
        footer=_render_footer(metadata.source_url),
    )


def _render_source_link(source_url: str | None) -> str:
    if source_url is None:
        return ""
    return f'<a class="source-link" href="{_e(source_url)}">Source on GitHub</a>'


def _render_footer(source_url: str | None) -> str:
    if source_url is None:
        return ""
    repository = source_url.removeprefix("https://github.com/")
    return (
        "\n      <footer>Generated from "
        f'<a href="{_e(source_url)}">{_e(repository)}</a>.'
        "</footer>"
    )


def _render_favicon_link(favicon: Path | None) -> str:
    if favicon is None:
        return ""
    return f'<link rel="icon" type="image/svg+xml" href="{_asset_url(favicon)}">'


def _render_sections(artifacts: list[Artifact]) -> str:
    finding_artifacts = [
        artifact for artifact in artifacts if _is_finding_artifact(artifact)
    ]
    generic_artifacts = [
        artifact for artifact in artifacts if not _is_finding_artifact(artifact)
    ]
    recording_spectrograms = [
        artifact
        for artifact in generic_artifacts
        if _is_recording_spectrogram(artifact)
    ]
    audio = [artifact for artifact in generic_artifacts if artifact.kind == "audio"]
    images = [
        artifact
        for artifact in generic_artifacts
        if artifact.kind == "image" and not _is_recording_spectrogram(artifact)
    ]
    files = [artifact for artifact in generic_artifacts if artifact.kind == "file"]

    sections = [
        _render_audio_section(audio),
        _render_finding_section(finding_artifacts),
        _render_recording_spectrogram_section(recording_spectrograms),
        _render_image_section(images),
        _render_file_section(files),
    ]
    rendered = [section for section in sections if section]
    if rendered:
        return "\n".join(rendered)
    return '\n      <p class="empty">No audio or plots were generated.</p>\n'


def _is_finding_artifact(artifact: Artifact) -> bool:
    return artifact.path.name.startswith("finding_3m56_3m59_")


def _is_recording_spectrogram(artifact: Artifact) -> bool:
    return artifact.path.name.startswith(RECORDING_SPECTROGRAM_PREFIX)


def _render_finding_section(artifacts: list[Artifact]) -> str:
    if not artifacts:
        return ""

    by_name = {artifact.path.name: artifact for artifact in artifacts}
    context = by_name.get("finding_3m56_3m59_context.wav")
    target = by_name.get("finding_3m56_3m59_target.wav")
    highpass = by_name.get("finding_3m56_3m59_target_highpass_2k.mp3")
    spectrogram = by_name.get("finding_3m56_3m59_context_spectrogram.png")
    delta = by_name.get("finding_3m56_3m59_delta_bands.png")

    cards = [
        "\n      <h2>Notable Moment</h2>",
        '      <section class="finding">',
        "        <h3>Laatste stuk, 3:56-3:59</h3>",
        "        <p>Broad high-frequency rattle: total level is slightly lower, "
        "but the 2-24 kHz bands rise by about 1.8-3.0 dB. Peak is around "
        "3:57.5-3:58.0.</p>",
    ]

    for label, artifact in (
        ("Context, 3:53-4:02", context),
        ("Target, 3:56-3:59", target),
        ("Target high-pass above 2 kHz", highpass),
    ):
        if artifact is None:
            continue
        url = _asset_url(artifact.path)
        cards.append(
            '        <div class="card compact">\n'
            f'          <h4>{_e(label)} <span class="src">{_e(_file_label(artifact))}</span></h4>\n'
            f'          <audio controls preload="metadata" src="{url}"></audio>\n'
            "        </div>"
        )

    for label, artifact in (
        ("Spectrogram context", spectrogram),
        ("Band delta vs surrounding context", delta),
    ):
        if artifact is None:
            continue
        cards.append(_render_figure(artifact, label, indent="        "))

    cards.append("      </section>")
    return "\n".join(cards)


def _render_recording_spectrogram_section(artifacts: list[Artifact]) -> str:
    if not artifacts:
        return ""

    rendered = [
        "\n      <h2>Recording Spectrograms</h2>",
        '      <p class="section-copy">Every file in ./recordings/, rendered the same way so bad bands and timing patterns compare cleanly.</p>',
    ]
    for artifact in artifacts:
        rendered.append(
            _render_figure(artifact, _recording_spectrogram_caption(artifact))
        )
    return "\n".join(rendered)


def _recording_spectrogram_caption(artifact: Artifact) -> str:
    stem = artifact.path.stem.removeprefix(RECORDING_SPECTROGRAM_PREFIX)
    return f"recording spectrogram: {stem.replace('_', ' ')}"


def _render_audio_section(artifacts: list[Artifact]) -> str:
    if not artifacts:
        return ""
    cards: list[str] = ["\n      <h2>Audio</h2>"]
    for artifact in _ordered_artifacts(artifacts, AUDIO_ORDER):
        title, label, description = _audio_copy(artifact)
        url = _asset_url(artifact.path)
        cards.append(
            '      <div class="card">\n'
            f'        <h3>{_e(title)} <span class="src">{_e(label)}</span></h3>\n'
            f"        <p>{_e(description)}</p>\n"
            f'        <audio controls preload="metadata" src="{url}"></audio>\n'
            f'        <p><a href="{url}" download>Download</a></p>\n'
            "      </div>"
        )
    return "\n\n".join(cards)


def _render_image_section(artifacts: list[Artifact]) -> str:
    if not artifacts:
        return ""
    ordered = _ordered_artifacts(artifacts, IMAGE_ORDER)
    rendered: list[str] = []

    spectrogram = _artifact_named(ordered, "spectrogram.png")
    if spectrogram is not None:
        rendered.extend([
            "\n      <h2>Spectrogram</h2>",
            '      <p class="section-copy">log-frequency dB spectrogram of the original recording</p>',
            _render_figure(spectrogram, "spectrogram of the original recording"),
        ])

    analysis = [
        artifact for artifact in ordered if artifact.path.name != "spectrogram.png"
    ]
    if analysis:
        rendered.extend([
            "\n      <h2>Analysis</h2>",
            '      <p class="section-copy">Frame features and per-octave-band energy contrast across the 13 s mark.</p>',
        ])

    for artifact in analysis:
        _, _, caption = _image_copy(artifact)
        rendered.append(_render_figure(artifact, caption))
        if artifact.path.name == "analysis.png":
            rendered.append(
                '      <p class="section-copy">Same analysis applied to the rattles stem validates the split.</p>'
            )

    return "\n".join(rendered)


def _render_figure(artifact: Artifact, caption: str, *, indent: str = "      ") -> str:
    label = _file_label(artifact)
    asset_url = _asset_url(artifact.path)
    lightbox_id = _element_id("image", artifact.path)
    return (
        f'{indent}<input class="lightbox-toggle" id="{lightbox_id}" type="checkbox" aria-hidden="true">\n'
        f"{indent}<figure>\n"
        f'{indent}  <label class="image-link" for="{lightbox_id}" aria-label="Open {_e(caption)} large">\n'
        f'{indent}    <img src="{asset_url}" alt="{_e(caption)}">\n'
        f"{indent}  </label>\n"
        f'{indent}  <figcaption>{_e(caption)} <span class="src">{_e(label)}</span></figcaption>\n'
        f"{indent}</figure>\n"
        f'{indent}<div class="lightbox" role="dialog" aria-modal="true" aria-label="{_e(caption)}">\n'
        f'{indent}  <label class="lightbox-backdrop" for="{lightbox_id}" aria-label="Close image"></label>\n'
        f'{indent}  <label class="lightbox-close" for="{lightbox_id}" aria-label="Close image">Close</label>\n'
        f'{indent}  <img src="{asset_url}" alt="{_e(caption)}">\n'
        f"{indent}</div>"
    )


def _ordered_artifacts(
    artifacts: list[Artifact], names: tuple[str, ...]
) -> list[Artifact]:
    known = [
        artifact
        for name in names
        for artifact in artifacts
        if artifact.path.name == name
    ]
    unknown = [artifact for artifact in artifacts if artifact.path.name not in names]
    return [*known, *unknown]


def _artifact_named(artifacts: list[Artifact], name: str) -> Artifact | None:
    for artifact in artifacts:
        if artifact.path.name == name:
            return artifact
    return None


def _audio_copy(artifact: Artifact) -> tuple[str, str, str]:
    match artifact.path.name:
        case "Shitty motor (goede recording).m4a":
            return (
                "Original recording",
                "m4a",
                "Untouched input - motorcycle engine with intermittent rattles.",
            )
        case "engine.mp3":
            return (
                "Engine stem",
                "mp3 - low band, < 1800 Hz",
                "Combustion drone, idle fundamentals, low-mid body.",
            )
        case "rattles.mp3":
            return (
                "Rattles stem",
                "mp3 - high band, > 1800 Hz",
                "Mechanical rattles, knocks, dangling-part chatter.",
            )

    return (
        _display_name(artifact.path),
        _file_label(artifact),
        "Generated audio artifact.",
    )


def _image_copy(artifact: Artifact) -> tuple[str, str, str]:
    match artifact.path.name:
        case "analysis.png":
            return (
                "Analysis",
                _file_label(artifact),
                "analysis plot of the original recording",
            )
        case "rattles_analysis.png":
            return (
                "Rattles analysis",
                _file_label(artifact),
                "analysis plot of the rattles stem",
            )
        case "spectrogram.png":
            return (
                "Spectrogram",
                _file_label(artifact),
                "spectrogram of the original recording",
            )

    title = _display_name(artifact.path)
    return (title, _file_label(artifact), title)


def _render_file_section(artifacts: list[Artifact]) -> str:
    if not artifacts:
        return ""
    rendered = ["\n      <h2>Downloads</h2>", '      <ul class="files">']
    for artifact in artifacts:
        url = _asset_url(artifact.path)
        rendered.append(
            "        <li>"
            f'<a href="{url}">{_e(artifact.path.as_posix())}</a> '
            f'<span class="src">{_e(_file_label(artifact))}</span>'
            "</li>"
        )
    rendered.append("      </ul>")
    return "\n".join(rendered)


def _display_name(path: Path) -> str:
    return path.stem.replace("_", " ").replace("-", " ").strip().title() or path.name


def _file_label(artifact: Artifact) -> str:
    suffix = artifact.path.suffix.lower().lstrip(".") or "file"
    return f"{suffix} - {_format_size(artifact.size_bytes)}"


def _format_size(size_bytes: int) -> str:
    size = float(size_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.1f} {unit}" if unit != "B" else f"{int(size)} B"
        size /= 1024
    return f"{size_bytes} B"


def _relative_name(path: Path, output_dir: Path) -> str:
    return path.relative_to(output_dir).as_posix()


def _asset_url(path: Path) -> str:
    return "/".join(quote(part, safe="") for part in path.as_posix().split("/"))


def _element_id(prefix: str, path: Path) -> str:
    chars: list[str] = []
    for char in path.stem.lower():
        if char.isalnum():
            chars.append(char)
        elif chars and chars[-1] != "-":
            chars.append("-")
    slug = "".join(chars).strip("-") or "asset"
    return f"{prefix}-{slug}"


def _as_mapping(value: object) -> Mapping[str, object]:
    if isinstance(value, dict):
        return cast(Mapping[str, object], value)
    return {}


def _string_value(value: object, *, fallback: str) -> str:
    if isinstance(value, str) and value:
        return value
    return fallback


def _e(value: str) -> str:
    return html.escape(value, quote=True)
