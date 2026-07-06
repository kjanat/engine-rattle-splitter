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
    html_out.write_text(
        _render_html(
            metadata=metadata,
            artifacts=artifacts,
            stylesheet=stylesheet_out.relative_to(output_dir),
            favicon=favicon_out.relative_to(output_dir) if favicon_out else None,
        ),
        encoding="utf-8",
    )
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
    return target


def _copy_optional(source: Path | None, output_dir: Path) -> Path | None:
    if source is None or not source.exists():
        return None
    target = output_dir / source.name
    _ = shutil.copy(source, target)
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
    if value.endswith(".git"):
        value = value[:-4]
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
        lede=_e(metadata.description),
        source_link=_render_source_link(metadata.source_url),
        favicon_link=_render_favicon_link(favicon),
        stylesheet=_asset_url(stylesheet),
        sections=_render_sections(artifacts),
    )


def _render_source_link(source_url: str | None) -> str:
    if source_url is None:
        return ""
    return f'<a class="source-link" href="{_e(source_url)}">Source</a>'


def _render_favicon_link(favicon: Path | None) -> str:
    if favicon is None:
        return ""
    return f'<link rel="icon" type="image/svg+xml" href="{_asset_url(favicon)}">'


def _render_sections(artifacts: list[Artifact]) -> str:
    audio = [artifact for artifact in artifacts if artifact.kind == "audio"]
    images = [artifact for artifact in artifacts if artifact.kind == "image"]
    files = [artifact for artifact in artifacts if artifact.kind == "file"]

    sections = [
        _render_audio_section(audio),
        _render_image_section(images),
        _render_file_section(files),
    ]
    rendered = [section for section in sections if section]
    if rendered:
        return "\n".join(rendered)
    return '\n      <p class="empty">No audio or plots were generated.</p>\n'


def _render_audio_section(artifacts: list[Artifact]) -> str:
    if not artifacts:
        return ""
    cards: list[str] = ["\n      <h2>Audio</h2>"]
    for artifact in artifacts:
        title = _e(_display_name(artifact.path))
        label = _e(_file_label(artifact))
        url = _asset_url(artifact.path)
        cards.append(
            '      <div class="card">\n'
            f'        <h3>{title} <span class="src">{label}</span></h3>\n'
            f'        <audio controls preload="metadata" src="{url}"></audio>\n'
            f'        <p><a href="{url}" download>Download</a></p>\n'
            "      </div>"
        )
    return "\n\n".join(cards)


def _render_image_section(artifacts: list[Artifact]) -> str:
    if not artifacts:
        return ""
    rendered: list[str] = ["\n      <h2>Plots</h2>"]
    for artifact in artifacts:
        title = _e(_display_name(artifact.path))
        label = _e(_file_label(artifact))
        rendered.append(
            "      <figure>\n"
            f'        <img src="{_asset_url(artifact.path)}" alt="{title}">\n'
            f'        <figcaption>{title} <span class="src">{label}</span></figcaption>\n'
            "      </figure>"
        )
    return "\n".join(rendered)


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
