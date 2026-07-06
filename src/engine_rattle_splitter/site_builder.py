"""Static site renderer for generated stems and diagnostic plots."""

import html
import json
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Literal, TypedDict, cast
from urllib.parse import quote

type SectionKind = Literal["audio", "image"]


class LinkConfig(TypedDict):
    label: str
    url: str


class FooterConfig(TypedDict):
    prefix: str


class SiteItem(TypedDict):
    asset: str
    title: str
    label: str
    description: str
    alt: str


class SiteSection(TypedDict):
    kind: SectionKind
    heading: str
    items: list[SiteItem]


class SiteConfig(TypedDict):
    title: str
    lede: str
    source: LinkConfig
    repo: LinkConfig
    footer: FooterConfig
    sections: list[SiteSection]


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
\t<head>
\t\t<meta charset="utf-8">
\t\t<meta name="viewport" content="width=device-width, initial-scale=1">
\t\t<title>{title}</title>
\t\t<link rel="icon" type="image/svg+xml" href="{favicon}">
\t\t<link rel="stylesheet" href="{stylesheet}">
\t</head>
\t<body>
\t\t<main>
\t\t\t<h1>{title}</h1>
\t\t\t<p class="lede">
\t\t\t\t{lede}
\t\t\t\t<a href="{source_url}">{source_label}</a>.
\t\t\t</p>
{sections}
\t\t\t<footer>
\t\t\t\t{footer_prefix} <a href="{repo_url}">{repo_label}</a>.
\t\t\t</footer>
\t\t</main>
\t</body>
</html>
"""


def build(
    *,
    config_path: Path,
    stylesheet_path: Path,
    favicon_path: Path | None,
    output_dir: Path,
    assets: Mapping[str, Path],
    values: Mapping[str, object],
) -> Path:
    """Render the site HTML and copy static web assets into output_dir."""
    config = _load_config(config_path)
    stylesheet_out = _copy_required(stylesheet_path, output_dir)
    favicon_out = _copy_optional(favicon_path, output_dir)

    html_out = output_dir / "index.html"
    html_out.write_text(
        _render_html(
            config=config,
            assets=assets,
            values=values,
            stylesheet=stylesheet_out.name,
            favicon=favicon_out.name if favicon_out else "",
        ),
        encoding="utf-8",
    )
    return html_out


def _load_config(path: Path) -> SiteConfig:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        msg = f"{path} must contain a JSON object"
        raise TypeError(msg)
    return cast(SiteConfig, data)


def _copy_required(source: Path, output_dir: Path) -> Path:
    if not source.exists():
        msg = f"missing: {source}"
        raise FileNotFoundError(msg)
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / source.name
    _ = shutil.copy(source, target)
    return target


def _copy_optional(source: Path | None, output_dir: Path) -> Path | None:
    if source is None or not source.exists():
        return None
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / source.name
    _ = shutil.copy(source, target)
    return target


def _render_html(
    *,
    config: SiteConfig,
    assets: Mapping[str, Path],
    values: Mapping[str, object],
    stylesheet: str,
    favicon: str,
) -> str:
    source = config["source"]
    repo = config["repo"]
    return HTML_TEMPLATE.format(
        title=_e(config["title"]),
        lede=_e(_fmt(config["lede"], values)),
        source_url=_e(source["url"]),
        source_label=_e(source["label"]),
        repo_url=_e(repo["url"]),
        repo_label=_e(repo["label"]),
        footer_prefix=_e(config["footer"]["prefix"]),
        stylesheet=_asset_url(stylesheet),
        favicon=_asset_url(favicon),
        sections=_render_sections(config["sections"], assets, values),
    )


def _render_sections(
    sections: list[SiteSection],
    assets: Mapping[str, Path],
    values: Mapping[str, object],
) -> str:
    rendered: list[str] = []
    for section in sections:
        items = section["items"]
        if section["kind"] == "audio":
            body = _render_audio_items(items, assets, values)
        else:
            body = _render_image_items(items, assets, values)
        rendered.append(f"\n\t\t\t<h2>{_e(section['heading'])}</h2>\n{body}")
    return "\n".join(rendered)


def _render_audio_items(
    items: list[SiteItem],
    assets: Mapping[str, Path],
    values: Mapping[str, object],
) -> str:
    cards: list[str] = []
    for item in items:
        src = _lookup_asset(item["asset"], assets)
        title = _e(_fmt(item["title"], values))
        label = _e(_fmt(item["label"], values))
        description = _e(_fmt(item["description"], values))
        cards.append(
            '\t\t\t<div class="card">\n'
            f'\t\t\t\t<h3>{title} <span class="src">{label}</span></h3>\n'
            f"\t\t\t\t<p>{description}</p>\n"
            f'\t\t\t\t<audio controls preload="metadata" src="{src}"></audio>\n'
            "\t\t\t</div>"
        )
    return "\n\n".join(cards)


def _render_image_items(
    items: list[SiteItem],
    assets: Mapping[str, Path],
    values: Mapping[str, object],
) -> str:
    rendered: list[str] = []
    for item in items:
        description = _fmt(item["description"], values)
        if description:
            rendered.append(f'\t\t\t<p class="src">{_e(description)}</p>')
        rendered.append(
            f'\t\t\t<img src="{_lookup_asset(item["asset"], assets)}" '
            f'alt="{_e(_fmt(item["alt"], values))}">'
        )
    return "\n".join(rendered)


def _lookup_asset(key: str, assets: Mapping[str, Path]) -> str:
    try:
        path = assets[key]
    except KeyError as exc:
        msg = f"site config references unknown asset: {key}"
        raise KeyError(msg) from exc
    return _asset_url(path.name)


def _fmt(value: str, values: Mapping[str, object]) -> str:
    return value.format_map(values)


def _asset_url(value: str) -> str:
    return quote(value, safe="")


def _e(value: str) -> str:
    return html.escape(value, quote=True)
