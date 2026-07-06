# Replace Hardcoded Site Config With Artifact-Driven Build

## Summary

The JSON manifest approach should be removed. It only moved hardcoded site content
out of web/index.html into web/site.json; it did not make the site data-driven.

The corrected implementation will generate the site from actual build inputs,
generated artifacts, and pyproject.toml metadata. web/ will contain reusable
presentation assets only, not committed page content.

## Key Changes

- Remove `web/site.json` and the current config-driven site rendering path.
- Do not restore a committed `web/index.html`; generated HTML belongs under `artifacts/site/index.html`.
- Keep `web/site.css` and `web/favicon.svg` as static presentation assets.
- Add a minimal `web/template.html` only for document structure and insertion points.
  It must not contain project-specific copy, audio filenames, image filenames, or
  card content.
- Generate site data inside the CLI from:
  - `[project]` metadata in `pyproject.toml`
  - `[project.urls].Source`, adding it if missing
  - the selected input audio path
  - split parameters such as crossover frequency
  - generated output artifacts from the pipeline
  - generated analysis image paths
- Generate labels from artifact roles and filenames, for example:
  - `Original`
  - `Engine`
  - `Rattles`
  - `Spectrogram`
  - `Analysis`
- Replace `--site-config` with no config option. Keep only useful asset overrides
  such as `--stylesheet`, `--favicon`, and optionally `--template`.

## Implementation Notes

- Update the site builder so it accepts a typed build model from the CLI instead
  of loading content from JSON.
- Make missing artifacts fail with clear errors before writing the final HTML.
- Fix the current broken `raise Ty(...)` typo by replacing the JSON-loading implementation
  entirely.
- Preserve current repository paths and remotes; this work must not rename directories,
  remotes, or push anything unless separately requested.

## Test Plan

- Run `uv run ruff check`.
- Run `uv run basedpyright`.
- Add a small stdlib unittest test for site rendering:
  - generated HTML references supplied artifact filenames
  - no unresolved template placeholders remain
  - missing required assets raise a clear exception
- Run `uv run python -m unittest`.
- Run `uv run engine-rattle-splitter site`.
- Inspect generated `artifacts/site/index.html` to confirm content comes from build
  data, not a committed JSON or HTML content manifest.

## Assumptions

- “No hardcoding” means no committed final HTML and no committed content manifest.
- CSS, favicon, and a generic template are acceptable source assets.
- Domain terms like engine and rattles may remain as semantic output roles because
  they are part of the tool’s actual behavior.
