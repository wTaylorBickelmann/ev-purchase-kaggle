"""
Build script for marimo notebooks.

Adapted from https://github.com/marimo-team/marimo-gh-pages-template
(`.github/scripts/build.py`): export html-wasm, then write an index.html.

    uv run .github/scripts/build.py [--output-dir OUTPUT_DIR]
"""

# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "jinja2==3.1.3",
#     "fire==0.7.0",
#     "loguru==0.7.0"
# ]
# ///

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import List, Union

import fire
import jinja2
from loguru import logger

WASM_PY = [
    "__init__.py",
    "schema.py",
    "features.py",
    "experiments.py",
    "paths.py",
    "data.py",
    "nb.py",
]

TITLES = {
    "eda": "EDA",
    "features": "Data engineering",
    "training": "Training & experiments",
}


def _stage_public() -> None:
    """Copy WASM-safe library + EXPERIMENTS.md into notebooks/public for html-wasm."""
    pub = Path("notebooks/public")
    pub.mkdir(parents=True, exist_ok=True)
    lib = pub / "ev_s6e9"
    if lib.exists():
        shutil.rmtree(lib)
    lib.mkdir(parents=True)
    src = Path("src/ev_s6e9")
    for name in WASM_PY:
        shutil.copy2(src / name, lib / name)
    exp = Path("EXPERIMENTS.md")
    if exp.exists():
        shutil.copy2(exp, pub / "EXPERIMENTS.md")
    reports = Path("reports")
    if reports.exists():
        dest = pub / "reports"
        dest.mkdir(exist_ok=True)
        for p in reports.glob("*.png"):
            shutil.copy2(p, dest / p.name)
    logger.info(f"Staged WASM assets under {pub}")


def _export_html_wasm(notebook_path: Path, output_dir: Path, as_app: bool = False) -> bool:
    output_path: Path = notebook_path.with_suffix(".html")
    cmd: List[str] = ["uvx", "marimo", "export", "html-wasm", "--sandbox"]
    if as_app:
        logger.info(f"Exporting {notebook_path} to {output_path} as app")
        cmd.extend(["--mode", "run", "--no-show-code"])
    else:
        logger.info(f"Exporting {notebook_path} to {output_path} as notebook")
        cmd.extend(["--mode", "edit"])
    try:
        output_file: Path = output_dir / notebook_path.with_suffix(".html")
        output_file.parent.mkdir(parents=True, exist_ok=True)
        cmd.extend([str(notebook_path), "-o", str(output_file)])
        logger.debug(f"Running command: {cmd}")
        subprocess.run(cmd, capture_output=True, text=True, check=True)
        logger.info(f"Successfully exported {notebook_path}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Error exporting {notebook_path}:")
        logger.error(f"Command output: {e.stderr}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error exporting {notebook_path}: {e}")
        return False


def _generate_index(
    output_dir: Path,
    template_file: Path,
    notebooks_data: List[dict] | None = None,
    apps_data: List[dict] | None = None,
) -> None:
    logger.info("Generating index.html")
    index_path: Path = output_dir / "index.html"
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(template_file.parent),
            autoescape=jinja2.select_autoescape(["html", "xml"]),
        )
        template = env.get_template(template_file.name)
        rendered_html = template.render(notebooks=notebooks_data, apps=apps_data)
        index_path.write_text(rendered_html, encoding="utf-8")
        logger.info(f"Successfully generated index.html at {index_path}")
    except OSError as e:
        logger.error(f"Error generating index.html: {e}")
    except jinja2.exceptions.TemplateError as e:
        logger.error(f"Error rendering template: {e}")


def _export(folder: Path, output_dir: Path, as_app: bool = False) -> List[dict]:
    if not folder.exists():
        logger.warning(f"Directory not found: {folder}")
        return []
    notebooks = sorted(
        p for p in folder.rglob("*.py") if "public" not in p.parts
    )
    logger.debug(f"Found {len(notebooks)} Python files in {folder}")
    if not notebooks:
        logger.warning(f"No notebooks found in {folder}!")
        return []
    notebook_data = [
        {
            "display_name": TITLES.get(nb.stem, nb.stem.replace("_", " ").title()),
            "html_path": str(nb.with_suffix(".html")),
        }
        for nb in notebooks
        if _export_html_wasm(nb, output_dir, as_app=as_app)
    ]
    logger.info(
        f"Successfully exported {len(notebook_data)} out of {len(notebooks)} files from {folder}"
    )
    return notebook_data


def main(
    output_dir: Union[str, Path] = "_site",
    template: Union[str, Path] = "templates/index.html.j2",
) -> None:
    logger.info("Starting marimo build process")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / ".nojekyll").write_text("", encoding="utf-8")
    _stage_public()
    template_file = Path(template)
    notebooks_data = _export(Path("notebooks"), output_dir, as_app=False)
    apps_data = _export(Path("apps"), output_dir, as_app=True)
    if not notebooks_data and not apps_data:
        logger.warning("No notebooks or apps found!")
        return
    _generate_index(
        output_dir=output_dir,
        notebooks_data=notebooks_data,
        apps_data=apps_data,
        template_file=template_file,
    )
    logger.info(f"Build completed successfully. Output directory: {output_dir}")


if __name__ == "__main__":
    fire.Fire(main)
