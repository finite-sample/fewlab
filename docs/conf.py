"""Sphinx configuration — fleet standard via py-canon."""

from py_canon.sphinx import configure

configure(globals())

# Repo-specific additions on top of the fleet standard. The theme and the
# extension list stay with py-canon; only what is true of fewlab alone is here.

# fewlab's public signatures are built out of numpy and pandas types, so the
# rendered API pages are much more useful when those resolve.
intersphinx_mapping.update(  # noqa: F821  # set by configure(globals())
    {
        "numpy": ("https://numpy.org/doc/stable/", None),
        "pandas": ("https://pandas.pydata.org/docs/", None),
    }
)

# The marimo notebook and its data are exported by the hook below, not built
# as pages; the data folder's README would otherwise be an orphan document.
exclude_patterns.append("examples")  # noqa: F821  # set by configure(globals())

html_theme_options = {
    "source_repository": "https://github.com/finite-sample/fewlab/",
    "source_branch": "main",
    "source_directory": "docs/",
}


# The marimo notebook is published as a page that runs Python in the reader's
# browser. Running it here first against this checkout catches API drift before
# release; the export then runs it again against fewlab from PyPI, which is what
# the browser installs, and embeds the outputs so the page shows results while
# Python loads. Either failing fails the build.
def _export_notebook(app):
    if app.builder.name != "html":
        return
    import subprocess
    import sys
    from pathlib import Path

    notebook = Path(__file__).parent / "examples" / "label_budget.py"
    # S603: both commands run fixed paths inside this repository.
    subprocess.run([sys.executable, str(notebook)], check=True)  # noqa: S603
    subprocess.run(  # noqa: S603
        [
            sys.executable,
            "-m",
            "marimo",
            "export",
            "html-wasm",
            str(notebook),
            "-o",
            str(Path(app.outdir) / "demo"),
            "--mode",
            "run",
            "--execute",
            "--force",
        ],
        check=True,
    )


def setup(app):
    app.connect("builder-inited", _export_notebook)
