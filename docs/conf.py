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

html_theme_options = {
    "source_repository": "https://github.com/finite-sample/fewlab/",
    "source_branch": "main",
    "source_directory": "docs/",
}
