"""Configuration file for the Sphinx documentation builder."""

import os
import sys

# Add the parent directory to the path so we can import fewlab
sys.path.insert(0, os.path.abspath(".."))

# Project information
project = "fewlab"
copyright = "2024, Gaurav Sood"
author = "Gaurav Sood"
release = "0.1.0"

# Extensions
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.autosummary",
    "sphinx.ext.mathjax",
]

# Napoleon settings for NumPy-style docstrings
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_type_aliases = None
napoleon_attr_annotations = True

# Autodoc settings
autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "special-members": "__init__",
    "undoc-members": True,
    "exclude-members": "__weakref__",
}
autodoc_typehints = "description"
autodoc_preserve_defaults = True

# Autosummary settings
autosummary_generate = True

# Intersphinx mapping to link to other documentation
intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
}

# Templates path
templates_path = ["_templates"]

# List of patterns to ignore when looking for source files
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# HTML output
html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_title = f"{project} v{release}"
html_short_title = project
html_favicon = None

# Theme options
html_theme_options = {
    "display_version": True,
    "prev_next_buttons_location": "bottom",
    "style_external_links": False,
    "vcs_pageview_mode": "",
    "style_nav_header_background": "#2980B9",
    # Toc options
    "collapse_navigation": True,
    "sticky_navigation": True,
    "navigation_depth": 4,
    "includehidden": True,
    "titles_only": False,
}

# Add any paths that contain custom static files (such as style sheets)
html_static_path = []

# Add source code link
html_show_sourcelink = True

# Math support
mathjax3_config = {
    "tex": {
        "inlineMath": [["$", "$"], ["\\(", "\\)"]],
        "displayMath": [["$$", "$$"], ["\\[", "\\]"]],
    }
}