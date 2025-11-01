"""Configuration file for the Sphinx documentation builder."""

import os
import sys

# Add the parent directory to the path so we can import fewlab
sys.path.insert(0, os.path.abspath(".."))

# Project information
project = "fewlab"
copyright = "2024, Gaurav Sood"
author = "Gaurav Sood"
release = "0.2.0"

# Extensions
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.autosummary",
    "sphinx.ext.mathjax",
    "sphinx_design",
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
html_theme = "furo"
html_title = f"{project} v{release}"
html_short_title = project
html_favicon = None

# Furo theme options with modern dark/light mode support
html_theme_options = {
    "sidebar_hide_name": False,
    "navigation_with_keys": True,
    "top_of_page_button": "edit",
    "source_repository": "https://github.com/finite-sample/fewlab/",
    "source_branch": "main",
    "source_directory": "docs/",
    # Modern color scheme
    "light_css_variables": {
        "color-brand-primary": "#2563eb",  # Modern blue
        "color-brand-content": "#1d4ed8",
        "color-admonition-background": "#f8fafc",
        "color-sidebar-background": "#ffffff",
        "color-sidebar-background-border": "#e2e8f0",
        "font-stack": "system-ui,-apple-system,BlinkMacSystemFont,Segoe UI,Roboto,Oxygen,Ubuntu,Cantarell,sans-serif",
        "font-stack--monospace": "SFMono-Regular,Menlo,Monaco,Consolas,Liberation Mono,Courier New,monospace",
    },
    "dark_css_variables": {
        "color-brand-primary": "#60a5fa",  # Lighter blue for dark mode
        "color-brand-content": "#93c5fd",
        "color-admonition-background": "#1e293b",
        "color-sidebar-background": "#0f172a",
        "color-sidebar-background-border": "#1e293b",
        "color-background-primary": "#0f172a",
        "color-background-secondary": "#1e293b",
    },
}

# Add source code link
html_show_sourcelink = True

# Custom CSS (if needed)
html_static_path = ["_static"] if os.path.exists("_static") else []

# Math support
mathjax3_config = {
    "tex": {
        "inlineMath": [["$", "$"], ["\\(", "\\)"]],
        "displayMath": [["$$", "$$"], ["\\[", "\\]"]],
    }
}
