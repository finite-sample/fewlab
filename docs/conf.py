"""Configuration file for the Sphinx documentation builder."""

import sys
import tomllib
from pathlib import Path

# Add the parent directory to the path so we can import fewlab
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Read project metadata from pyproject.toml
pyproject_path = project_root / "pyproject.toml"
with pyproject_path.open("rb") as f:
    pyproject_data = tomllib.load(f)

project_info = pyproject_data["project"]
project = project_info["name"]
release = project_info["version"]
author = project_info["authors"][0]["name"]
copyright = f"2024, {author}"

# Extensions
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.autosummary",
    "sphinx.ext.mathjax",
    "sphinx_design",
    "nbsphinx",
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
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "**.ipynb_checkpoints"]

# nbsphinx configuration
nbsphinx_execute = "always"  # Force execution of notebooks
nbsphinx_allow_errors = True  # Allow errors during execution for debugging
nbsphinx_kernel_name = "python3"
nbsphinx_timeout = 600  # 10 minute timeout for notebook execution
nbsphinx_execute_arguments = [
    "--InlineBackend.figure_formats={'svg', 'pdf'}",
    "--InlineBackend.rc={'figure.dpi': 96}"
]

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
    "footer_icons": [
        {
            "name": "GitHub",
            "url": "https://github.com/finite-sample/fewlab",
            "html": """
                <svg stroke="currentColor" fill="currentColor" stroke-width="0" viewBox="0 0 16 16">
                    <path fill-rule="evenodd" d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z"/>
                </svg>
            """,
        }
    ],
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
static_dir = Path(__file__).parent / "_static"
html_static_path = ["_static"] if static_dir.exists() else []

# Math support
mathjax3_config = {
    "tex": {
        "inlineMath": [["$", "$"], ["\\(", "\\)"]],
        "displayMath": [["$$", "$$"], ["\\[", "\\]"]],
    }
}
