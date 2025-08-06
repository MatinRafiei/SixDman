import os
import sys
sys.path.insert(0, os.path.abspath('../src'))

import sixdman
print("DEBUG: Imported sixdman version:", getattr(sixdman, "__version__", "unknown"))


# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'SixDman'
copyright = '2025, Matin Rafiei Forooshani'
author = 'Matin Rafiei Forooshani'
release = '0.1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

templates_path = ['_templates']
exclude_patterns = []

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',       # Google/NumPy style docstrings
    'sphinx_autodoc_typehints',  # Show type hints
]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'press'

# html_permalinks_icon = '§'
# html_theme = 'insipid'


autodoc_member_order = "bysource"
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
}