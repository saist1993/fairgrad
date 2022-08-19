# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import os
import sys
import sphinx_rtd_theme

html_theme = "sphinx_rtd_theme"

sys.path.insert(0, os.path.abspath("../../src/"))
html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]

project = "fairgrad"
copyright = "2022, Gaurav Maheshwari, Michael Perrot"
author = "Gaurav Maheshwari, Michael Perrot"
release = "0.1.1"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.coverage",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
]

templates_path = ["_templates"]
exclude_patterns = []

html_static_path = ["_static"]

intersphinx_mapping = {"python": ("http://docs.python.org/3", None)}

autodoc_typehints = "description"


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "alabaster"
html_static_path = ["_static"]
