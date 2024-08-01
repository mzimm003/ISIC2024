# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'ISIC 2024'
copyright = '2024, Mark Zimmerman'
author = 'Mark Zimmerman'
release = '0.0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = []

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
html_static_path = ['_static']
html_theme_options = {
    "nosidebar": "false",
    "description": "Identify histologically confirmed skin cancer cases with single-lesion crops from 3D total body photos.",
    "github_button":True,
    'github_user': 'mzimm003',
    'github_repo': 'https://github.com/mzimm003/ISIC2024',
    "fixed_sidebar":True,
    "sidebar_width":"15%",
    "page_width":"90%"
}
html_css_files = [
    'custom.css',
]