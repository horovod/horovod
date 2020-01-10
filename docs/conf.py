# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# http://www.sphinx-doc.org/en/master/config

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.abspath('..'))


# -- Project information -----------------------------------------------------

project = 'Horovod'
copyright = '2019, The Horovod Authors'
author = 'The Horovod Authors'

from horovod import __version__
version = __version__


# -- Mocking configuration ---------------------------------------------------

import mocks
mocks.instrument()


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinxcontrib.napoleon',
    'nbsphinx',
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# The master toctree document.
master_doc = 'index'

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Autodoc configuration ---------------------------------------------------

autodoc_default_options = {
    'members': None,
    'member-order': 'bysource',
    'imported-members': None,
    'exclude-members': 'contextmanager, LooseVersion, tf, keras, torch, mx, pyspark',
}


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'alabaster'

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
#
# For alabaster: https://alabaster.readthedocs.io/en/latest/customization.html
#
html_theme_options = {
    'logo': 'logo.png',
    'description': 'Distributed training framework for TensorFlow, Keras, PyTorch, and Apache MXNet.',
    'github_user': 'horovod',
    'github_repo': 'horovod',
    'github_button': True,
    'github_type': 'star',
    'github_count': 'true',
    'fixed_sidebar': False,
    'sidebar_collapse': True,
    'font_family': 'Helvetica Neue'
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
