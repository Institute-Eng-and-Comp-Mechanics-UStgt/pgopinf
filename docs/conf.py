from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

project = "pgopinf"
author = "Johannes Rettberg, Jonas Nicodemus"
copyright = "2026, Johannes Rettberg, Jonas Nicodemus"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "myst_parser",
    "sphinx_autodoc_typehints",
]

autosummary_generate = True
autosummary_generate_overwrite = True
napoleon_google_docstring = False
napoleon_numpy_docstring = True

autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "show-inheritance": True,
    "undoc-members": True,
}
autodoc_typehints = "description"

autodoc_mock_imports = [
    "control",
    "cvxpy",
    "matplotlib",
    "mosek",
    "pymor",
    "slycot",
]

html_theme = "pydata_sphinx_theme"
html_title = "PetrovGalerkinOperatorInference"
