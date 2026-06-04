"""Top-level package shim: re-exports from feature_methods.src so that
`from feature_methods import load_model` (and similar) resolve correctly.
"""
from .src import *  # noqa: F401,F403
# Explicit re-export of load_model, since `*` only re-exports names in __all__
from .src import load_model  # noqa: F401
