"""Compatibility shim: re-exports from feature_methods.src.supcon_loss
so that `from feature_methods.supcon_loss import TwoCropTransform`
(used by datasets/__init__.py) resolves correctly.
"""
from .src.supcon_loss import *  # noqa: F401,F403
from .src.supcon_loss import TwoCropTransform  # explicit re-export
__all__ = ["TwoCropTransform"]
