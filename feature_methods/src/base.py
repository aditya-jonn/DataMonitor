"""Compatibility shim: re-exports Model and FeatureSpace from models.base
so that `from .base import Model` (in src/__init__.py) resolves correctly.
"""
from .models.base import Model, FeatureSpace
__all__ = ["Model", "FeatureSpace"]
