"""Compatibility shim: re-exports ConvAutoEncoder + feature-space wrapper
from the actual upstream module Unsuper_conv_autoencoder, and adds the
_key() static method that src/__init__.py's load_model() dispatches on.
"""
from .Unsuper_conv_autoencoder import (
    ConvAutoEncoder as _UpstreamConvAutoEncoder,
    ConvAutoEncoderFeatureSpace,
)


class ConvAutoEncoder(_UpstreamConvAutoEncoder):
    """Thin wrapper that adds the _key() method src/__init__.py expects.
    The upstream class doesn't define it, so load_model()'s dispatch fails.
    """
    @staticmethod
    def _key():
        return "conv-autoencoder"


__all__ = ["ConvAutoEncoder", "ConvAutoEncoderFeatureSpace"]
