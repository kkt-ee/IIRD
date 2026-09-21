"""
IIRD: Fast trainable D-dimensional IIR filter layers in TensorFlow.
Copyright 2025 Kishore Kumar Tarafdar.
Licensed under the Apache License, Version 2.0. See LICENSE for details.
"""



# import tensorflow as tf

__version__="0.0.1"

from .IIR1D import IIR1D  # noqa: E402,F401
from .IIR2D import IIR2D  # noqa: E402,F401
from .IIR3D import IIR3D  # noqa: E402,F401
from .constraints import Positive  # noqa: E402,F401


def plot_frequency_response(*args, **kwargs):
    from .plotting import plot_frequency_response as _plot_frequency_response

    return _plot_frequency_response(*args, **kwargs)

__all__ = [
    "IIR1D",
    "IIR2D",
    "IIR3D",
    "Positive",
    "plot_frequency_response",
]
