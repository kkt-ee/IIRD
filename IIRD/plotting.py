""" Plotting utilities for IIRD.
Provides a robust frequency response plotting function for IIR filters.

IIRD: Fast trainable D-dimensional IIR filter layers in TensorFlow.
Copyright 2025 Kishore Kumar Tarafdar.
Licensed under the Apache License, Version 2.0. See LICENSE for details.
"""
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import freqz
from typing import Iterable, Tuple, Optional

__all__ = ["plot_frequency_response"]

def plot_frequency_response(
    b: Iterable[float],
    a: Iterable[float],
    fs: Optional[float] = None,
    whole: bool = False,
    worN: int = 1024,
    dB_floor: float = -300.0,
    unwrap_phase: bool = True,
    ax: Optional[Tuple[plt.Axes, plt.Axes]] = None,
    show: bool = True,
):
    """Plot magnitude (dB) and phase of an IIR filter.

    Parameters
    ----------
    b : iterable of float
        Numerator coefficients.
    a : iterable of float
        Denominator coefficients. If a[0] != 1 it is normalized.
    fs : float | None, optional
        Sampling frequency in Hz. If None, frequencies are in rad/sample.
    whole : bool, default False
        If False plot 0..π (or 0..fs/2). If True plot -π..π (or -fs/2..fs/2).
    worN : int, default 1024
        Number of frequency samples.
    dB_floor : float, default -300
        Minimum magnitude (in dB) to avoid -inf.
    unwrap_phase : bool, default True
        Apply phase unwrapping.
    ax : tuple(axes, axes) | None
        (ax_mag, ax_phase). If None new figure is created.
    show : bool, default True
        Whether to call plt.show().

    Returns
    -------
    w_plot : ndarray
        Frequencies (rad/sample if fs is None else Hz).
    h : ndarray
        Complex frequency response.
    (ax_mag, ax_phase) : tuple of Axes
        The matplotlib axes used.
    """
    b = np.asarray(b, dtype=float)
    a = np.asarray(a, dtype=float)

    if a.size == 0:
        raise ValueError("Denominator 'a' must not be empty")
    if a[0] == 0:
        raise ValueError("a[0] must be non-zero")
    if a[0] != 1.0:
        b = b / a[0]
        a = a / a[0]

    if fs is None:
        if whole:
            w_raw, h = freqz(b, a, worN=worN, whole=True)  # 0..2π
            h = np.fft.fftshift(h)
            w_plot = np.linspace(-np.pi, np.pi, worN, endpoint=False)
        else:
            w_plot, h = freqz(b, a, worN=worN)  # 0..π
    else:
        if whole:
            w_raw, h = freqz(b, a, worN=worN, whole=True, fs=fs)  # 0..fs
            h = np.fft.fftshift(h)
            w_plot = np.linspace(-fs/2, fs/2, worN, endpoint=False)
        else:
            w_plot, h = freqz(b, a, worN=worN, fs=fs)  # 0..fs/2

    mag_db = 20 * np.log10(np.maximum(np.abs(h), 10 ** (dB_floor / 20.0)))
    phase = np.unwrap(np.angle(h)) if unwrap_phase else np.angle(h)

    created = False
    if ax is None:
        fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        ax_mag, ax_phase = axes
        created = True
    else:
        ax_mag, ax_phase = ax

    ax_mag.plot(w_plot, mag_db, 'b')
    ax_mag.set_ylabel('Magnitude (dB)')
    ax_mag.set_title('Frequency Response')
    ax_mag.grid(True, which='both', ls=':')

    ax_phase.plot(w_plot, phase, 'r')
    ax_phase.set_ylabel('Phase (radians)')
    ax_phase.grid(True, which='both', ls=':')

    if fs is None:
        if whole:
            ticks = [-np.pi, -np.pi/2, 0, np.pi/2, np.pi]
            labels = ['-π', '-π/2', '0', 'π/2', 'π']
            ax_mag.set_xlim(-np.pi, np.pi)
        else:
            ticks = [0, np.pi/2, np.pi]
            labels = ['0', 'π/2', 'π']
            ax_mag.set_xlim(0, np.pi)
        ax_phase.set_xlabel('Normalized Frequency (rad/sample)')
        ax_phase.set_xticks(ticks, labels)
        ax_mag.set_xticks(ticks, labels)
    else:
        if whole:
            half = fs / 2
            ticks = [-half, -half/2, 0, half/2, half]
            labels = [f'{-half:g}', f'{-half/2:g}', '0', f'{half/2:g}', f'{half:g}']
            ax_mag.set_xlim(-half, half)
            ax_phase.set_xticks(ticks, labels)
            ax_mag.set_xticks(ticks, labels)
        else:
            nyq = fs / 2
            ax_mag.set_xlim(0, nyq)
        ax_phase.set_xlabel('Frequency (Hz)')

    if created:
        plt.tight_layout()
        if show:
            plt.show()

    return w_plot, h, (ax_mag, ax_phase)
