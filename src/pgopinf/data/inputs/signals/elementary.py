# yourpkg/inputs/signals/elementary.py
from __future__ import annotations
import numpy as np
from scipy.signal import sawtooth


def constant(value: float, n_u: int = 1):
    """Create a constant input signal.

    Parameters
    ----------
    value : float
        Constant value.
    n_u : int, optional
        Number of identical input channels.

    Returns
    -------
    callable
        Function ``u(t)`` returning shape ``(n_u, n_t)``.
    """
    def u(t):
        """Evaluate the constant signal."""
        t = np.asarray(t)
        return np.full((n_u, t.size), value, dtype=float)

    return u


def sine(freq_hz: float, amp: float = 1.0, phase: float = 0.0, n_u: int = 1):
    """Create a sinusoidal input signal.

    Parameters
    ----------
    freq_hz : float
        Frequency in hertz.
    amp : float, optional
        Amplitude.
    phase : float, optional
        Phase offset in radians.
    n_u : int, optional
        Number of identical input channels.

    Returns
    -------
    callable
        Function ``u(t)`` returning shape ``(n_u, n_t)``.
    """
    w = 2.0 * np.pi * freq_hz

    def u(t):
        """Evaluate the sinusoidal signal."""
        t = np.asarray(t)
        sig = amp * np.sin(w * t + phase)
        return (
            np.tile(sig[None, :], (n_u, 1)) if sig.ndim == 1 else np.tile(sig, (n_u, 1))
        )

    return u


def cosine(freq_hz: float, amp: float = 1.0, phase: float = 0.0, n_u: int = 1):
    """Create a cosine input signal.

    Parameters
    ----------
    freq_hz : float
        Frequency in hertz.
    amp : float, optional
        Amplitude.
    phase : float, optional
        Phase offset in radians.
    n_u : int, optional
        Number of identical input channels.

    Returns
    -------
    callable
        Function ``u(t)`` returning shape ``(n_u, n_t)``.
    """
    w = 2.0 * np.pi * freq_hz

    def u(t):
        """Evaluate the cosine signal."""
        t = np.atleast_1d(np.asarray(t, dtype=float))
        sig = amp * np.cos(w * t + phase)
        return (
            np.tile(sig[None, :], (n_u, 1)) if sig.ndim == 1 else np.tile(sig, (n_u, 1))
        )

    return u


def sawtooth_wave(freq_hz: float, amp: float = 1.0, n_u: int = 1):
    """Create a sawtooth input signal.

    Parameters
    ----------
    freq_hz : float
        Frequency in hertz.
    amp : float, optional
        Amplitude.
    n_u : int, optional
        Number of identical input channels.

    Returns
    -------
    callable
        Function ``u(t)`` returning shape ``(n_u, n_t)``.
    """
    w = 2.0 * np.pi * freq_hz

    def u(t):
        """Evaluate the sawtooth signal."""
        t = np.atleast_1d(np.asarray(t, dtype=float))
        sig = amp * sawtooth(w * t)
        return (
            np.tile(sig[None, :], (n_u, 1)) if sig.ndim == 1 else np.tile(sig, (n_u, 1))
        )

    return u


def damped_sin(
    delta: float, amp: float, freq_hz: float, chirp: bool = True, n_u: int = 1
):
    """Create a damped sinusoidal input signal.

    Parameters
    ----------
    delta : float
        Exponential decay rate.
    amp : float
        Amplitude.
    freq_hz : float
        Base frequency in hertz.
    chirp : bool, optional
        If ``True``, use a quadratic phase ``w * t**2``.
    n_u : int, optional
        Number of identical input channels.

    Returns
    -------
    callable
        Function ``u(t)`` returning shape ``(n_u, n_t)``.
    """
    w = 2.0 * np.pi * freq_hz

    def u(t):
        """Evaluate the damped sinusoidal signal."""
        t = np.atleast_1d(np.asarray(t, dtype=float))
        phase = w * t**2 if chirp else w * t
        sig = amp * np.exp(-delta * t) * np.sin(phase)
        return (
            np.tile(sig[None, :], (n_u, 1)) if sig.ndim == 1 else np.tile(sig, (n_u, 1))
        )

    return u


def damped_cos(
    delta: float, amp: float, freq_hz: float, chirp: bool = True, n_u: int = 1
):
    """Create a damped cosine input signal.

    Parameters
    ----------
    delta : float
        Exponential decay rate.
    amp : float
        Amplitude.
    freq_hz : float
        Base frequency in hertz.
    chirp : bool, optional
        If ``True``, use a quadratic phase ``w * t**2``.
    n_u : int, optional
        Number of identical input channels.

    Returns
    -------
    callable
        Function ``u(t)`` returning shape ``(n_u, n_t)``.
    """
    w = 2.0 * np.pi * freq_hz

    def u(t):
        """Evaluate the damped cosine signal."""
        t = np.atleast_1d(np.asarray(t, dtype=float))
        phase = w * t**2 if chirp else w * t
        sig = amp * np.exp(-delta * t) * np.cos(phase)
        return (
            np.tile(sig[None, :], (n_u, 1)) if sig.ndim == 1 else np.tile(sig, (n_u, 1))
        )

    return u
