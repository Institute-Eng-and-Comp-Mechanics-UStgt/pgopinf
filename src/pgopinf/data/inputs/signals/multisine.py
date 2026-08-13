# yourpkg/inputs/signals/multisine.py
from __future__ import annotations
import numpy as np


def multisine(
    *,
    f_low: float,
    f_high: float,
    n_freqs: int,
    amplitudes: str | np.ndarray | None = None,
    overall_scale: float = 1.0,
    random_phase: bool = True,
    phase_seed: int = 1,
):
    """Create a scalar multisine input signal.

    Parameters
    ----------
    f_low : float
        Lowest frequency in hertz.
    f_high : float
        Highest frequency in hertz.
    n_freqs : int
        Number of logarithmically spaced frequencies.
    amplitudes : {"envelope"} or ndarray, optional
        Per-frequency amplitudes. The special value ``"envelope"`` uses
        ``1 / sqrt(freq)``.
    overall_scale : float, optional
        Multiplicative scale applied to the summed signal.
    random_phase : bool, optional
        If ``True``, draw random phases.
    phase_seed : int, optional
        Random seed used for phases.

    Returns
    -------
    callable
        Function ``u(t)`` returning shape ``(1, n_t)``.

    Raises
    ------
    ValueError
        If frequencies or amplitudes are invalid.
    """
    if f_low <= 0 or f_high <= f_low:
        raise ValueError("Require 0 < f_low < f_high.")

    freqs_hz = np.logspace(np.log10(f_low), np.log10(f_high), n_freqs)
    w = 2.0 * np.pi * freqs_hz

    if amplitudes is None:
        amps = np.ones(n_freqs)
    elif isinstance(amplitudes, str) and amplitudes == "envelope":
        amps = 1.0 / np.sqrt(freqs_hz)
    else:
        amps = np.asarray(amplitudes, dtype=float)
        if amps.shape != (n_freqs,):
            raise ValueError("amplitudes must have shape (n_freqs,)")

    if random_phase:
        rng = np.random.default_rng(phase_seed)
        phases = rng.uniform(0, 2 * np.pi, size=n_freqs)
    else:
        phases = np.zeros(n_freqs)

    def u(t):
        """Evaluate the multisine signal.

        Parameters
        ----------
        t : float or ndarray
            Evaluation time or times.

        Returns
        -------
        ndarray, shape (1, n_t)
            Signal values.
        """
        t = np.atleast_1d(np.asarray(t, dtype=float))
        # sum_k amps_k sin(w_k t + phase_k)
        sig = np.sum(
            amps[:, None] * np.sin(w[:, None] * t[None, :] + phases[:, None]), axis=0
        )
        return (overall_scale * sig)[None, :]  # (1, n_t)

    return u
