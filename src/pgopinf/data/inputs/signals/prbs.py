# yourpkg/inputs/signals/prbs.py
from __future__ import annotations
import numpy as np
from scipy.signal import max_len_seq


def prbs_sequence(order: int, length: int) -> np.ndarray:
    """Generate a pseudo-random binary sequence.

    Parameters
    ----------
    order : int
        Linear-feedback shift-register order.
    length : int
        Number of samples.

    Returns
    -------
    ndarray, shape (length,)
        Sequence with values in ``{-1, 1}``.
    """
    prbs, _ = max_len_seq(order, length=length)
    return 2 * prbs - 1  # map {0,1}->{-1,1}


def prbs_time_signal(*, dt: float, order: int, length: int, amp: float = 1.0):
    """Create a piecewise-constant PRBS input signal.

    Parameters
    ----------
    dt : float
        Duration of each PRBS sample.
    order : int
        Linear-feedback shift-register order.
    length : int
        Number of PRBS samples.
    amp : float, optional
        Signal amplitude.

    Returns
    -------
    callable
        Function ``u(t)`` returning shape ``(1, n_t)``.
    """
    seq = amp * prbs_sequence(order=order, length=length)

    def u(t):
        """Evaluate the PRBS signal.

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
        idx = np.floor(t / dt).astype(int)
        idx = np.clip(idx, 0, len(seq) - 1)
        return seq[idx][None, :]  # (1, n_t)

    return u
