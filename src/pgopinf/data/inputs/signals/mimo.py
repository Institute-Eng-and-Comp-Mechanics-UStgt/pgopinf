# yourpkg/inputs/signals/mimo.py
from __future__ import annotations
import numpy as np
from typing import Callable, Sequence


def stack_channels(channels: Sequence[Callable[[np.ndarray], np.ndarray]]):
    """Stack scalar input channels into a MIMO input callable.

    Parameters
    ----------
    channels : sequence of callable
        Channel callables. Each callable must return shape ``(1, n_t)`` or
        ``(n_t,)``.

    Returns
    -------
    callable
        Function ``u(t)`` returning shape ``(n_u, n_t)``.
    """

    def u(t):
        """Evaluate the stacked MIMO signal.

        Parameters
        ----------
        t : float or ndarray
            Evaluation time or times.

        Returns
        -------
        ndarray, shape (n_u, n_t)
            Stacked channel values.

        Raises
        ------
        ValueError
            If a channel returns more than one row.
        """
        t = np.atleast_1d(np.asarray(t, dtype=float))
        outs = []
        for ch in channels:
            y = ch(t)
            y = np.asarray(y, dtype=float)
            if y.ndim == 1:
                y = y[None, :]
            if y.shape[0] != 1:
                raise ValueError("Each channel must be SISO (shape (1,n_t) or (n_t,))")
            outs.append(y)
        return np.vstack(outs)

    return u
