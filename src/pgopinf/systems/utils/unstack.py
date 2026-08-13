import numpy as np


def unstack(AA, n, no_feedtrough=False):
    """
    Unstack a matrix into blocks.

    Parameters
    ----------
    AA : numpy.ndarray
        Matrix to unstack.
    n : int
        Number of rows/columns in the first block.
    no_feedtrough
        If `True`, the feedtrough block is set to zero.

    Returns
    -------
    A : numpy.ndarray
        First block.
    B : numpy.ndarray
        Second block.
    C : numpy.ndarray
        Third block.
    D : numpy.ndarray
        Fourth block.
    """
    A = AA[:n, :n]
    B = AA[:n, n:]
    C = AA[n:, :n]
    D = AA[n:, n:]

    if no_feedtrough:
        D = np.zeros_like(D)

    return A, B, C, D
