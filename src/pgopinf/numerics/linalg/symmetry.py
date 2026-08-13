import numpy as np
import scipy


def is_sym(A, rtol=1.0e-5, atol=1.0e-8):
    """Return whether a matrix is symmetric within tolerance.

    Parameters
    ----------
    A : array_like or sparse matrix
        Matrix to check.
    rtol : float, optional
        Relative tolerance.
    atol : float, optional
        Absolute tolerance.

    Returns
    -------
    bool
        ``True`` if ``A`` is close to ``A.T``.
    """
    A = check_sparsity_and_convert_to_dense(A)
    return np.allclose(A, A.T, rtol=rtol, atol=atol)


def is_skewsym(A, rtol=1.0e-5, atol=1.0e-8):
    """Return whether a matrix is skew-symmetric within tolerance.

    Parameters
    ----------
    A : array_like or sparse matrix
        Matrix to check.
    rtol : float, optional
        Relative tolerance.
    atol : float, optional
        Absolute tolerance.

    Returns
    -------
    bool
        ``True`` if ``A`` is close to ``-A.T``.
    """
    A = check_sparsity_and_convert_to_dense(A)
    return np.allclose(A, -A.T, rtol=rtol, atol=atol)


def sym(A):
    """Return the symmetric part of a matrix.

    Parameters
    ----------
    A : array_like
        Input matrix.

    Returns
    -------
    ndarray
        ``0.5 * (A + A.T)``.
    """
    return 0.5 * (A + A.T)


def skewsym(A):
    """Return the skew-symmetric part of a matrix.

    Parameters
    ----------
    A : array_like
        Input matrix.

    Returns
    -------
    ndarray
        ``0.5 * (A - A.T)``.
    """
    return 0.5 * (A - A.T)


def check_sparsity_and_convert_to_dense(A):
    """Convert sparse matrices to dense arrays.

    Parameters
    ----------
    A : array_like or sparse matrix
        Matrix to normalize.

    Returns
    -------
    array_like
        Dense array if ``A`` is sparse; otherwise ``A`` unchanged.
    """
    if scipy.sparse.issparse(A):
        A = A.toarray()
    return A


def hermitian_part(X):
    """Computes (X + X.H) / 2"""
    return 0.5 * (X + X.conj().T)


def skew_hermitian(X):
    """Computes (X - X.H) / 2"""
    return 0.5 * (X - X.conj().T)
