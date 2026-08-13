import numpy as np


import numpy as np
from scipy import linalg

from pgopinf.numerics.linalg.symmetry import (
    skew_hermitian,
    hermitian_part,
)


def truncation(d, L, trunc_tol=1e-12):
    """
    Computes a rank revealing factorization for a given LDL-decomposition.
    """
    Q, R = linalg.qr(L, mode="economic")
    # tmp = R * diag(d) * R.T
    tmp = R @ np.diag(d) @ R.conj().T
    tmp = hermitian_part(tmp)  # Ensure symmetry for eigen

    d_vals, U = linalg.eigh(tmp)

    # Sort by absolute value descending
    p = np.argsort(np.abs(d_vals))[::-1]
    d_vals = d_vals[p]
    U = U[:, p]

    # Find truncation index
    relative_vals = np.abs(d_vals / d_vals[0])
    trunc_idx = np.where(relative_vals >= trunc_tol)[0][-1] + 1

    return d_vals[:trunc_idx], Q @ U[:, :trunc_idx]


def lrcholesky(X, trunc_tol=1e-12):
    """
    Low-rank approximate Cholesky-like factorization X = L * L.H
    """
    X_h = hermitian_part(X)
    d, L = linalg.eigh(X_h)

    # remove negative eigenvalues (numerical errors)
    idx = d >= 0
    d_pos = d[idx]
    L_pos = L[:, idx]

    dr, Lr = truncation(d_pos, L_pos, trunc_tol=trunc_tol)
    return Lr @ np.diag(np.sqrt(dr))
