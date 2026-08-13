import numpy as np
import scipy
import logging

from pgopinf.numerics.linalg.symmetry import sym


def check_spsd(A, rtol=1e-06, atol=1e-08):
    """
    Check if a matrix is symmetric positive definite (SPD).

    This function checks whether the input matrix `A` is symmetric and positive definite by
    attempting a Cholesky decomposition. If the matrix is symmetric but not positive definite,
    a small regularization term is added to try and make it positive definite.

    Parameters
    ----------
    A : array-like, shape (n, n)
        The matrix to be checked for symmetric positive definiteness.
    rtol : float, optional
        Relative tolerance for checking the symmetry of the matrix (default is 1e-06).
    atol : float, optional
        Absolute tolerance for checking the symmetry of the matrix (default is 1e-08).

    Returns
    -------
    bool
        True if the matrix is symmetric positive definite, False otherwise.

    Notes
    -----
    - If the matrix is not symmetric, the function immediately returns False.
    - If the Cholesky decomposition fails, a small regularization is applied before a second attempt.

    References
    ----------
    Adapted from [MorandinNicodemusUnger22].
    """
    if scipy.sparse.issparse(A):
        A = A.toarray()

    if np.allclose(A, A.T, rtol=rtol, atol=atol):
        try:
            np.linalg.cholesky(A)
            return True
        except np.linalg.LinAlgError:
            A = A + 1e-10 * np.eye(A.shape[0])
            try:
                np.linalg.cholesky(A)
                return True
            except np.linalg.LinAlgError:
                logging.info(f"Minimal eigenvalue is {np.min(np.linalg.eigvals(A))}")
                return False
    else:
        logging.info(f"The matrix is not symmetric, hence not spsd.")
        return False


def project_definite_cone(A, max_min_eig_val=0, definite_type="pos"):
    """
    Projects a matrix onto the set of symmetric positive or negative semi-definite matrices.

    The function computes the eigenvalue decomposition of the matrix `A`, adjusts the eigenvalues
    according to the specified `definite_type`, and reconstructs the matrix to ensure it lies within
    the desired definite cone (positive or negative semi-definite).

    Parameters
    ----------
    A : array_like, shape (n, n)
        The matrix to project onto the set of symmetric positive or negative semi-definite matrices.
    max_min_eig_val : float, optional
        The threshold for eigenvalue clipping. For positive semi-definite, eigenvalues are clipped
        from below at this value; for negative semi-definite, they are clipped from above (default is 0).
    definite_type : {"pos", "neg"}, optional
        Type of definiteness to project onto. "pos" for positive semi-definite,
        "neg" for negative semi-definite (default is "pos").

    Returns
    -------
    A_definite : numpy.ndarray, shape (n, n)
        The projected matrix which is symmetric and lies in the specified definite cone.
    """

    w, U = np.linalg.eigh(sym(A))
    if definite_type == "pos":
        if max_min_eig_val < 0:
            logging.warning(
                "Negative minimal eigenvalue given. The projected matrix will not be positive (semi)-definite"
            )
        A_definite = U @ np.diag(w.clip(min=max_min_eig_val)) @ U.T
    elif definite_type == "neg":
        if max_min_eig_val > 0:
            logging.warning(
                "Positive minimal eigenvalue given. The projected matrix will not be negative (semi)-definite"
            )
        A_definite = U @ np.diag(w.clip(max=max_min_eig_val)) @ U.T
    else:
        raise ValueError(f"Unknown definite_type {definite_type!r}.")
    return A_definite


def project_spsd(A):
    """
    Projects a matrix onto the set of symmetric positive semi-definite (SPSD) matrices.

    This function is a wrapper around `project_definite_cone`, specifically for projecting
    a matrix onto the SPSD cone.

    Parameters
    ----------
    A : array_like, shape (n, n)
        The matrix to be projected onto the SPSD cone.

    Returns
    -------
    A_spsd : numpy.ndarray, shape (n, n)
        The projected symmetric positive semi-definite matrix.
    """
    A_spsd = project_definite_cone(A, definite_type="pos")
    return A_spsd


def project_spd(A, min_eig_val=1e-10):
    """
    Projects a matrix onto the set of symmetric positive definite (SPD) matrices.

    This function projects a matrix onto the SPD cone by ensuring all eigenvalues
    are above a specified minimum eigenvalue.

    Parameters
    ----------
    A : array_like, shape (n, n)
        The matrix to be projected onto the SPD cone.
    min_eig_val : float, optional
        The minimum eigenvalue to enforce during projection (default is 1e-14).

    Returns
    -------
    A_spd : numpy.ndarray, shape (n, n)
        The projected symmetric positive definite matrix.
    """
    A_spd = project_definite_cone(A, min_eig_val, definite_type="pos")
    return A_spd


def project_snsd(A):
    """
    Projects a matrix onto the set of symmetric negative semi-definite (SNSD) matrices.

    This function is a wrapper around `project_definite_cone`, specifically for projecting
    a matrix onto the SNSD cone.

    Parameters
    ----------
    A : array_like, shape (n, n)
        The matrix to be projected onto the SNSD cone.

    Returns
    -------
    A_snsd : numpy.ndarray, shape (n, n)
        The projected symmetric negative semi-definite matrix.
    """
    A_snsd = project_definite_cone(A, definite_type="neg")
    return A_snsd


def project_snd(A, max_eig_val=-1e-14):
    """
    Projects a matrix onto the set of symmetric negative definite (SND) matrices.

    This function projects a matrix onto the SND cone by ensuring all eigenvalues
    are below a specified maximum eigenvalue.

    Parameters
    ----------
    A : array_like, shape (n, n)
        The matrix to be projected onto the SND cone.
    max_eig_val : float, optional
        The maximum eigenvalue to enforce during projection (default is -1e-14).

    Returns
    -------
    A_snd : numpy.ndarray, shape (n, n)
        The projected symmetric negative definite matrix.
    """
    A_snd = project_definite_cone(A, max_eig_val, definite_type="neg")
    return A_snd
