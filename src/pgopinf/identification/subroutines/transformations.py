import numpy as np
import control as ct
import logging

from pymor.models.iosys import LTIModel as PymorLTIModel
from pgopinf.systems.ph_system import PHSystem
from pgopinf.numerics.linalg.passivity import kyp_lmi


def get_Riccati_transform(system):
    """
    Docstring for get_Riccati_transform

    :param system: Description
    """

    A, B, C, D, E = system.abcde

    try:
        X = solve_Riccati(system)
        XinvAT = np.linalg.solve(X, A.T)
        AXinv = XinvAT.T  # X is symmetric
        XinvCT = np.linalg.solve(X, C.T)

        Q = X
        J = 0.5 * (AXinv - XinvAT)
        R = -0.5 * (AXinv + XinvAT)
        G = 0.5 * (XinvCT + B)
        P = 0.5 * (XinvCT - B)
        S = 0.5 * (D + D.T)
        N = 0.5 * (D - D.T)
        # phlti_model, X = PHLTIModel.from_passive_LTIModel(lti_model)
        ph_system = PHSystem(J=J, R=R, Q=Q, G=G, P=P, S=S, N=N, E=E)
        logging.info(f"PH system successfully transformed!")
    except Exception as e:
        logging.error(f"PH model could not be calculated.\n Error: {e}")
        # return input system
        ph_system, X = system, None

    # get transformation matrix
    try:
        T = np.linalg.cholesky(X).conj().T  # transformation matrix
    except Exception as e:
        T = None
        logging.error(f"Transformation matrix could not be calculated.\n Error: {e}")

    return T, X, ph_system


def solve_Riccati(system):
    """Solve the positive-real Riccati equation used for pH conversion.

    Parameters
    ----------
    system
        LTI-like system exposing ``abcde`` matrices.

    Returns
    -------
    ndarray or None
        Riccati solution ``X`` if computation and KYP validation succeed;
        otherwise ``None``.
    """
    A, B, C, D, E = system.abcde
    # add regularization to D to be nonsingular
    D = D + np.eye(D.shape[0]) * 1e-6
    pymor_lti_model = PymorLTIModel.from_matrices(A, B, C, D, E)

    # %% Check for positive-real systems at certain frequency points
    calT = lambda s: D + C @ np.linalg.inv(s * np.eye(A.shape[0]) - A) @ B
    Phi = lambda s: (calT(-s)).conj().T + calT(s)
    # Transfer function Phi needs to be positive semidefinite for all omega in R (checked for several points)
    omega_vec = np.linspace(0.1, 1000, 100)
    negEigValPhi = False
    omega_neg_eig_list = []
    for omega in omega_vec:
        eig_vals_Phi, _ = np.linalg.eig(Phi(1j * omega))
        if (eig_vals_Phi < 0).any():
            omega_neg_eig_list.append(omega)
            # print(f"negative eigenvalues at i{omega} rad/s \n")
            negEigValPhi = True
    if not negEigValPhi:
        logging.info(
            f"Phi is positive semidefinite - X that satisfies W(X) should exist"
        )
    else:
        logging.info(
            f"There are negative eigenvalues at {len(omega_neg_eig_list)} frequency points out of {len(omega_vec)} tested points."
            f"These points are {omega_neg_eig_list}."
        )
        logging.warning(
            f"Phi is NOT positive semidefinite - X that satisfies W(X) should NOT exist"
        )

    if E is not None:
        if not (np.allclose(np.eye(E.shape[0]), E)):
            # E not identity
            logging.info(
                f"Transform E to identity to check passivity with control systems library."
            )
            # transform ss system for passivity check
            A_invE = np.linalg.solve(E, A)
            B_invE = np.linalg.solve(E, B)
            ct_lti_system = ct.StateSpace(A_invE, B_invE, C, D)
        else:
            # E identity
            ct_lti_system = ct.StateSpace(A, B, C, D)
    else:
        # E None
        ct_lti_system = ct.StateSpace(A, B, C, D)

    if not A.shape[0] > 30:
        # check passivity (only for small n)
        is_passive = ct_lti_system.ispassive()
        logging.info(f"System is passive: {is_passive}.")
        if is_passive:
            logging.info(f"System is passive.")
        else:
            logging.info("System might be passive. Only sufficient condition tested.")

    # %% get ph model
    try:
        # %% transform into pH system
        logging.info(
            f"Trying to transform state-space system into pH system through Riccati solution..."
        )
        method = "pymor"  # "pymor" | "control_riccati"
        if method == "pymor":
            # copied from  pymor.PHLTIModel.from_passive_LTIModel() in order to return X
            X = pymor_lti_model.gramian("pr_o_dense")
        elif method == "control_LMI":
            #
            raise ValueError("control_LMI method temporarily disabled (not working).")
            X = ct.passivity.solve_passivity_LMI(ct_lti_system, rho=1e-12)
        elif method == "control_riccati":
            logging.warning(f"TODO: Check if matrices are correct.")
            method_ct = "slycot"  # "slycot"| "scipy"
            n = A.shape[0]
            Q = np.zeros((n, n))  # for strict PR KYP Riccati
            R = -(D + D.T)  # must be symmetric positive definite
            S = -C.T
            X, _, _ = ct.care(A=A, B=B, Q=Q, R=R, S=S, E=E, method=method_ct)
        else:
            raise ValueError(f"Unknown method {method} to compute Riccati solution.")

        residual_norm = residual_Riccati(A, B, C, D, E, X)
        logging.info(f"Riccati residual norm: {residual_norm:e}")

        # check observability of (C, A)
        check_observability(A, C)
        # check controllability of (A, B)
        check_controllability(A, B)

        _, is_spsd = kyp_lmi(
            A=A,
            B=B,
            C=C,
            D=D,
            X=X,
            E=E,
            relaxed=False,
        )

        if not is_spsd:
            logging.error(
                f"The solution X of the Riccati equation does not satisfy the KYP-LMI - system is not pH."
            )
            raise ValueError(
                f"The solution X of the Riccati equation does not satisfy the KYP-LMI - system is not pH."
            )
    except Exception as e:
        X = None
        logging.error(f"Error: {e}")

    return X


def check_controllability(A, B):
    """Print a PBH controllability check for a matrix pair.

    Parameters
    ----------
    A : ndarray, shape (n, n)
        State matrix.
    B : ndarray, shape (n, n_u)
        Input matrix.
    """
    n = A.shape[0]

    # 1. Calculate Eigenvalues of A
    evals = np.linalg.eigvals(A)

    # 2. Check PBH rank for each eigenvalue
    is_controllable = True
    for lam in evals:
        # Form the PBH matrix: [ A - lambda*I, B ]
        pbh_matrix = np.hstack((A - lam * np.eye(n), B))

        # Check singular values (better than rank)
        s = np.linalg.svdvals(pbh_matrix)

        # If the smallest singular value is essentially zero, it's uncontrollable
        if np.min(s) < 1e-9:
            print(f"Uncontrollable mode at lambda = {lam}")
            is_controllable = False
            break

    if is_controllable:
        print("System is controllable (verified via PBH test).")


def check_observability(A, C):
    """Print a PBH observability check for a matrix pair.

    Parameters
    ----------
    A : ndarray, shape (n, n)
        State matrix.
    C : ndarray, shape (n_y, n)
        Output matrix.
    """
    # A, C are your system matrices
    n = A.shape[0]

    # 1. Calculate Eigenvalues of A
    evals = np.linalg.eigvals(A)

    # 2. Check PBH rank for each eigenvalue
    is_observable = True
    for lam in evals:
        # Form the PBH matrix: [ (A - lambda*I).T, C.T ].T
        # The matrix should have n columns (states) and (n + num_outputs) rows
        pbh_matrix = np.vstack((A - lam * np.eye(n), C))

        # Check singular values (better than rank)
        s = np.linalg.svdvals(pbh_matrix)

        # If the smallest singular value is essentially zero, it's unobservable
        if np.min(s) < 1e-9:
            print(f"Unobservable mode at lambda = {lam}")
            is_observable = False
            break

    if is_observable:
        print("System is observable (verified via PBH test).")


def residual_Riccati(A, B, C, D, E, X):
    """Compute the residual of the KYP Riccati equation for passivity.

    Parameters
    ----------
    A : np.ndarray
        State matrix.
    B : np.ndarray
        Input matrix.
    C : np.ndarray
        Output matrix.
    D : np.ndarray
        Feedthrough matrix.
    E : np.ndarray or None
        Descriptor matrix (None if standard state-space).
    X : np.ndarray
        Candidate solution to the Riccati equation.
    Returns
    -------
    residual_norm : float
        The Frobenius norm of the Riccati residual.
    """
    R = D + D.T  # must be symmetric positive definite
    Q = np.zeros_like(A)
    residual_norm = np.linalg.norm(
        -A.T @ X @ E
        - E.T @ X @ A
        - (C.T - E.T @ X @ B) @ np.linalg.inv(R) @ (C - B.T @ X @ E)
        + Q
    )
    return residual_norm
