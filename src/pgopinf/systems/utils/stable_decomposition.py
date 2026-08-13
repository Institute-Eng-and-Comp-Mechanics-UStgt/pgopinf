import numpy as np
from scipy.linalg import ordqz
from scipy.linalg import schur, rsf2csf, eigvals


def stable_decomposition(A, E, B, C, D, discrete=False, tol=1e-10):
    """
    Extract the stable part of a descriptor LTI system

        E xdot = A x + B u
            y  = C x + D u

    or, in discrete time,

        E x[k+1] = A x[k] + B u[k]

    Parameters
    ----------
    A, E, B, C, D : ndarray
        Descriptor state-space matrices.
    discrete : bool, optional
        False for continuous-time, True for discrete-time.
    tol : float, optional
        Numerical tolerance.

    Returns
    -------
    As, Es, Bs, Cs, Ds : ndarray
        Stable descriptor subsystem.
    stable_eigs : ndarray
        Stable generalized eigenvalues.
    unstable_eigs : ndarray
        Unstable generalized eigenvalues.

    Notes
    -----
    Continuous-time stability:
        Re(lambda) < 0

    Discrete-time stability:
        |lambda| < 1

    Infinite generalized eigenvalues correspond to beta = 0.
    They are not asymptotically stable and are excluded here.
    """

    A = np.asarray(A)
    E = np.asarray(E)
    B = np.asarray(B)
    C = np.asarray(C)
    D = np.asarray(D)

    n = A.shape[0]
    if A.shape != E.shape or A.shape[0] != A.shape[1]:
        raise ValueError("A and E must be square and of the same shape.")
    if B.shape[0] != n:
        raise ValueError("B has incompatible shape.")
    if C.shape[1] != n:
        raise ValueError("C has incompatible shape.")

    def select_function(alpha, beta):
        """
        Return True for stable generalized eigenvalues alpha/beta.
        beta ~ 0 => infinite eigenvalue => discard.
        """
        beta_abs = np.abs(beta)
        finite = beta_abs > tol

        lam = np.empty_like(alpha, dtype=np.complex128)
        lam[:] = np.inf
        lam[finite] = alpha[finite] / beta[finite]

        if discrete:
            stable = finite & (np.abs(lam) < 1.0 - tol)
        else:
            stable = finite & (np.real(lam) < -tol)

        return stable

    # Reorder QZ so stable generalized eigenvalues come first
    S, T, alpha, beta, Q, Z = ordqz(A, E, sort=select_function, output="complex")

    # Generalized eigenvalues lambda = alpha / beta
    finite = np.abs(beta) > tol
    eigs = np.full(alpha.shape, np.inf + 0j, dtype=np.complex128)
    eigs[finite] = alpha[finite] / beta[finite]

    if discrete:
        stable_mask = finite & (np.abs(eigs) < 1.0 - tol)
    else:
        stable_mask = finite & (np.real(eigs) < -tol)

    ns = int(np.sum(stable_mask))

    stable_eigs = eigs[stable_mask]
    unstable_eigs = eigs[~stable_mask]

    if ns == 0:
        As = np.zeros((0, 0), dtype=A.dtype)
        Es = np.zeros((0, 0), dtype=E.dtype)
        Bs = np.zeros((0, B.shape[1]), dtype=B.dtype)
        Cs = np.zeros((C.shape[0], 0), dtype=C.dtype)
        Ds = D.copy()
        return As, Es, Bs, Cs, Ds, stable_eigs, unstable_eigs

    # Stable right deflating subspace
    Zs = Z[:, :ns]

    # Projection onto stable subspace
    As = Zs.conj().T @ A @ Zs
    Es = Zs.conj().T @ E @ Zs
    Bs = Zs.conj().T @ B
    Cs = C @ Zs
    Ds = D.copy()

    tol_to_real = 1000
    As = np.real_if_close(As, tol=tol_to_real)
    Es = np.real_if_close(Es, tol=tol_to_real)
    Bs = np.real_if_close(Bs, tol=tol_to_real)
    Cs = np.real_if_close(Cs, tol=tol_to_real)
    Ds = np.real_if_close(Ds, tol=tol_to_real)

    return As, Es, Bs, Cs, Ds, stable_eigs, unstable_eigs


def stable_part_lti(A, B, C, D, discrete=False, tol=1e-9):
    """
    Extract the stable part of an LTI state-space system.

    Parameters
    ----------
    system : scipy.signal.StateSpace
        State-space system with matrices (A, B, C, D).
    discrete : bool, optional
        False for continuous-time systems.
        True for discrete-time systems.
    tol : float, optional
        Numerical tolerance for stability test.

    Returns
    -------
    stable_sys : scipy.signal.StateSpace
        Reduced system containing only stable modes.
    stable_eigs : ndarray
        Stable eigenvalues kept in the reduced system.
    unstable_eigs : ndarray
        Unstable eigenvalues removed from the system.

    Notes
    -----
    Continuous-time stability: Re(lambda) < 0
    Discrete-time stability:   |lambda| < 1
    """

    A = np.asarray(A)
    B = np.asarray(B)
    C = np.asarray(C)
    D = np.asarray(D)

    n = A.shape[0]
    if n == 0:
        return A.copy(), B.copy(), C.copy(), D.copy(), np.array([]), np.array([])

    poles = eigvals(A)

    if discrete:
        stable_mask = np.abs(poles) < (1.0 - tol)
    else:
        stable_mask = np.real(poles) < -tol

    stable_poles = poles[stable_mask]
    unstable_poles = poles[~stable_mask]

    # # If already fully stable, return original realization exactly
    # if np.all(stable_mask):
    #     return A.copy(), B.copy(), C.copy(), D.copy(), stable_poles, unstable_poles

    # # If nothing is stable, return only static gain
    # if not np.any(stable_mask):
    #     A_s = np.zeros((0, 0), dtype=A.dtype)
    #     B_s = np.zeros((0, B.shape[1]), dtype=B.dtype)
    #     C_s = np.zeros((C.shape[0], 0), dtype=C.dtype)
    #     D_s = D.copy()
    #     return A_s, B_s, C_s, D_s, stable_poles, unstable_poles

    # Real ordered Schur decomposition
    def select(x):
        """Return whether an eigenvalue belongs to the stable subspace."""
        if discrete:
            return abs(x) < 1.0 - tol
        return x.real < -tol

    T, Z, sdim = schur(A, output="real", sort=select)

    ns = sdim
    Zs = Z[:, :ns]

    A_s = Zs.T @ A @ Zs
    B_s = Zs.T @ B
    C_s = C @ Zs
    D_s = D.copy()

    # Remove negligible imaginary parts if input was real
    tol_to_real = 1000
    A_s = np.real_if_close(A_s, tol=tol_to_real)
    B_s = np.real_if_close(B_s, tol=tol_to_real)
    C_s = np.real_if_close(C_s, tol=tol_to_real)
    D_s = np.real_if_close(D_s, tol=tol_to_real)

    return A_s, B_s, C_s, D_s, stable_poles, unstable_poles


if __name__ == "__main__":
    # Example: continuous-time unstable system
    from pgopinf.systems.lti_system import LTISystem
    from pgopinf.evaluation.metrics.subroutines.hinf import (
        hinf_error,
        hinf_norm,
    )

    A = np.array([[1.0, 0.0], [0.0, -2.0]])  # unstable
    A = np.array([[-1.0, 1.0], [-3.0, -2.0]])  # stable

    B = np.array([[1.0], [1.0]])
    C = np.array([[1.0, 1.0]])
    D = np.array([[0.0]])

    A_s, B_s, C_s, D_s, stable_poles, unstable_poles = stable_part_lti(
        A, B, C, D, discrete=False
    )

    stable_sys = LTISystem(A=A_s, B=B_s, C=C_s, D=D_s)
    sys = LTISystem(A=A, B=B, C=C, D=D)

    print("Stable poles kept:", stable_poles)
    print("Unstable poles removed:", unstable_poles)
    print("\nStable subsystem:")
    print("A =\n", stable_sys.A)
    print("B =\n", stable_sys.B)
    print("C =\n", stable_sys.C)
    print("D =\n", stable_sys.D)
    print("Eigenvalues of stable subsystem:", stable_sys.eigvals())
    print("Eigenvalues of original system:", sys.eigvals())

    print(
        "H-infinity error between original and stable subsystem:",
        hinf_error(sys, stable_sys),
    )
    print("H-infinity norm of original system:", hinf_norm(sys))
    print("H-infinity norm of stable subsystem:", hinf_norm(stable_sys))

    # Example descriptor system
    E = np.array([[1.0, 0.0], [0.0, 1.0]])

    A = np.array([[1.0, 0.0], [0.0, -2.0]])

    B = np.array([[1.0], [1.0]])
    C = np.array([[1.0, 1.0]])
    D = np.array([[0.0]])

    As, Es, Bs, Cs, Ds, stable_eigs, unstable_eigs = stable_decomposition(
        A, E, B, C, D, discrete=False
    )

    print("Stable generalized eigenvalues kept:", stable_eigs)
    print("Unstable generalized eigenvalues removed:", unstable_eigs)
    print("\nStable descriptor subsystem:")
    print("Es =\n", Es)
    print("As =\n", As)
    print("Bs =\n", Bs)
    print("Cs =\n", Cs)
    print("Ds =\n", Ds)
    print("Eigenvalues of stable descriptor subsystem:", stable_sys.eigvals())
