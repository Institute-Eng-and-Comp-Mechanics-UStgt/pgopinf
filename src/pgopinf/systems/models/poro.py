import os
import logging

import numpy as np

from scipy.io import loadmat

from pgopinf.systems.ph_system import PHSystem

logger = logging.getLogger(__name__)


def poro(
    n_system: int = 980,
    use_Berlin: bool = False,
    use_mimo: bool = True,
    Rshift: float = 1e-3,
) -> PHSystem:
    """
    Code from [MorandinNicodemusUnger22] Port-Hamiltonian Dynamic Mode Decomposition
    Returns a port-Hamiltonian model of linear poroelasticity in a
    bounded Lipschitz domain as described in :cite:`AltMU21`.

    Parameters
    ----------
    n_system : int, optional
        System dimension (can only be either: 320, 980, or 1805). Default = 980.
    use_Berlin : bool, optional
        Whether to use the Berlin form with E=I. Default = False
    use_mimo : bool, optional
        Whether to use the MIMO version of the system. Default = True

    Returns
    -------
    H : numpy.ndarray
        Hamiltonian matrix.
    J : numpy.ndarray
        Skew-symmetric matrix.
    R : numpy.ndarray
        Dissipative matrix.
    G : numpy.ndarray
        Input matrix.
    P : numpy.ndarray
        Input matrix.
    S : numpy.ndarray
        Symmetric part of feed trough matrix.
    N : numpy.ndarray
        Skew-symmetric part of feed trough matrix.
    """
    # load matrices
    assert n_system in [320, 980, 1805]  # only those precalculated versions exist
    working_dir = os.path.dirname(os.path.realpath(__file__))
    data_name = f"poro-n{n_system}"
    path_to_mat = os.path.join(
        working_dir,
        "..",
        "..",
        "..",
        "..",
        "models_data_files",
        "poro",
        f"{data_name}.mat",
    )

    logger.info(f"Loading poroelastic data from matlab file {data_name}.mat")

    data = loadmat(path_to_mat)
    A = data["A"]

    # parameter settings
    rho = 1e-3
    alpha = 0.79
    Minv = 7.80e3
    kappaNu = 633.33
    Rshift = Rshift  # denoted eta in the manuscript AltMU21

    # update discretization matrices to reflect the parameter settings
    Y = rho * data["Y"]
    D = alpha * data["D"]
    M = Minv * data["M"]
    K = kappaNu * data["K"]
    Bp = data["Bp"].T
    Bf = data["Bf"].T

    # construct first-order port-Hamiltonian descriptor system
    # representation, similarly as in
    # R.Altmann, V.Mehrmann, B.Unger: Port - Hamiltonian
    # formulations of poroelastic network models, arXiv preprint
    # arXiv: 2012.01949, 2020.
    n = A.shape[0]
    m = M.shape[0]
    H = np.block(
        [
            [Y, np.zeros((n, n + m))],
            [np.zeros((n, n)), A, np.zeros((n, m))],
            [np.zeros((m, n + n)), M],
        ]
    )

    J = np.block(
        [
            [np.zeros((n, n)), -A, D.T],
            [A, np.zeros((n, n + m))],
            [-D, np.zeros((m, n + m))],
        ]
    )

    R = np.block(
        [
            [np.zeros((n, 2 * n + m))],
            [np.zeros((n, 2 * n + m))],
            [np.zeros((m, 2 * n)), K],
        ]
    ) + Rshift * np.eye(2 * n + m)

    if use_mimo:
        G = np.block(
            [[Bf, np.zeros((n, 1))], [np.zeros((n, 2))], [np.zeros((m, 1)), Bp]]
        )
    else:
        G = np.vstack([np.zeros((n, 1)), Bf, Bp])  # only fluid input, no solid input

    P = np.zeros(G.shape)
    S = np.zeros((G.shape[1], G.shape[1]))
    N = np.zeros((G.shape[1], G.shape[1]))

    if use_Berlin:
        Q = np.eye(J.shape[0])
    else:
        Q = np.linalg.solve(H, np.eye(H.shape[0]))
        H = np.eye(
            H.shape[0]
        )  # set H to identity, and move it into Q, to get standard PH form with E=I

    # create PHSystem instance
    ph_system = PHSystem(J=J, R=R, G=G, Q=Q, E=H, P=P, S=S, N=N)

    return ph_system


if __name__ == "__main__":
    # example call
    poro_model = poro(n_system=980)
