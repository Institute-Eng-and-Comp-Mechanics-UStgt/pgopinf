import logging
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigvalsh
from scipy.sparse import coo_matrix

from scipy.sparse.linalg import LinearOperator, lsmr


from pgopinf.numerics.linalg.definiteness import project_spd
from pgopinf.reduction.project_data import define_reduction_projector

logger = logging.getLogger(__name__)


def hamiltonian_identification_cvx(
    x, Ham, project: bool = False, solver: str = "mosek", epsilon: float = 1e-10
):
    """
    Identify the energy matrix Q from Hamiltonian data using convex optimization.
    """

    import cvxpy as cp

    n = x.shape[0]
    n_data = x.shape[1]
    Q = cp.Variable((n, n), symmetric=True)
    constraints = [Q - epsilon * np.eye(n) >> 0]

    # # Define the Hamiltonian values for each data point
    # # Ham_pred = np.zeros(n_data)
    # Ham_pred = []
    # for i in range(n_data):
    #     # Ham_pred.append(0.5 * x[:, i].T @ Q @ x[:, i])
    #     Ham_pred.append(0.5 * cp.quad_form(x[:, i], Q))
    # # Ham_pred = 0.5 * cp.quad_form(x, Q)
    # # Ham_pred = 0.5 * cp.sum(cp.multiply(x.T @ Q @ x, np.eye(n)))

    Ham_pred = []
    for i in range(n_data):
        xi = x[:, i]  # shape (n,)
        Ham_pred.append(0.5 * cp.trace(Q @ np.outer(xi, xi)))

    Ham_pred = cp.hstack(Ham_pred)  # shape (n_data,) equals 0.5 * x.T @ Q @ x

    # Define the loss function
    loss = cp.norm((Ham_pred - Ham), 2)

    # solve optimization problem
    solver_kwargs = {}
    if solver.lower() == "mosek":
        solver = cp.MOSEK
    elif solver.lower() == "scs":
        solver = cp.SCS
        solver_kwargs = {"max_iters": 10000}
    elif solver.lower() == "clarabel":
        solver = cp.CLARABEL
        solver_kwargs = {"max_iter": 50}
    else:
        # try to use input
        solver = solver

    # Define the optimization problem
    prob = cp.Problem(cp.Minimize(loss), constraints=constraints, **solver_kwargs)

    # Solve the problem using the specified solver
    prob.solve(solver=solver)

    if project:
        Q = project_spd(Q.value)
    else:
        Q = Q.value

    return Q


def hamiltonian_identification(x, Ham, project=False, V=None):
    """
    Identify the energy matrix Q from Hamiltonian data
    """

    if V is not None:
        # Project data onto V if provided
        projector = define_reduction_projector(V, V)
        x = projector @ x

    # Assemble data matrix
    n = x.shape[0]
    n_data = x.shape[1]
    min_data_points = n * (n + 1) // 2
    if n_data < min_data_points:
        logging.warning(
            f"There are not enough data points to identify Q from the Hamiltonian. It should be at least {min_data_points} but {n_data} data points were given."
        )

    xx = np.zeros((n_data, min_data_points))
    Dn = duplication_matrix(n)
    for i in range(n_data):
        xx[i, :] = (
            np.kron(x[:, i][:, np.newaxis], x[:, i][:, np.newaxis]).reshape(-1) @ Dn
        )
    # xx = xx @ Dn.toarray()

    # Solve for the symmetric part of Q
    Qvech = np.linalg.lstsq(0.5 * xx, Ham, rcond=None)[0]
    Qvech = Dn @ Qvech
    Q_id = Qvech.reshape(n, n)

    if project:
        # Project onto spsd matrices
        Q_id = project_spd(Q_id)

    if V is not None:
        # Lift back to full space if V was provided
        Q_id = V @ Q_id @ V.T

    return Q_id


def duplication_matrix(n):
    """
    Create the duplication matrix of size n^2 x n(n+1)/2.

    Parameters:
    n (int): The dimension parameter.

    Returns:
    coo_matrix: The duplication matrix.
    """
    m = n * (n + 1) // 2
    nsq = n**2
    r = 0
    a = 0
    v = np.zeros(nsq, dtype=int)
    cn = np.cumsum(np.arange(n, 1, -1)) - 1

    for i in range(1, n + 1):
        if i > 1:
            v[r : r + i - 1] = i - n + cn[: i - 1]
            r += i - 1

        v[r : r + n - i + 1] = np.arange(a, a + n - i + 1)
        r += n - i + 1
        a += n - i + 1

        # print(v)
        # print(f"round {i}")
    rows = np.arange(nsq)
    cols = v
    data = np.ones(nsq)

    D = coo_matrix((data, (rows, cols)), shape=(nsq, m))
    return D


def _upper_tri_indices(n):
    return np.triu_indices(n)


def _unpack_symmetric_upper(q, n):
    """
    Unpack q (upper-triangular entries) into a full symmetric matrix Q.
    """
    Q = np.zeros((n, n), dtype=q.dtype)
    iu = _upper_tri_indices(n)
    Q[iu] = q
    Q[(iu[1], iu[0])] = q
    return Q


def _make_hamiltonian_operator(X):
    """
    Create a LinearOperator A such that

        (A @ q)[i] = 0.5 * x_i^T Q x_i

    where q packs the upper-triangular entries of symmetric Q.

    Parameters
    ----------
    X : ndarray, shape (n, n_data)
        Data matrix whose columns are samples x_i.

    Returns
    -------
    A : scipy.sparse.linalg.LinearOperator
        Implicit least-squares operator of shape (n_data, n*(n+1)//2).
    """
    n, n_data = X.shape
    p = n * (n + 1) // 2
    iu = _upper_tri_indices(n)

    def matvec(q):
        """
        y = A @ q
        """
        Q = _unpack_symmetric_upper(q, n)
        # y_i = 0.5 * x_i^T Q x_i
        QX = Q @ X
        y = 0.5 * np.sum(X * QX, axis=0)
        return y

    def rmatvec(y):
        """
        z = A^T @ y

        Let M = sum_i y_i x_i x_i^T = X diag(y) X^T.
        Then:
          - diagonal packed entries get 0.5 * M_jj
          - off-diagonal packed entries get M_jk
        """
        # Compute M = X diag(y) X^T without forming diag(y)
        M = (X * y[np.newaxis, :]) @ X.T

        z = M[iu].copy()
        diag_mask = iu[0] == iu[1]
        z[diag_mask] *= 0.5
        return z

    return LinearOperator(
        shape=(n_data, p),
        matvec=matvec,
        rmatvec=rmatvec,
        dtype=X.dtype,
    )


def hamiltonian_identification_lsmr(
    x,
    Ham,
    project=False,
    damp=0.0,
    atol=1e-8,
    btol=1e-8,
    conlim=1e8,
    maxiter=None,
    show=False,
):
    """
    Identify the symmetric energy matrix Q from Hamiltonian data using LSMR
    without explicitly assembling the dense least-squares matrix.

    Solves:
        min_Q || 0.5 * x_i^T Q x_i - Ham_i ||_2
        subject to Q = Q^T

    Parameters
    ----------
    x : ndarray, shape (n, n_data)
        Data matrix, each column is one sample x_i.
    Ham : ndarray, shape (n_data,)
        Hamiltonian values corresponding to the columns of x.
    project : bool, optional
        If True, project the identified matrix onto SPD/SPSD using project_spd.
    damp : float, optional
        Tikhonov damping passed to scipy.sparse.linalg.lsmr.
        Useful if the problem is ill-conditioned.
    atol, btol, conlim, maxiter, show :
        Parameters forwarded to scipy.sparse.linalg.lsmr.

    Returns
    -------
    Q_id : ndarray, shape (n, n)
        Identified symmetric matrix.
    info : dict
        Solver diagnostics.
    """
    n = x.shape[0]
    n_data = x.shape[1]
    p = n * (n + 1) // 2

    if Ham.shape[0] != n_data:
        raise ValueError(
            f"Shape mismatch: x has {n_data} samples but Ham has length {Ham.shape[0]}."
        )

    if n_data < p:
        logging.warning(
            f"There are not enough data points to uniquely identify a general symmetric "
            f"Q. Need at least {p}, but only {n_data} data points were given."
        )

    A = _make_hamiltonian_operator(x)

    result = lsmr(
        A,
        Ham,
        damp=damp,
        atol=atol,
        btol=btol,
        conlim=conlim,
        maxiter=maxiter,
        show=show,
    )

    q = result[0]
    Q_id = _unpack_symmetric_upper(q, n)

    if project:
        Q_id = project_spd(Q_id)

    info = {
        "istop": result[1],
        "itn": result[2],
        "normr": result[3],
        "normar": result[4],
        "norma": result[5],
        "conda": result[6],
        "normx": result[7],
    }

    return Q_id, info


def test_hamiltonian_identification():
    """Run a manual Hamiltonian-identification smoke test.

    The function generates synthetic Hamiltonian data, identifies an energy
    matrix with the selected method, prints diagnostics, and writes a diagnostic
    plot to ``hamIdentification.png``.
    """
    # test case
    # Small script to test if the Hamiltonian identification problem is reasonable

    method = "lstsq"  # "lstsq" or "lsmr" or "cvx"

    np.random.seed(0)

    example = "msd"  # "diagonal", "msd", "random", "high_dim"
    n = 5
    nrData = n**2

    if example == "diagonal":
        Q = np.diag(np.arange(1, n + 1))
    elif example == "msd":
        Q = np.array(
            [
                [4, 0, -4, 0, 0, 0],
                [0, 1 / 4, 0, 0, 0, 0],
                [-4, 0, 8, 0, -4, 0],
                [0, 0, 0, 1 / 4, 0, 0],
                [0, 0, -4, 0, 8, 0],
                [0, 0, 0, 0, 0, 1 / 4],
            ]
        )
        n = 6
        nrData = n**2
    elif example == "random":
        Q = 3 * np.random.rand(n, n)
        Q = Q.T @ Q
    elif example == "high_dim":
        n = 500
        Q = 3 * np.random.rand(n, n)
        Q = Q.T @ Q
        nrData = n * (n + 1) // 2 + 50  # more than the number of parameters to identify
    else:
        raise ValueError("example not known")

    print(f"Example:\t\t\t {example}")
    print(f"Dimension:\t\t\t {n}")
    print(f"maximal training data:\t\t {nrData} (n^2 = {n**2})")
    print(f"minimal eigenvalue of Q:\t {min(eigvalsh(Q)):.3e}")

    import time

    t0 = time.perf_counter()
    X = np.random.rand(n, nrData)
    Ham = 0.5 * np.array([X[:, i].T @ Q @ X[:, i] for i in range(nrData)])

    # reduce
    V, S, Vh = np.linalg.svd(X)
    r = 3

    Qred = Q @ V[:, :r]

    t1 = time.perf_counter()

    t2 = time.perf_counter()
    H = 0.5 * np.einsum("ij,ij->j", X, Q @ X)
    t3 = time.perf_counter()

    print(f"Time for loop computation of Ham: {t1 - t0} seconds")
    print(f"Time for einsum computation of H: {t3 - t2} seconds")

    # # %% time kronecker product assembly
    # t_kron1 = time.perf_counter()
    # # Assemble data matrix
    # n = X.shape[0]
    # n_data = X.shape[1]
    # min_data_points = n * (n + 1) // 2
    # xx = np.zeros((n_data, min_data_points))
    # Dn = duplication_matrix(n)
    # for i in range(n_data):
    #     xx[i, :] = np.squeeze(kron(X[:, i][:, np.newaxis], X[:, i][:, np.newaxis])) @ Dn
    # t_kron2 = time.perf_counter()

    # t_kron3 = time.perf_counter()
    # tri = np.triu_indices(n)
    # xx2 = (X[tri[0], :] * X[tri[1], :]).T
    # print(f"Are xx and xx2 close? {np.allclose(xx, xx2)}")
    # t_kron4 = time.perf_counter()

    # print(f"Time for loop assembly of xx: {t_kron2 - t_kron1} seconds")
    # print(f"Time for vectorized assembly of xx2: {t_kron4 - t_kron3} seconds")

    if nrData > 200:
        ii = [nrData]

    elif nrData > 50:
        ii = np.linspace(1, nrData + 1, 8, dtype=np.int32)

    else:
        ii = np.arange(1, nrData + 1, 2)

    err = np.zeros(len(ii))
    errQred = np.zeros(len(ii))
    errAfterProjection = np.zeros(len(ii))
    errAfterProjectionQred = np.zeros(len(ii))
    minEigVal = np.zeros(len(ii))
    minEigValAfterProjection = np.zeros(len(ii))

    for idx, i in enumerate(ii):
        # identify from Hamiltonian
        print(f"Test Hamiltonian identification for {i} data samples")
        if method == "lsmr":
            Q_id, info = hamiltonian_identification_lsmr(
                X[:, :i],
                Ham[:i],
                project=False,
                damp=0.0,
                atol=1e-10,
                btol=1e-10,
                maxiter=200,
            )
        elif method == "lstsq":
            Q_id = hamiltonian_identification(X[:, :i], Ham[:i])
        elif method == "cvx":
            Q_id = hamiltonian_identification_cvx(
                X[:, :i], Ham[:i], solver="mosek", epsilon=1e-10
            )
        # Performance measures
        minEigVal[idx] = min(eigvalsh(Q_id))
        err[idx] = np.linalg.norm(Q - Q_id) / np.linalg.norm(Q)
        print(f"Diag Q: {np.diag(Q)}")
        print(f"Diag Q_id: {np.diag(Q_id)}")
        # errQred[idx] = np.linalg.norm(Qred - Q_id @ V[:, :r]) / np.linalg.norm(Qred)
        if method == "lsmr":
            Q_id_projected, _ = hamiltonian_identification_lsmr(
                X[:, :i],
                Ham[:i],
                project=True,
                damp=0.0,
                atol=1e-10,
                btol=1e-10,
                maxiter=200,
            )
        elif method == "lstsq":
            Q_id_projected = hamiltonian_identification(X[:, :i], Ham[:i], project=True)

        # errAfterProjectionQred[idx] = np.linalg.norm(
        #     Qred - Q_id_projected @ V[:, :r]
        # ) / np.linalg.norm(Qred)
        elif method == "cvx":
            Q_id_projected = hamiltonian_identification_cvx(
                X[:, :i], Ham[:i], project=True, solver="mosek", epsilon=1e-10
            )

        minEigValAfterProjection[idx] = min(eigvalsh(Q_id_projected))
        errAfterProjection[idx] = np.linalg.norm(Q - Q_id_projected) / np.linalg.norm(Q)
    print(f"Error without projection {err}.")
    print(f"Error with projection {errAfterProjection}.")
    # print(f"Error with projection and reduction {errAfterProjectionQred}.")

    # Plot figures
    m = n * (n + 1) // 2
    fig, (ax1, ax2) = plt.subplots(2, 1)

    ax1.semilogy(ii, err, label="err")
    # ax1.semilogy(ii, errQred, label="err reduced")
    ax1.semilogy(ii, errAfterProjection, label="err after projection", linestyle="--")
    ax1.semilogy(
        ii,
        errAfterProjectionQred,
        label="err after projection and reduction",
        linestyle="-.",
    )
    ax1.axvline(m, color="k", linestyle="--")
    ax1.set_ylabel("rel. error ||H-H_id||_F/||H||_F")
    ax1.legend()
    # ax1.set_aspect("equal", "box")

    ax2.plot(ii, minEigVal, label="min. eigenvalue")
    ax2.plot(
        ii,
        minEigValAfterProjection,
        label="min. eigenvalue after projection",
        linestyle="--",
    )
    ax2.axvline(m, color="k", linestyle="--")
    ax2.legend(loc="lower right")
    ax2.set_ylabel("min. eigenvalue")
    ax2.set_xlabel("n_data")
    # ax2.set_aspect("equal", "box")

    plt.savefig("hamIdentification.png")
    plt.show(block=False)


if __name__ == "__main__":
    test_hamiltonian_identification()
