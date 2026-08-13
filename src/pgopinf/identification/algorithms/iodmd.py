import logging
import numpy as np

from pgopinf.systems.utils.unstack import unstack


def iodmd(
    X,
    Y,
    U,
    X1=None,
    E=None,
    seperate_output_inf: bool = False,
    rcond: float = 1e-12,
    lambda_reg: float = 0.0,
):
    """
    Input Output Dynamic Mode Decomposition.

    From [MorandinNicodemusUnger22]

    Parameters
    ----------
    X : numpy.ndarray
        Sequence of states.
    Y : numpy.ndarray
        Sequence of outputs.
    U : numpy.ndarray
        Sequence of inputs.
    X1 : numpy.ndarray, optional
        Shifted sequence of states. If not given the shifted matrix is obtained from `X`.
    E : numpy.ndarray, optional
        E matrix of the LTI system. If not given this is assumed to be the identity.
    seperate_output_inf : bool, optional
        If True, perform operator inference with separate output inference.
        For testing: Should not influence the result.
    Returns
    -------
    A : numpy.ndarray
        The identified state matrix.
    e : float
        The relative error of the DMD problem.
    """
    if X1 is None:
        X1 = X[:, 1:]
        X0 = X[:, :-1]
        U = U[:, :-1]
        Y = Y[:, :-1]
    else:
        X0 = X

    # T_unscaled = np.concatenate((X0, U))

    # X0, sx = row_scale(X0)
    # U, su = row_scale(U)

    # logging.warning(f"Using minus Y in iodmd, i.e. Y = -Y.")
    # Y = -Y

    if E is None:
        E = np.eye(X.shape[0])

    T = np.concatenate((X0, U))
    Z = np.concatenate((E @ X1, Y))

    assert lambda_reg >= 0.0
    if lambda_reg > 0.0:
        logging.info(f"Using ridge regression with lambda={lambda_reg:.2e}.")
        n_T = T.shape[0]
        T_reg = np.hstack((T, lambda_reg * np.eye(n_T)))
        Z_reg = np.hstack((Z, np.zeros((n_T, Z.shape[0]))))
        T = T_reg
        Z = Z_reg

    # check condition of data matrix T
    cond_T = np.linalg.cond(T)
    if cond_T > 1e10:
        logging.warning(
            f"Data matrix T in OpInf is ill-conditioned with condition number {cond_T:.2e}."
        )
        # logging.info(
        #     f"Minimal singular value of T is {np.min(np.linalg.svdvals(T)):.2e}."
        # )
        # logging.info(
        #     f"Maximal singular value of T is {np.max(np.linalg.svdvals(T)):.2e}."
        # )
    else:
        logging.info(f"Data matrix T in OpInf has condition number {cond_T:.2e}.")
        # logging.info(
        #     f"Minimal singular value of T is {np.min(np.linalg.svdvals(T)):.2e}."
        # )
        # logging.info(
        #     f"Maximal singular value of T is {np.max(np.linalg.svdvals(T)):.2e}."
        # )
        # # norm of pseudo inverse
        # logging.info(
        #     f"Norm of pseudoinverse of T is {np.linalg.norm(np.linalg.pinv(T, rcond=rcond)):.2e}."
        # )

    pinv_method = "numpy"  # numpy | ridge
    if pinv_method == "numpy":
        pseudo_inv_calculation = lambda Z, T: Z @ np.linalg.pinv(T, rcond=rcond)
    elif pinv_method == "ridge":
        pseudo_inv_calculation = lambda Z, T: (Z @ T.T) @ np.linalg.inv(T @ T.T)
    else:
        raise ValueError(
            f"Unknown pseudoinverse method {pinv_method}. Supported are 'numpy' and 'ridge'."
        )
    if not seperate_output_inf:
        # standard infer A,B,C,D together
        logging.info("Perform ioDMD")

        # Solve ioDMD
        Acal = pseudo_inv_calculation(Z, T)
    else:
        # infer A,B first and C,D second
        logging.info("Perform OpInf with separate output inference")
        T1 = np.concatenate((X0, U))
        Z1 = E @ X1
        Acal1 = pseudo_inv_calculation(Z1, T1)
        Acal = Acal1
        Z2 = Y
        C_D = pseudo_inv_calculation(Z2, T1)
        Acal = np.vstack((Acal, C_D))

    # # rescale
    # n = X0.shape[0]
    # K = Acal.copy()
    # K[:, :n] /= sx[None, :]
    # K[:, n:] /= su[None, :]
    # Acal = K
    # print(f"Condition number of T_scaled: {np.linalg.cond(T):.2e}")
    # T = T_unscaled

    e_absolute = np.linalg.norm(Z - Acal @ T)
    e_rel = e_absolute / np.linalg.norm(Z)
    logging.info("ioDMD Result")
    logging.info(f"Absolute: ||Z - A@T||_F = {e_absolute:.2e}")
    logging.info(f"Relative: ||Z - A@T||_F / ||Z||_F = {e_rel:.2e}")

    # # load full system matrices
    # full_system_matrices = np.load("full_system_matrices.npz")
    # A_full = full_system_matrices["A"]
    # B_full = full_system_matrices["B"]
    # C_full = full_system_matrices["C"]
    # D_full = full_system_matrices["D"]
    # E_full = full_system_matrices["E"]
    # # calculate residual of E,A,B with data
    # np.linalg.norm(E_full @ X1 - A_full @ X0 - B_full @ U, "fro")
    # # calculate residual of C,D with data
    # np.linalg.norm(Y - C_full @ X0 - D_full @ U, "fro")

    # # Calculate error of block residuals
    # K = np.block(
    #     [
    #         [A_full, B_full],
    #         [C_full, D_full],
    #     ]
    # )
    # K_id = Acal
    # rel_error_res_full = np.linalg.norm(Z - K @ T, "fro") / np.linalg.norm(Z, "fro")
    # rel_error_res_id = np.linalg.norm(Z - K_id @ T, "fro") / np.linalg.norm(Z, "fro")
    # print(f"Relative residual error full system: {rel_error_res_full:.2e}")
    # print(f"Relative residual error identified system: {rel_error_res_id:.2e}")
    # print(f"Matrix rank of T: {np.linalg.matrix_rank(T)} / {T.shape[0]}")
    # print(f"Condition number of T: {np.linalg.cond(T):.2e}")
    # print(f"Minimal singular value of T: {np.min(np.linalg.svdvals(T)):.2e}")

    return Acal, e_rel


def row_scale(A, eps=1e-16):
    """Scale matrix rows by their Euclidean norms.

    Parameters
    ----------
    A : ndarray, shape (m, n)
        Matrix to scale.
    eps : float, optional
        Lower bound for row scaling factors.

    Returns
    -------
    A_scaled : ndarray, shape (m, n)
        Row-scaled matrix.
    s : ndarray, shape (m,)
        Row scaling factors.
    """
    s = np.linalg.norm(A, axis=1)  # should be (rows,)
    s = np.asarray(s).reshape(-1)  # force 1D
    s = np.maximum(s, eps)
    A_scaled = A / s[:, None]  # broadcast (rows,1)
    return A_scaled, s


if __name__ == "__main__":
    logging.basicConfig()
    logging.getLogger().setLevel(logging.INFO)

    # simple test
    n = 8
    m = 2
    p = 3
    k = 100

    # define seed
    np.random.seed(42)
    A_true = np.random.randn(n, n)
    B_true = np.random.randn(n, m)
    C_true = np.random.randn(p, n)
    D_true = np.random.randn(p, m)
    E_true = np.eye(n) + 0.1 * np.random.randn(n, n)

    X = np.random.randn(n, k)
    U = np.random.randn(m, k)
    Y = C_true @ X + D_true @ U
    X1 = np.linalg.solve(E_true, (A_true @ X + B_true @ U))

    Acal, e_rel = iodmd(X, Y, U, X1=X1, E=E_true)
    A_id, B_id, C_id, D_id = unstack(Acal, n)

    print(f"Relative error A: {np.linalg.norm(A_true - A_id)/np.linalg.norm(A_true)}")
    print(f"Relative error B: {np.linalg.norm(B_true - B_id)/np.linalg.norm(B_true)}")
    print(f"Relative error C: {np.linalg.norm(C_true - C_id)/np.linalg.norm(C_true)}")
    print(f"Relative error D: {np.linalg.norm(D_true - D_id)/np.linalg.norm(D_true)}")
