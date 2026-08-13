import numpy as np

from pgopinf.data.hamiltonian_identification import duplication_matrix


def get_supplied_power(y, u):
    """Compute supplied power samples from output and input data.

    Parameters
    ----------
    y : ndarray, shape (n_y, n_data)
        Output samples.
    u : ndarray, shape (n_u, n_data)
        Input samples.

    Returns
    -------
    ndarray, shape (n_data, 1)
        Sample-wise inner products ``y[:, i] @ u[:, i]``.
    """
    n_data = y.shape[1]
    supplied_power = np.zeros((n_data, 1))
    for i_sample in range(n_data):
        supplied_power[i_sample, 0] = y[:, i_sample] @ u[:, i_sample].T
    return supplied_power


def get_kron_x_dx_data(x, dxdt, symmetric=True):
    """Compute Kronecker products between state and derivative samples.

    Parameters
    ----------
    x : ndarray, shape (n, n_data)
        State samples.
    dxdt : ndarray, shape (n, n_data)
        Derivative samples.
    symmetric : bool, optional
        If ``True``, map products through the duplication matrix.

    Returns
    -------
    ndarray
        Kronecker-product data matrix.
    """
    kron_x_dx = get_kron_data(x, dxdt, symmetric=symmetric)
    return kron_x_dx


def get_kron_x_x_data(x, symmetric=True):
    """Compute Kronecker products between state samples.

    Parameters
    ----------
    x : ndarray, shape (n, n_data)
        State samples.
    symmetric : bool, optional
        If ``True``, map products through the duplication matrix.

    Returns
    -------
    ndarray
        Kronecker-product data matrix.
    """
    kron_x_x = get_kron_data(x, x, symmetric=symmetric)
    return kron_x_x


def get_kron_data(state1, state2, symmetric=True):
    """Compute sample-wise Kronecker products.

    Parameters
    ----------
    state1 : ndarray, shape (n, n_data)
        First sample matrix.
    state2 : ndarray, shape (n, n_data)
        Second sample matrix.
    symmetric : bool, optional
        If ``True``, return compressed symmetric coordinates using the
        duplication matrix.

    Returns
    -------
    ndarray
        Matrix whose rows contain sample-wise Kronecker products.
    """
    n = state1.shape[0]
    n_data = state1.shape[1]
    kron_state1_state2 = np.zeros((n_data, n**2))

    Dn = duplication_matrix(n)
    for i_sample in range(n_data):
        kron_state1_state2[i_sample, :] = np.kron(
            state1[:, i_sample][:, np.newaxis], state2[:, i_sample][:, np.newaxis]
        ).reshape(-1)
    if symmetric:
        kron_state1_state2 = kron_state1_state2 @ Dn.toarray()

    return kron_state1_state2
