import numpy as np


def pick_trajectories(
    data_traj,
    idx_type="rand",
    max_size=4,
    idx: np.ndarray | list | tuple | None = None,
    seed=1,
    return_idx: bool = False,
):
    """
    Pick trajectories from a dataset along a the first dimension (num states or outputs),
    either randomly, by taking the first ones or by providing a list of indices.
    Parameters
    ----------
    data_traj: np.ndarray
        The input trajectory data of shape (n, n_t, n_sim)
    idx_type: str
        The type of index selection to perform. Options are:
        - "rand": Randomly select trajectories (default)
        - "first": Select the first trajectories
        - "idx": Select trajectories based on provided indices in `idx`
    max_size: int
        The maximum number of trajectories to select (default: 4)
    idx: np.ndarray | list | tuple | None
        The specific indices to select when `idx_type` is "idx". Ignored otherwise.
    seed: int
        Random seed for reproducibility when `idx_type` is "rand" (default: 1)
    return_idx: bool
        Whether to return the selected indices along with the data (default: False)
    Returns
    -------
    np.ndarray or tuple[np.ndarray, np.ndarray]
        The selected trajectory data of shape (max_size, n_t, n_sim),
        and optionally the indices of the selected trajectories if `return_idx` is True.
    """

    if data_traj.ndim != 3:
        raise ValueError(
            f"data_traj must be a 3D array of shape (n, n_t, n_sim), got shape {data_traj.shape}"
        )
    n = data_traj.shape[0]
    if max_size > n:
        # use all trajectories
        return data_traj if not return_idx else (data_traj, np.arange(n))

    # index types
    if idx_type == "rand":
        # pick random indices
        rng = np.random.default_rng(seed=seed)
        idx = rng.choice(
            np.arange(0, n),
            size=max_size,
            replace=False,
        )
    elif idx_type == "first":
        # pick first indices
        idx = np.arange(0, max_size)
    elif idx_type == "idx":
        if idx is None:
            raise ValueError(f"List of indices must be given as input 'idx'.")
        assert (
            isinstance(idx, list)
            or isinstance(idx, np.ndarray)
            or isinstance(idx, tuple)
        ), f"idx must be a list, np.ndarray or tuple, got {type(idx)}"
    else:
        raise ValueError(f"Unknown option for cut data type {type}.")

    # pick selected data
    picked_data_traj = np.take(data_traj, idx, axis=0)
    if return_idx:
        return picked_data_traj, idx
    else:
        return picked_data_traj
