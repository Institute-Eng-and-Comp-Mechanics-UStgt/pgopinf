import numpy as np


def convert_to_sample_format(X):
    """
    Convert data from trajectory format (n, n_t, n_sim) to sample format (n, n_samples) where n_samples = n_t * n_sim.

    Parameters
    ----------
    X: np.ndarray of shape (n, n_t, n_sim)
        Data in trajectory format.
    Returns
    -------
    np.ndarray of shape (n, n_samples)
        Data in sample format.
    """
    n, n_t, n_sim = X.shape
    x = np.reshape(X, (n, n_t * n_sim), order="F")
    return x
