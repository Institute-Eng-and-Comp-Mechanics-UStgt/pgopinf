import numpy as np


def error_time_trajectories(
    ref_traj,
    cmp_traj,
    relative=True,
    rel_error_type: str = "mean_over_time",
    rel_threshold: float = 1e-8,
) -> dict[str, np.ndarray | float]:
    """
    Compute error trajectories over time between reference and comparison trajectories.
    Parameters
    ----------
    ref_traj: np.ndarray
        Reference trajectory of shape (n, n_t, n_sim)
    cmp_traj: np.ndarray
        Comparison trajectory of shape (n, n_t, n_sim)
    relative: bool
        Whether to compute relative error (default: True)
    rel_error_type: str
        Type of relative error to compute (default: "mean_over_time")
        - options:
        -- "mean_over_time":
            Compute mean over time for each state and simulation, then use this mean to compute relative error.
        -- "individual_time_step_with_treshold"
            Compute relative error for each individual time step, but apply a threshold to avoid division by small numbers.
    rel_threshold: float
        Threshold for relative error to avoid division by small numbers (default: 1e-8)
    Returns
    -------
    dict[str, np.ndarray | float]
        Dictionary containing error trajectories and overall error metrics:
        - "error_value": np.ndarray of shape (n, n_t, n_sim) with error values over time
        - "overall_max_error": float with maximum error over all time steps and simulations
        - "overall_mean_error": float with mean error over all time steps and simulations
    """

    # absolute error
    error_abs = np.abs(ref_traj - cmp_traj)  # (n, n_t, n_sim)
    overall_max_abs_error = np.max(error_abs)  # (n_t, n_sim)
    overall_mean_abs_error = np.mean(error_abs)  # (n_t, n_sim)

    if relative:
        if rel_error_type == "mean_over_time":
            # calculate mean over time trajectories for absolute value of the original state
            mean_over_time = np.mean(
                np.abs(ref_traj), axis=1
            )  # (n, n_sim) - mean over time
            # repeat to work element-wise
            mean_over_time = np.repeat(
                np.expand_dims(mean_over_time, axis=1),
                ref_traj.shape[1],
                axis=1,
            )  # (n, n_t, n_sim) - repeat mean over time to match original shape
            # values of some states can still be close to zero, use relative treshold
            mean_over_time[np.abs(mean_over_time) < rel_threshold] = rel_threshold
            error_rel = error_abs / mean_over_time
            overall_max_rel_error = np.max(error_rel)
            overall_mean_rel_error = np.mean(error_rel)

        elif rel_error_type == "individual_time_step_with_treshold":

            # values of some states can be close to zero, use relative treshold
            ref_traj[np.abs(ref_traj) < rel_threshold] = (
                rel_threshold  # (n, n_t, n_sim)
            )

            error_rel = error_abs / ref_traj  # (n, n_t, n_sim)
            overall_max_rel_error = np.max(error_rel)  # scalar
            overall_mean_rel_error = np.mean(error_rel)  # scalar

        else:
            raise ValueError(f"Unknown rel_error_type {rel_error_type!r}")

        error_value = error_rel
        overall_mean_error = overall_mean_rel_error
        overall_max_error = overall_max_rel_error
    else:
        error_value = error_abs
        overall_mean_error = overall_mean_abs_error
        overall_max_error = overall_max_abs_error

    # result dict
    result = {
        "error_value": error_value,
        "overall_max_error": overall_max_error,
        "overall_mean_error": overall_mean_error,
    }
    return result
