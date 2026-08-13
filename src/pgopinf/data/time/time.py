import numpy as np
import logging


class Time:
    """Equally spaced one-dimensional time grid."""

    def __init__(self, t) -> None:
        """Initialize and validate a time grid.

        Parameters
        ----------
        t : array_like
            One-dimensional, nonnegative, strictly increasing, equally spaced
            time points.

        Raises
        ------
        ValueError
            If the grid is not one-dimensional, has fewer than two points, is
            negative, non-increasing, or not equally spaced.
        """
        t = np.asarray(t, dtype=float)
        if t.ndim != 1:
            raise ValueError(f"t must be one-dimensional, got shape {t.shape}")
        if t.size < 2:
            raise ValueError("t must contain at least two time points")
        if np.any(t < 0):
            raise ValueError("t must be nonnegative")
        self.t = t

        dt_array = t[1:] - t[:-1]
        if np.any(dt_array <= 0):
            raise ValueError("t must be strictly increasing")
        if dt_array.max() - dt_array.min() >= 1e-12:
            raise ValueError("t must be equally spaced")
        self.dt = dt_array[0]

    @property
    def n_t(self) -> int:
        """Number of time points.

        Returns
        -------
        int
            Length of the time grid.
        """
        return len(self.t)

    @classmethod
    def from_start_end_dt(cls, start, end, dt):
        """Create a time grid from start, end, and time step.

        Parameters
        ----------
        start : float
            Start time.
        end : float
            End time.
        dt : float
            Time step.

        Returns
        -------
        Time
            Equally spaced time grid.
        """
        t = np.arange(start, end + dt, dt)  # +dt to include endpoint
        if t[-1] - end > 1e-10:
            # remove last time step
            t = t[:-1]
            if t[-1] - end > 1e-10:
                logging.warning(
                    f"Multiples of dt do not correspond to the end value {end}. The last time step is {t[-1]}. The difference is {t[-1] - end}."
                )
        return cls(t)

    @classmethod
    def from_start_end_timesteps(cls, start, end, timesteps):
        """Create a time grid with a fixed number of points.

        Parameters
        ----------
        start : float
            Start time.
        end : float
            End time.
        timesteps : int
            Number of time points.

        Returns
        -------
        Time
            Equally spaced time grid.
        """
        t = np.linspace(start, end, timesteps)
        return cls(t)


# class TimeSet:
#     def __init__(
#         self, t_train: Time | np.ndarray, t_test: np.ndarray | Time | None = None
#     ) -> None:
#         # train time
#         if isinstance(t_train, np.ndarray):
#             self.TRAIN = Time(t_train)
#         elif isinstance(t_train, Time):
#             self.TRAIN = t_train
#         else:
#             raise ValueError(f"Unknown type {type(t_train)} for train time.")
#         # test time
#         if t_test is None:
#             self.TEST = t_train
#         elif isinstance(t_test, np.ndarray):
#             self.TEST = Time(t_test)
#         elif isinstance(t_test, Time):
#             self.TEST = t_test
#         else:
#             raise ValueError(f"Unknown type {type(t_test)} for test time.")

#     @classmethod
#     def from_start_end_dt(
#         cls,
#         start_train,
#         end_train,
#         dt_train,
#         start_test=None,
#         end_test=None,
#         dt_test=None,
#     ):
#         t_train_instance = Time.from_start_end_dt(start_train, end_train, dt_train)
#         t_train = t_train_instance.t
#         if start_test is None:
#             # use train time
#             start_test = start_train
#         if end_test is None:
#             end_test = end_train
#         if dt_test is None:
#             dt_test = dt_train
#         t_test_instance = Time.from_start_end_dt(start_test, end_test, dt_test)
#         t_test = t_test_instance.t
#         return cls(t_train, t_test)

#     @classmethod
#     def from_start_end_timesteps(
#         cls,
#         start_train,
#         end_train,
#         timesteps_train,
#         start_test=None,
#         end_test=None,
#         timesteps_test=None,
#     ):
#         t_train_instance = Time.from_start_end_timesteps(
#             start_train, end_train, timesteps_train
#         )
#         t_train = t_train_instance.t
#         if start_test is None:
#             # use train time
#             start_test = start_train
#         if end_test is None:
#             end_test = end_train
#         if timesteps_test is None:
#             timesteps_test = timesteps_train
#         t_test_instance = Time.from_start_end_timesteps(
#             start_test, end_test, timesteps_test
#         )
#         t_test = t_test_instance.t
#         return cls(t_train, t_test)
