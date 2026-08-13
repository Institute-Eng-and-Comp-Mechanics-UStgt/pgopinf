import numpy as np

# from pgopinf.data.data import Data


class InitialCondition:
    """Initial conditions for one or more simulations.

    Attributes
    ----------
    x0 : ndarray, shape (n, n_sim)
        Initial state columns for each simulation.
    """

    def __init__(self, x0) -> None:
        """Initialize initial conditions.

        Parameters
        ----------
        x0 : array_like, shape (n, n_sim)
            Initial state columns.

        Raises
        ------
        ValueError
            If ``x0`` is not two-dimensional.
        """
        x0 = np.asarray(x0)
        if x0.ndim != 2:
            raise ValueError(f"x0 must have shape (n, n_sim), got {x0.shape}")
        self.x0 = x0

    @classmethod
    def repeat_n_sim_times(cls, x0, n_sim):
        """Repeat one initial condition for multiple simulations.

        Parameters
        ----------
        x0 : ndarray, shape (n,) or (n, 1)
            Initial state to repeat.
        n_sim : int
            Number of simulation columns to create.

        Returns
        -------
        InitialCondition
            Initial-condition object with shape ``(n, n_sim)``.
        """
        if x0.ndim == 1:
            x0 = x0[:, np.newaxis]
        x0 = np.tile(x0, (1, n_sim))
        return cls(x0)

    @classmethod
    def create_random_initial_conditions(
        cls, n: int, n_sim: int, seed: int | None = None, scaling: float = 1.0
    ):
        """Create random initial conditions.

        Parameters
        ----------
        n : int
            State dimension.
        n_sim : int
            Number of simulations.
        seed : int, optional
            Random seed.
        scaling : float, optional
            Multiplicative scale applied to samples from ``[0, 1)``.

        Returns
        -------
        InitialCondition
            Random initial conditions with shape ``(n, n_sim)``.
        """
        rng = np.random.default_rng(seed=seed)
        x0 = scaling * rng.random((n, n_sim))
        return cls(x0)

    def reduce(self, projector: np.ndarray):
        """Project initial conditions to a reduced space.

        Parameters
        ----------
        projector : ndarray, shape (r, n)
            Projection matrix applied to ``x0``.

        Returns
        -------
        InitialCondition
            Reduced initial conditions with shape ``(r, n_sim)``.
        """
        x0r = projector @ self.x0
        # x0r = V.T @ Q.T @ self.x0

        return InitialCondition(x0=x0r)


# class InitialConditionSet:
#     def __init__(self, x0_train, x0_test=None) -> None:
#         if isinstance(x0_train, InitialCondition):
#             self.TRAIN = x0_train
#         else:
#             self.TRAIN = InitialCondition(x0_train)
#         if x0_test is None:
#             x0_test = x0_train
#         if isinstance(x0_test, InitialCondition):
#             self.TEST = x0_test
#         else:
#             self.TEST = InitialCondition(x0_test)

#     def reduce(self, projector: np.ndarray):
#         x0r_train = self.TRAIN.reduce(projector=projector)
#         x0r_test = self.TEST.reduce(projector=projector)
#         return InitialConditionSet(x0r_train, x0r_test)

#     @classmethod
#     def repeat_n_sim_times(cls, x0_train, n_sim_train, x0_test=None, n_sim_test=None):
#         initial_condition_train = InitialCondition.repeat_n_sim_times(
#             x0_train, n_sim_train
#         )
#         if x0_test is None:
#             x0_test = x0_train
#         if n_sim_test is None:
#             n_sim_test = n_sim_train
#         initial_condition_test = InitialCondition.repeat_n_sim_times(
#             x0_test, n_sim_test
#         )
#         return cls(
#             x0_train=initial_condition_train.x0, x0_test=initial_condition_test.x0
#         )

#     @classmethod
#     def create_random_initial_conditions(
#         cls,
#         n: int,
#         n_sim_train: int,
#         n_sim_test: int | None = None,
#         seed: int | None = None,
#     ):
#         """
#         n: (int) state dimension
#         n_sim_train: (int) number of initial conditions for training
#         n_sim_test: (int|None) number of initial conditions for testing
#         seed: (int|None) random seed
#         """
#         initial_condition_train = InitialCondition.create_random_initial_conditions(
#             n, n_sim=n_sim_train, seed=seed
#         )

#         if n_sim_test is None:
#             initial_condition_test = initial_condition_train
#         else:
#             seed_test = (
#                 seed + 1 if seed is not None else None
#             )  # avoid same random numbers
#             initial_condition_test = InitialCondition.create_random_initial_conditions(
#                 n, n_sim=n_sim_test, seed=seed_test
#             )
#         return cls(
#             x0_train=initial_condition_train.x0, x0_test=initial_condition_test.x0
#         )
