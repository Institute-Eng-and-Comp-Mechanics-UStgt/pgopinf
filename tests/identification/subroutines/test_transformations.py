from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from pgopinf.identification.subroutines import transformations
from pgopinf.identification.subroutines.transformations import (
    check_controllability,
    check_observability,
    get_Riccati_transform,
    residual_Riccati,
    solve_Riccati,
)


class FakeSystem:
    def __init__(self, *, A, B, C, D, E=None):
        self.A = A
        self.B = B
        self.C = C
        self.D = D
        self.E = E if E is not None else np.eye(A.shape[0])

    @property
    def abcde(self):
        return self.A, self.B, self.C, self.D, self.E


def test_residual_riccati_matches_manual_formula() -> None:
    A = np.array([[-1.0]])
    B = np.array([[2.0]])
    C = np.array([[3.0]])
    D = np.array([[4.0]])
    E = np.array([[5.0]])
    X = np.array([[6.0]])

    residual = residual_Riccati(A, B, C, D, E, X)

    R = D + D.T
    expected_matrix = (
        -A.T @ X @ E
        - E.T @ X @ A
        - (C.T - E.T @ X @ B) @ np.linalg.inv(R) @ (C - B.T @ X @ E)
    )
    assert residual == np.linalg.norm(expected_matrix)


def test_check_controllability_reports_controllable_system(capsys) -> None:
    A = np.diag([-1.0, -2.0])
    B = np.eye(2)

    check_controllability(A, B)

    assert "System is controllable" in capsys.readouterr().out


def test_check_controllability_reports_uncontrollable_mode(capsys) -> None:
    A = np.diag([-1.0, -2.0])
    B = np.array([[1.0], [0.0]])

    check_controllability(A, B)

    assert "Uncontrollable mode" in capsys.readouterr().out


def test_check_observability_reports_observable_system(capsys) -> None:
    A = np.diag([-1.0, -2.0])
    C = np.eye(2)

    check_observability(A, C)

    assert "System is observable" in capsys.readouterr().out


def test_check_observability_reports_unobservable_mode(capsys) -> None:
    A = np.diag([-1.0, -2.0])
    C = np.array([[1.0, 0.0]])

    check_observability(A, C)

    assert "Unobservable mode" in capsys.readouterr().out


def test_get_riccati_transform_builds_ph_system_from_riccati_solution(monkeypatch) -> None:
    created = []

    class FakePHSystem:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.n = kwargs["J"].shape[0]
            created.append(kwargs)

    monkeypatch.setattr(transformations, "solve_Riccati", lambda system: np.array([[2.0]]))
    monkeypatch.setattr(transformations, "PHSystem", FakePHSystem)
    system = FakeSystem(
        A=np.array([[-2.0]]),
        B=np.array([[6.0]]),
        C=np.array([[4.0]]),
        D=np.array([[3.0]]),
        E=np.array([[5.0]]),
    )

    T, X, ph_system = get_Riccati_transform(system)

    np.testing.assert_allclose(T, np.array([[np.sqrt(2.0)]]))
    np.testing.assert_allclose(X, np.array([[2.0]]))
    assert isinstance(ph_system, FakePHSystem)
    kwargs = created[0]
    np.testing.assert_allclose(kwargs["Q"], np.array([[2.0]]))
    np.testing.assert_allclose(kwargs["J"], np.array([[0.0]]))
    np.testing.assert_allclose(kwargs["R"], np.array([[1.0]]))
    np.testing.assert_allclose(kwargs["G"], np.array([[4.0]]))
    np.testing.assert_allclose(kwargs["P"], np.array([[-2.0]]))
    np.testing.assert_allclose(kwargs["S"], np.array([[3.0]]))
    np.testing.assert_allclose(kwargs["N"], np.array([[0.0]]))
    np.testing.assert_allclose(kwargs["E"], np.array([[5.0]]))


def test_get_riccati_transform_returns_original_system_when_transform_fails(monkeypatch) -> None:
    system = FakeSystem(
        A=np.array([[-1.0]]),
        B=np.array([[0.0]]),
        C=np.array([[0.0]]),
        D=np.array([[1.0]]),
    )

    def raise_error(system):
        raise RuntimeError("boom")

    monkeypatch.setattr(transformations, "solve_Riccati", raise_error)

    T, X, ph_system = get_Riccati_transform(system)

    assert T is None
    assert X is None
    assert ph_system is system


def test_solve_riccati_returns_pymor_gramian_when_kyp_check_passes(monkeypatch) -> None:
    gramian = np.array([[2.0]])
    from_matrices_calls = []
    state_space_calls = []

    class FakePymorLTIModel:
        @classmethod
        def from_matrices(cls, A, B, C, D, E):
            from_matrices_calls.append({"A": A, "B": B, "C": C, "D": D, "E": E})
            return cls()

        def gramian(self, name):
            assert name == "pr_o_dense"
            return gramian

    class FakeStateSpace:
        def __init__(self, A, B, C, D):
            state_space_calls.append({"A": A, "B": B, "C": C, "D": D})

        def ispassive(self):
            return True

    monkeypatch.setattr(transformations, "PymorLTIModel", FakePymorLTIModel)
    monkeypatch.setattr(transformations.ct, "StateSpace", FakeStateSpace)
    monkeypatch.setattr(transformations, "residual_Riccati", lambda *args: 0.0)
    monkeypatch.setattr(transformations, "check_observability", lambda *args: None)
    monkeypatch.setattr(transformations, "check_controllability", lambda *args: None)
    monkeypatch.setattr(transformations, "kyp_lmi", lambda **kwargs: (np.eye(2), True))
    system = FakeSystem(
        A=np.array([[-1.0]]),
        B=np.array([[1.0]]),
        C=np.array([[1.0]]),
        D=np.array([[1.0]]),
        E=np.array([[2.0]]),
    )

    X = solve_Riccati(system)

    np.testing.assert_allclose(X, gramian)
    np.testing.assert_allclose(from_matrices_calls[0]["D"], np.array([[1.000001]]))
    np.testing.assert_allclose(state_space_calls[0]["A"], np.array([[-0.5]]))
    np.testing.assert_allclose(state_space_calls[0]["B"], np.array([[0.5]]))


def test_solve_riccati_returns_none_when_kyp_check_fails(monkeypatch) -> None:
    class FakePymorLTIModel:
        @classmethod
        def from_matrices(cls, *args):
            return cls()

        def gramian(self, name):
            return np.array([[2.0]])

    class FakeStateSpace:
        def __init__(self, *args):
            pass

        def ispassive(self):
            return True

    monkeypatch.setattr(transformations, "PymorLTIModel", FakePymorLTIModel)
    monkeypatch.setattr(transformations.ct, "StateSpace", FakeStateSpace)
    monkeypatch.setattr(transformations, "residual_Riccati", lambda *args: 0.0)
    monkeypatch.setattr(transformations, "check_observability", lambda *args: None)
    monkeypatch.setattr(transformations, "check_controllability", lambda *args: None)
    monkeypatch.setattr(transformations, "kyp_lmi", lambda **kwargs: (np.eye(2), False))
    system = FakeSystem(
        A=np.array([[-1.0]]),
        B=np.array([[1.0]]),
        C=np.array([[1.0]]),
        D=np.array([[1.0]]),
    )

    assert solve_Riccati(system) is None


def test_solve_riccati_uses_standard_state_space_when_e_is_identity(monkeypatch) -> None:
    state_space_calls = []

    class FakePymorLTIModel:
        @classmethod
        def from_matrices(cls, *args):
            return cls()

        def gramian(self, name):
            return np.array([[1.0]])

    class FakeStateSpace:
        def __init__(self, A, B, C, D):
            state_space_calls.append({"A": A, "B": B, "C": C, "D": D})

        def ispassive(self):
            return True

    monkeypatch.setattr(transformations, "PymorLTIModel", FakePymorLTIModel)
    monkeypatch.setattr(transformations.ct, "StateSpace", FakeStateSpace)
    monkeypatch.setattr(transformations, "residual_Riccati", lambda *args: 0.0)
    monkeypatch.setattr(transformations, "check_observability", lambda *args: None)
    monkeypatch.setattr(transformations, "check_controllability", lambda *args: None)
    monkeypatch.setattr(transformations, "kyp_lmi", lambda **kwargs: (np.eye(2), True))
    A = np.array([[-1.0]])
    B = np.array([[1.0]])
    C = np.array([[2.0]])
    D = np.array([[3.0]])
    system = FakeSystem(A=A, B=B, C=C, D=D, E=np.eye(1))

    solve_Riccati(system)

    assert state_space_calls[0] == {"A": A, "B": B, "C": C, "D": D + np.eye(1) * 1e-6}
