from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from pgopinf.reduction.interfaces import BasisArtifact


def test_basis_artifact_exposes_reduced_dimension() -> None:
    basis = BasisArtifact(
        V=np.zeros((5, 3)),
        W=np.ones((5, 3)),
        meta={"kind": "test"},
    )

    assert basis.r == 3
    assert basis.meta == {"kind": "test"}


def test_basis_artifact_is_frozen_value_object() -> None:
    basis = BasisArtifact(V=np.zeros((2, 1)), W=None, meta={})

    with pytest.raises(FrozenInstanceError):
        basis.W = np.ones((2, 1))  # type: ignore[misc]
