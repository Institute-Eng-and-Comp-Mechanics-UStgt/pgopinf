from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from pgopinf.core.q_matrix_store import QMatrixArtifact, QMatrixStore
from pgopinf.data.dataset import DatasetArtifact
from pgopinf.io.results_path import ResultsPath
from pgopinf.reduction.reducer import Reducer, ReductionResult
from pgopinf.reduction.reducer import Reducer
from pgopinf.specs.base import stable_id
from pgopinf.core.full_basis_store import FullBasisStore
from pgopinf.specs.reduction.base import ReductionSpec


class ReductionStore:
    """Build reduced models and projected datasets.

    The store coordinates cached full bases and optional Q matrices used by
    test-basis construction. The reduced result itself is returned by the
    reducer and is identified by the training data and reduction spec.
    """

    def __init__(
        self,
        paths: ResultsPath,
        *,
        full_basis_store: FullBasisStore,
        q_matrix_store: QMatrixStore,
    ):
        """Initialize the reduction store.

        Parameters
        ----------
        paths : ResultsPath
            Result-directory manager.
        full_basis_store : FullBasisStore
            Store used to cache full bases before truncation.
        q_matrix_store : QMatrixStore
            Store used to cache Q matrices for test-basis construction.
        """
        self.paths = paths
        self.full_basis_store = full_basis_store
        self.q_matrix_store = q_matrix_store

    def compute_id(
        self,
        *,
        data_train_id: str,
        reduction_spec: ReductionSpec,
    ) -> str:
        """Compute the stable identifier for a reduction.

        Parameters
        ----------
        data_train_id : str
            Identifier of the training split used for reduction.
        reduction_spec : ReductionSpec
            Reduction specification.

        Returns
        -------
        str
            Stable reduction identifier.
        """
        key = (
            {
                "data_train_id": data_train_id,
                "reduction": reduction_spec.to_dict(),
            },
        )
        return stable_id(key)

    # def exists(self, reduction_id: str) -> bool:
    #     d = self.paths.reductions_dir(reduction_id)
    #     return (d / "spec.json").exists() and (d / "basis.npz").exists()

    def get_or_create(
        self,
        *,
        dataset_artifact: DatasetArtifact,
        system,
        reduction_spec: ReductionSpec,
    ) -> ReductionResult:
        """
        Returns (reduction_id, reduction_result).
        """

        # get/create full basis (expensive)
        full_basis_id, full_basis = self.full_basis_store.get_or_create(
            dataset_artifact=dataset_artifact,
            system=system,
            full_basis_spec=reduction_spec.full_basis,
        )

        reducer = Reducer(reduction_spec)

        test_basis_kwargs = None

        if hasattr(reducer.test_basis, "compute_Q") and hasattr(
            reducer.test_basis, "q_cache_options"
        ):
            """
            Cache data from VQTestBasis (especially the expensive Hamiltonian identification)
            """
            q_id, q_artifact = self.q_matrix_store.get_or_create(
                source=reducer.test_basis.source,
                data_train_id=dataset_artifact.train_id,
                reduction_spec=reduction_spec,
                options=reducer.test_basis.q_cache_options(),
                compute_fn=lambda: QMatrixArtifact(
                    Q=reducer.test_basis.compute_Q(
                        system=system,
                        data_train=dataset_artifact.data.TRAIN,
                        V=full_basis.truncate(reduction_spec.r).V,
                    ),
                    meta={
                        "source": reducer.test_basis.source,
                        "data_train_id": dataset_artifact.train_id,
                    },
                ),
            )
            test_basis_kwargs = {"Q": q_artifact.Q}

        result = reducer.reduce(
            dataset=dataset_artifact.data,
            system=system,
            full_basis=full_basis,
            test_basis_kwargs=test_basis_kwargs,
        )

        return result
