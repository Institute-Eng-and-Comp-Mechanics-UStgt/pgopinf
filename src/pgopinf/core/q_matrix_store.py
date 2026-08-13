from __future__ import annotations

import json
from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Any, Callable

import numpy as np

# adapt import if needed
from pgopinf.io.results_path import ResultsPath
from pgopinf.specs.base import stable_id
from pgopinf.specs.reduction.base import ReductionSpec

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class QMatrixArtifact:
    """Cached Q-matrix artifact.

    Attributes
    ----------
    Q : ndarray
        Matrix used by a test-basis construction.
    meta : dict of str to Any
        JSON-serializable metadata describing how the matrix was computed.
    """

    Q: np.ndarray
    meta: dict[str, Any]


class QMatrixStore:
    """
    Cache for expensive Q matrices used by test-basis construction.

    Typical usage
    -------------
    q_id, q_art = q_store.get_or_create(
        source=test_basis.source,
        dataset_id=dataset_id_if_needed,
        options=test_basis.q_cache_options(),
        compute_fn=lambda: QMatrixArtifact(
            Q=test_basis.compute_Q(system=system, data_train=dataset.TRAIN, V=full_basis.truncate(r)),
            meta={...},
        ),
    )
    """

    def __init__(self, paths: ResultsPath, force_recompute: bool = False):
        """Initialize the Q-matrix store.

        Parameters
        ----------
        paths : ResultsPath
            Result-directory manager that provides the Q-matrix cache root.
        force_recompute : bool, optional
            If ``True``, recompute matrices even when cached files exist.
        """
        self.root = paths.inferred_q_matrix
        self.force_recompute = force_recompute

    def compute_id(
        self,
        *,
        source: str,
        data_train_id: str | None,
        options: dict[str, Any],
    ) -> str:
        """Compute the stable identifier for a Q-matrix artifact.

        Parameters
        ----------
        source : str
            Name of the Q-matrix source or construction strategy.
        data_train_id : str or None
            Training-data identifier associated with the matrix.
        options : dict of str to Any
            JSON-serializable cache options that affect the matrix.

        Returns
        -------
        str
            Stable Q-matrix identifier.
        """
        key = {
            "source": source,
            "data_train_id": data_train_id,
            "options": options,
        }
        return stable_id(key)

    def exists(self, q_id: str) -> bool:
        """Return whether a Q-matrix artifact exists.

        Parameters
        ----------
        q_id : str
            Q-matrix identifier.

        Returns
        -------
        bool
            ``True`` if both matrix and metadata files are present.
        """
        d = self._dir(q_id)
        return (d / "Q.npz").exists() and (d / "meta.json").exists()

    def save(self, q_id: str, artifact: QMatrixArtifact) -> None:
        """Persist a Q-matrix artifact.

        Parameters
        ----------
        q_id : str
            Q-matrix identifier.
        artifact : QMatrixArtifact
            Matrix and metadata to store.
        """
        d = self._dir(q_id)
        d.mkdir(parents=True, exist_ok=True)

        np.savez_compressed(d / "Q.npz", Q=np.asarray(artifact.Q))
        (d / "meta.json").write_text(
            json.dumps(artifact.meta, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def load(self, q_id: str) -> QMatrixArtifact:
        """Load a Q-matrix artifact from disk.

        Parameters
        ----------
        q_id : str
            Q-matrix identifier.

        Returns
        -------
        QMatrixArtifact
            Stored Q matrix and metadata.

        Raises
        ------
        FileNotFoundError
            If the artifact files are incomplete or missing.
        """
        d = self._dir(q_id)
        if not self.exists(q_id):
            raise FileNotFoundError(f"QMatrix artifact {q_id} not found in {d}")

        z = np.load(d / "Q.npz")
        meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        return QMatrixArtifact(Q=np.asarray(z["Q"]), meta=meta)

    def get_or_create(
        self,
        *,
        source: str,
        data_train_id: str | None,
        options: dict[str, Any],
        reduction_spec: ReductionSpec | None = None,
        compute_fn: Callable[[], QMatrixArtifact],
    ) -> tuple[str, QMatrixArtifact]:
        """Load a cached Q matrix or compute and store a new one.

        Parameters
        ----------
        source : str
            Name of the Q-matrix source or construction strategy.
        data_train_id : str or None
            Training-data identifier associated with the matrix.
        options : dict of str to Any
            Cache options that affect matrix construction.
        reduction_spec : ReductionSpec, optional
            Reduction specification included in the cache key for sources that
            depend on the reduced model.
        compute_fn : callable
            Zero-argument function that computes the artifact on cache miss.

        Returns
        -------
        tuple[str, QMatrixArtifact]
            Q-matrix identifier and artifact.

        Raises
        ------
        ValueError
            If ``source`` requires ``reduction_spec`` and none is provided.
        """
        if source == "Hamiltonian_red":
            if reduction_spec is None:
                raise ValueError(
                    "reduction_spec is required for source='Hamiltonian_red'"
                )
            q_id = self.compute_id(
                source=source,
                data_train_id=data_train_id,
                options={**options, "reduction_spec": reduction_spec.to_dict()},
            )
        else:
            q_id = self.compute_id(
                source=source,
                data_train_id=data_train_id,
                options=options,
            )

        if self.exists(q_id) and not self.force_recompute:
            logger.info(
                f"Q matrix artifact {q_id} already exists. Loading from disk..."
            )
            return q_id, self.load(q_id)

        logger.info(f"Computing Q matrix artifact {q_id}...")
        artifact = compute_fn()
        self.save(q_id, artifact)
        return q_id, artifact

    def _dir(self, q_id: str) -> Path:
        return self.root / f"q_{q_id}"
