from __future__ import annotations
import json
import logging
from pathlib import Path
import numpy as np
from zipfile import BadZipFile

from pgopinf.identification.identifier import IdentificationResult
from pgopinf.io.results_path import ResultsPath
from pgopinf.specs.base import stable_id
from pgopinf.data.generator import (
    generate_reduced_split_data,
    generate_split_data,
)
from pgopinf.data.dataset import DatasetArtifact, SplitDatasetArtifact
from pgopinf.reduction.reducer import ReductionResult
from pgopinf.data.reduced_data import ReducedData
from pgopinf.specs.data.base import DataSpec, SplitDataSpec
from pgopinf.specs.identification.base import IdentificationSpec
from pgopinf.specs.reduction.base import ReductionSpec
from pgopinf.specs.system.base import SystemSpec
from pgopinf.systems.lti_system import LTISystem
from pgopinf.systems.ph_system import PHSystem
from pgopinf.specs.base import dataclass_to_dict

logger = logging.getLogger(__name__)


def _spec_to_dict(spec):
    return spec.to_dict() if hasattr(spec, "to_dict") else dataclass_to_dict(spec)


def _optional_npz_value(z, key: str):
    if key not in z.files:
        return None
    value = z[key]
    if value.shape == () and value.dtype == object and value.item() is None:
        return None
    if value.shape == () and value.dtype.kind in {"U", "S", "O"}:
        return value.item()
    return value


class DataStore:
    """Store generated full-order train and test data splits."""

    def __init__(self, paths: ResultsPath, force_recompute: bool = False):
        """Initialize the data store.

        Parameters
        ----------
        paths : ResultsPath
            Result-directory manager used to locate dataset artifacts.
        force_recompute : bool, optional
            If ``True``, regenerate data even when cached splits exist.
        """
        self.paths = paths
        self.force_recompute = force_recompute

    def compute_split_id_full(
        self,
        *,
        system_spec: SystemSpec,
        data_spec: DataSpec,
        split: str,
    ) -> str:
        """Compute the cache identifier for the unreduced full split.

        Parameters
        ----------
        system_spec : SystemSpec
            Specification of the system used for data generation.
        data_spec : DataSpec
            Data-generation specification.
        split : str
            Split name, for example ``"train"`` or ``"test"``.

        Returns
        -------
        str
            Stable split identifier excluding ``reduce_sample_n``.
        """
        # without reduce_sample_n
        key = {
            "artifact": "data_split",
            "version": 1,
            "system": _spec_to_dict(system_spec),
            "data": data_spec.split_to_dict(split, exclude={"reduce_sample_n"}),
        }
        return stable_id(key)

    def compute_split_id_explicit(
        self,
        *,
        system_spec: SystemSpec,
        data_spec: DataSpec,
        split: str,
    ) -> str:
        """Compute the explicit identifier for a requested data split.

        Parameters
        ----------
        system_spec : SystemSpec
            Specification of the system used for data generation.
        data_spec : DataSpec
            Data-generation specification.
        split : str
            Split name, for example ``"train"`` or ``"test"``.

        Returns
        -------
        str
            Stable split identifier including ``reduce_sample_n``.
        """
        # with reduce_sample_n
        raw_key = {
            "artifact": "data_split",
            "version": 1,
            "system": _spec_to_dict(system_spec),
            "data": data_spec.split_to_dict(split),
        }
        return stable_id(raw_key)

    def split_exists(self, split_id: str) -> bool:
        """Return whether a full-order split exists on disk.

        Parameters
        ----------
        split_id : str
            Stored split identifier.

        Returns
        -------
        bool
            ``True`` if metadata and array files exist.
        """
        d = self.paths.dataset_dir(split_id)
        return (d / "meta.json").exists() and (d / "data.npz").exists()

    def get_or_create_split(
        self,
        *,
        system_spec: SystemSpec,
        system: LTISystem | PHSystem,
        data_spec: DataSpec,
        split: str,
    ) -> SplitDatasetArtifact:
        """Load or generate one full-order data split.

        Parameters
        ----------
        system_spec : SystemSpec
            Specification of the system used for data generation.
        system : LTISystem or PHSystem
            System instance to simulate.
        data_spec : DataSpec
            Data-generation specification.
        split : str
            Split name, for example ``"train"`` or ``"test"``.

        Returns
        -------
        SplitDatasetArtifact
            Split identifier, split name, and generated data.
        """
        split_l = split.lower()
        split_id_full = self.compute_split_id_full(
            system_spec=system_spec,
            data_spec=data_spec,
            split=split_l,
        )
        split_id_explicit = self.compute_split_id_explicit(
            system_spec=system_spec,
            data_spec=data_spec,
            split=split_l,
        )
        split_spec = data_spec.get_split_spec(split_l)

        if self.split_exists(split_id_full) and not self.force_recompute:
            logger.info(
                f"Dataset split {split_l} already exists (id={split_id_full}), loading..."
            )
            return SplitDatasetArtifact(
                id=split_id_explicit,
                split=split_l,
                data=self.load_split(split_id_full, split_spec),
            )

        logger.info(f"Generating dataset split {split_l} (id={split_id_full})...")
        seed = int(data_spec.seed) if split_l == "train" else int(data_spec.seed) + 1

        data = generate_split_data(
            system=system,
            split_spec=split_spec,
            seed=seed,
            split_name=split_l,
        )

        self.save_split(
            split_id_full,
            system_spec=system_spec,
            split_spec=split_spec,
            split=split_l,
            seed=seed,
            data=data,
        )

        # reduce samples if requested
        if split_spec.reduce_sample_n is not None:
            data.reduce_samples(n=split_spec.reduce_sample_n)

        return SplitDatasetArtifact(id=split_id_explicit, split=split_l, data=data)

    def get_or_create(
        self,
        *,
        system_spec: SystemSpec,
        system: LTISystem | PHSystem,
        data_spec: DataSpec,
    ) -> DatasetArtifact:
        """Load or generate the train and test full-order data splits.

        Parameters
        ----------
        system_spec : SystemSpec
            Specification of the system used for data generation.
        system : LTISystem or PHSystem
            System instance to simulate.
        data_spec : DataSpec
            Data-generation specification.

        Returns
        -------
        DatasetArtifact
            Dataset artifact containing train and test splits.
        """
        train = self.get_or_create_split(
            system_spec=system_spec,
            system=system,
            data_spec=data_spec,
            split="train",
        )
        test = self.get_or_create_split(
            system_spec=system_spec,
            system=system,
            data_spec=data_spec,
            split="test",
        )
        return DatasetArtifact(train=train, test=test)

    def save_split(
        self,
        split_id: str,
        *,
        system_spec,
        split_spec,
        split: str,
        seed: int,
        data,
    ) -> None:
        """Persist one full-order data split.

        Parameters
        ----------
        split_id : str
            Identifier under which the split is stored.
        system_spec
            Specification of the simulated system.
        split_spec
            Split-specific data-generation specification.
        split : str
            Split name.
        seed : int
            Random seed used for generation.
        data
            Data object to serialize.
        """
        d = self.paths.dataset_dir(split_id)

        (d / "spec.json").write_text(
            json.dumps(
                {
                    "system": _spec_to_dict(system_spec),
                    "split": split,
                    "split_spec": split_spec.to_dict(),
                    "seed": seed,
                    "materialization": {
                        "kind": "full",
                    },
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )

        meta = {
            "dataset_id": split_id,
            "split": split,
            "artifact_format_version": 1,
        }
        (d / "meta.json").write_text(
            json.dumps(meta, indent=2, sort_keys=True),
            encoding="utf-8",
        )

        self._save_split_arrays(d / "data.npz", data)

    def _save_split_arrays(self, path: Path, data):
        np.savez_compressed(
            path,
            t=np.asarray(data.time.t),
            X=np.asarray(data.X),
            U=np.asarray(data.U) if data.U is not None else None,
            Y=np.asarray(data.Y) if data.Y is not None else None,
            dXdt=np.asarray(data.dXdt) if data.dXdt is not None else None,
            deriv_method=data.deriv_method if data.deriv_method is not None else None,
        )

    def load_split(self, split_id: str, split_spec: SplitDataSpec):
        """Load one full-order data split.

        Parameters
        ----------
        split_id : str
            Identifier of the stored split.
        split_spec : SplitDataSpec
            Split specification, used to apply optional sample reduction after
            loading.

        Returns
        -------
        Data
            Loaded full-order data.
        """
        from pgopinf.data.data import Data
        from pgopinf.data.time.time import Time

        d = self.paths.dataset_dir(split_id)
        z = np.load(d / "data.npz", allow_pickle=True)

        t = z["t"]
        X = z["X"]
        U = _optional_npz_value(z, "U")
        Y = _optional_npz_value(z, "Y")
        dXdt = _optional_npz_value(z, "dXdt")
        deriv_method = _optional_npz_value(z, "deriv_method")

        data = Data(
            time=Time(t),
            X=X,
            U=U,
            Y=Y,
            dXdt=dXdt,
            deriv_method=deriv_method,
        )
        # reduce samples afterwards
        if split_spec.reduce_sample_n is not None:
            data.reduce_samples(n=split_spec.reduce_sample_n)

        return data


class IntrusiveDataStore:
    """Store reduced data generated by simulating the intrusive reduced system."""

    def __init__(self, paths: ResultsPath, force_recompute: bool = False):
        """Initialize the intrusive data store.

        Parameters
        ----------
        paths : ResultsPath
            Result-directory manager used to locate intrusive dataset artifacts.
        force_recompute : bool, optional
            If ``True``, regenerate data even when cached splits exist.
        """
        self.paths = paths
        self.force_recompute = force_recompute

    def compute_split_id_full(
        self,
        *,
        system_spec,
        data_spec: DataSpec,
        reduction_spec,
        split: str,
    ) -> str:
        """Compute the cache identifier for an intrusive reduced split.

        Parameters
        ----------
        system_spec
            Specification of the original system.
        data_spec : DataSpec
            Data-generation specification.
        reduction_spec
            Reduction specification used to form the intrusive system.
        split : str
            Split name, for example ``"train"`` or ``"test"``.

        Returns
        -------
        str
            Stable split identifier excluding ``reduce_sample_n``.
        """
        # without reduce_sample_n
        key = {
            "artifact": "intrusive_data_split",
            "version": 1,
            "system": _spec_to_dict(system_spec),
            "data": data_spec.split_to_dict(split, exclude={"reduce_sample_n"}),
            "reduction": reduction_spec.to_dict(),
        }
        return stable_id(key)

    def compute_split_id_explicit(
        self,
        *,
        reduction_id: str,
        data_spec: DataSpec,
        split: str,
    ) -> str:
        """Compute the explicit identifier for an intrusive reduced split.

        Parameters
        ----------
        reduction_id : str
            Identifier of the reduced system. From ReductionStore.compute_id
        data_spec : DataSpec
            Data-generation specification.
        reduction_spec
            Reduction specification used to form the intrusive system.
        split : str
            Split name, for example ``"train"`` or ``"test"``.

        Returns
        -------
        str
            Stable split identifier including ``reduce_sample_n``.
        """
        # with reduce_sample_n
        key = {
            "artifact": "intrusive_data_split",
            "version": 1,
            "reduction_id": reduction_id,
            "data": data_spec.split_to_dict(split),
        }
        return stable_id(key)

    def split_exists(self, split_id: str) -> bool:
        """Return whether an intrusive reduced split exists on disk.

        Parameters
        ----------
        split_id : str
            Stored split identifier.

        Returns
        -------
        bool
            ``True`` if metadata and array files exist.
        """
        d = self.paths.intrusive_dataset_dir(split_id)
        return (d / "meta.json").exists() and (d / "data.npz").exists()

    def get_or_create_split(
        self,
        *,
        reduction_id: str,
        system_spec,
        data_spec: DataSpec,
        reduction_spec,
        reduction_result: ReductionResult,
        split: str,
    ) -> SplitDatasetArtifact:
        """Load or generate one intrusive reduced data split.

        Parameters
        ----------
        system_spec
            Specification of the original system.
        data_spec : DataSpec
            Data-generation specification.
        reduction_spec
            Reduction specification used to form the intrusive system.
        reduction_result : ReductionResult
            Reduced system and basis used for simulation.
        split : str
            Split name, for example ``"train"`` or ``"test"``.

        Returns
        -------
        SplitDatasetArtifact
            Split identifier, split name, and reduced data.
        """
        split_l = split.lower()
        split_id_explicit = self.compute_split_id_explicit(
            reduction_id=reduction_id,
            data_spec=data_spec,
            split=split_l,
        )

        split_spec = data_spec.get_split_spec(split_l)
        if self.split_exists(split_id_explicit) and not self.force_recompute:
            logger.info(
                f"Intrusive dataset split {split_l} already exists (id={split_id_explicit}), loading..."
            )
            return SplitDatasetArtifact(
                id=split_id_explicit,
                split=split_l,
                data=self.load_split(split_id_explicit, split_spec=split_spec),
            )

        logger.info(
            f"Generating intrusive dataset split {split_l} (id={split_id_explicit})..."
        )
        seed = int(data_spec.seed) if split_l == "train" else int(data_spec.seed) + 1

        data = generate_reduced_split_data(
            reduced_system=reduction_result.reduced_system,
            split_spec=split_spec,
            seed=seed,
            split_name=split_l,
            mor=reduction_result.mor,
            basis=reduction_result.basis,
        )

        self.save_split(
            split_id_explicit,
            system_spec=system_spec,
            split_spec=split_spec,
            reduction_spec=reduction_spec,
            split=split_l,
            seed=seed,
            data=data,
        )

        if split_spec.reduce_sample_n is not None:
            data.reduce_samples(n=split_spec.reduce_sample_n)

        return SplitDatasetArtifact(id=split_id_explicit, split=split_l, data=data)

    def get_or_create(
        self,
        *,
        reduction_id: str,
        system_spec,
        data_spec,
        reduction_spec,
        reduction_result: ReductionResult,
    ) -> DatasetArtifact:
        """Load or generate train and test intrusive reduced data splits.

        Parameters
        ----------
        system_spec
            Specification of the original system.
        data_spec
            Data-generation specification.
        reduction_spec
            Reduction specification used to form the intrusive system.
        reduction_result : ReductionResult
            Reduced system and basis used for simulation.

        Returns
        -------
        DatasetArtifact
            Dataset artifact containing train and test intrusive splits.
        """
        train = self.get_or_create_split(
            reduction_id=reduction_id,
            system_spec=system_spec,
            data_spec=data_spec,
            reduction_spec=reduction_spec,
            reduction_result=reduction_result,
            split="train",
        )
        test = self.get_or_create_split(
            reduction_id=reduction_id,
            system_spec=system_spec,
            data_spec=data_spec,
            reduction_spec=reduction_spec,
            reduction_result=reduction_result,
            split="test",
        )
        return DatasetArtifact(train=train, test=test)

    def save_split(
        self,
        split_id: str,
        *,
        system_spec,
        split_spec,
        reduction_spec,
        split: str,
        seed: int,
        data,
    ) -> None:
        """Persist one intrusive reduced data split.

        Parameters
        ----------
        split_id : str
            Identifier under which the split is stored.
        system_spec
            Specification of the original system.
        split_spec
            Split-specific data-generation specification.
        reduction_spec
            Reduction specification used to form the intrusive system.
        split : str
            Split name.
        seed : int
            Random seed used for generation.
        data
            Reduced data object to serialize.
        """
        d = self.paths.intrusive_dataset_dir(split_id)

        (d / "spec.json").write_text(
            json.dumps(
                {
                    "system": _spec_to_dict(system_spec),
                    "split": split,
                    "split_spec": split_spec.to_dict(),
                    "reduction": reduction_spec.to_dict(),
                    "seed": seed,
                    "materialization": {
                        "kind": "full",
                    },
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )

        meta = {
            "dataset_id": split_id,
            "split": split,
            "artifact_format_version": 1,
        }
        (d / "meta.json").write_text(
            json.dumps(meta, indent=2, sort_keys=True),
            encoding="utf-8",
        )

        self._save_split_arrays(d / "data.npz", data)

    def _save_split_arrays(self, path: Path, data):
        np.savez_compressed(
            path,
            t=np.asarray(data.time.t),
            X=np.asarray(data.X),
            U=np.asarray(data.U) if data.U is not None else None,
            Y=np.asarray(data.Y) if data.Y is not None else None,
            dXdt=np.asarray(data.dXdt) if data.dXdt is not None else None,
            deriv_method=data.deriv_method if data.deriv_method is not None else None,
            V=np.asarray(data.V) if getattr(data, "V", None) is not None else None,
        )

    def load_split(self, split_id: str, split_spec: SplitDataSpec):
        """Load one intrusive reduced data split.

        Parameters
        ----------
        split_id : str
            Identifier of the stored split.
        split_spec : SplitDataSpec
            Split specification, used to apply optional sample reduction after
            loading.

        Returns
        -------
        ReducedData
            Loaded reduced data.
        """
        from pgopinf.data.reduced_data import ReducedData
        from pgopinf.data.time.time import Time

        d = self.paths.intrusive_dataset_dir(split_id)
        z = np.load(d / "data.npz", allow_pickle=True)

        t = z["t"]
        X = z["X"]
        U = _optional_npz_value(z, "U")
        Y = _optional_npz_value(z, "Y")
        dXdt = _optional_npz_value(z, "dXdt")
        deriv_method = _optional_npz_value(z, "deriv_method")
        V = _optional_npz_value(z, "V")

        reduced_data = ReducedData(
            time=Time(t),
            X=X,
            U=U,
            Y=Y,
            dXdt=dXdt,
            deriv_method=deriv_method,
            V=V,
        )
        # reduce samples afterwards if requested
        if split_spec.reduce_sample_n is not None:
            reduced_data.reduce_samples(n=split_spec.reduce_sample_n)

        return reduced_data


class IdentifiedDataStore:
    """Store reduced data generated by simulating an identified system."""

    def __init__(self, paths: ResultsPath, force_recompute: bool = False):
        """Initialize the identified data store.

        Parameters
        ----------
        paths : ResultsPath
            Result-directory manager used to locate identified dataset artifacts.
        force_recompute : bool, optional
            If ``True``, regenerate data even when cached splits exist.
        """
        self.paths = paths
        self.force_recompute = force_recompute

    def compute_split_id_explicit(
        self,
        *,
        reduction_id: str,
        data_spec: DataSpec,
        identification_spec: IdentificationSpec,
        split: str,
    ) -> str:
        """Compute the explicit identifier for an identified-system data split.

        Parameters
        ----------
        reduction_id : str
            ID of the reduced system.
        data_spec : DataSpec
            Data-generation specification.
        identification_spec : IdentificationSpec
            Identification specification used to obtain the system.
        split : str
            Split name, for example ``"train"`` or ``"test"``.

        Returns
        -------
        str
            Stable split identifier including ``reduce_sample_n``.
        """
        # with reduce_sample_n
        key = {
            "artifact": "identified_data_split",
            "version": 1,
            "reduction_id": reduction_id,
            "data": data_spec.split_to_dict(split),
            "identification": identification_spec.to_dict(),
        }
        return stable_id(key)

    def split_exists(self, split_id: str) -> bool:
        """Return whether an identified-system split exists on disk.

        Parameters
        ----------
        split_id : str
            Stored split identifier.

        Returns
        -------
        bool
            ``True`` if metadata and array files exist.
        """
        d = self.paths.identified_dataset_dir(split_id)
        return (d / "meta.json").exists() and (d / "data.npz").exists()

    def get_or_create_split(
        self,
        *,
        reduction_id: str,
        system_spec: SystemSpec,
        data_spec: DataSpec,
        reduction_spec: ReductionSpec,
        identification_spec: IdentificationSpec,
        identification_result: IdentificationResult,
        reduction_result: ReductionResult,
        split: str,
    ) -> SplitDatasetArtifact:
        """Load or generate one data split from an identified system.

        Parameters
        ----------
        system_spec : SystemSpec
            Specification of the original system.
        data_spec : DataSpec
            Data-generation specification.
        reduction_spec : ReductionSpec
            Reduction specification associated with the identified system.
        identification_spec : IdentificationSpec
            Identification specification used to obtain the system.
        identification_result : IdentificationResult
            Identified system used for simulation.
        reduction_result : ReductionResult
            Reduction result that supplies the basis and MOR metadata.
        split : str
            Split name, for example ``"train"`` or ``"test"``.

        Returns
        -------
        SplitDatasetArtifact
            Split identifier, split name, and reduced data.
        """
        split_l = split.lower()
        split_id_explicit = self.compute_split_id_explicit(
            reduction_id=reduction_id,
            data_spec=data_spec,
            identification_spec=identification_spec,
            split=split_l,
        )

        split_spec = data_spec.get_split_spec(split_l)
        if self.split_exists(split_id_explicit) and not self.force_recompute:
            logger.info(
                f"Identified dataset split {split_l} already exists (id={split_id_explicit}), loading..."
            )

            return SplitDatasetArtifact(
                id=split_id_explicit,
                split=split_l,
                data=self.load_split(split_id_explicit, split_spec=split_spec),
            )

        logger.info(
            f"Generating identified dataset split {split_l} (id={split_id_explicit})..."
        )
        seed = int(data_spec.seed) if split_l == "train" else int(data_spec.seed) + 1

        data = generate_reduced_split_data(
            reduced_system=identification_result.system_identified,
            split_spec=split_spec,
            seed=seed,
            split_name=split_l,
            mor=reduction_result.mor,
            basis=reduction_result.basis,
        )

        self.save_split(
            split_id_explicit,
            system_spec=system_spec,
            split_spec=split_spec,
            reduction_spec=reduction_spec,
            identification_spec=identification_spec,
            split=split_l,
            seed=seed,
            data=data,
        )

        if split_spec.reduce_sample_n is not None:
            data.reduce_samples(n=split_spec.reduce_sample_n)

        return SplitDatasetArtifact(id=split_id_explicit, split=split_l, data=data)

    def get_or_create(
        self,
        *,
        reduction_id: str,
        system_spec: SystemSpec,
        data_spec: DataSpec,
        reduction_spec: ReductionSpec,
        identification_spec: IdentificationSpec,
        identification_result: IdentificationResult,
        reduction_result: ReductionResult,
    ) -> DatasetArtifact:
        """Load or generate train and test splits from an identified system.

        Parameters
        ----------
        system_spec : SystemSpec
            Specification of the original system.
        data_spec : DataSpec
            Data-generation specification.
        reduction_spec : ReductionSpec
            Reduction specification associated with the identified system.
        identification_spec : IdentificationSpec
            Identification specification used to obtain the system.
        identification_result : IdentificationResult
            Identified system used for simulation.
        reduction_result : ReductionResult
            Reduction result that supplies the basis and MOR metadata.

        Returns
        -------
        DatasetArtifact
            Dataset artifact containing train and test identified-system splits.
        """
        train = self.get_or_create_split(
            reduction_id=reduction_id,
            system_spec=system_spec,
            data_spec=data_spec,
            reduction_spec=reduction_spec,
            identification_spec=identification_spec,
            identification_result=identification_result,
            reduction_result=reduction_result,
            split="train",
        )
        test = self.get_or_create_split(
            reduction_id=reduction_id,
            system_spec=system_spec,
            data_spec=data_spec,
            reduction_spec=reduction_spec,
            identification_spec=identification_spec,
            identification_result=identification_result,
            reduction_result=reduction_result,
            split="test",
        )
        return DatasetArtifact(train=train, test=test)

    def save_split(
        self,
        split_id: str,
        *,
        system_spec,
        split_spec,
        reduction_spec,
        identification_spec,
        split: str,
        seed: int,
        data,
    ) -> None:
        """Persist one identified-system data split.

        Parameters
        ----------
        split_id : str
            Identifier under which the split is stored.
        system_spec
            Specification of the original system.
        split_spec
            Split-specific data-generation specification.
        reduction_spec
            Reduction specification associated with the identified system.
        identification_spec
            Identification specification used to obtain the system.
        split : str
            Split name.
        seed : int
            Random seed used for generation.
        data
            Reduced data object to serialize.
        """
        d = self.paths.identified_dataset_dir(split_id)

        (d / "spec.json").write_text(
            json.dumps(
                {
                    "system": _spec_to_dict(system_spec),
                    "split": split,
                    "split_spec": split_spec.to_dict(),
                    "reduction": reduction_spec.to_dict(),
                    "identification": identification_spec.to_dict(),
                    "seed": seed,
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )

        meta = {
            "dataset_id": split_id,
            "split": split,
            "artifact_format_version": 1,
        }
        (d / "meta.json").write_text(
            json.dumps(meta, indent=2, sort_keys=True),
            encoding="utf-8",
        )

        self._save_split_arrays(d / "data.npz", data)

    def _save_split_arrays(self, path: Path, data):
        np.savez_compressed(
            path,
            t=np.asarray(data.time.t),
            X=np.asarray(data.X),
            U=np.asarray(data.U) if data.U is not None else None,
            Y=np.asarray(data.Y) if data.Y is not None else None,
            dXdt=np.asarray(data.dXdt) if data.dXdt is not None else None,
            deriv_method=data.deriv_method if data.deriv_method is not None else None,
            V=np.asarray(data.V) if getattr(data, "V", None) is not None else None,
        )

    def load_split(self, split_id: str, split_spec: SplitDataSpec):
        """Load one identified-system data split.

        Parameters
        ----------
        split_id : str
            Identifier of the stored split.
        split_spec : SplitDataSpec
            Split specification, used to apply optional sample reduction after
            loading.

        Returns
        -------
        ReducedData
            Loaded reduced data generated from the identified system.
        """
        from pgopinf.data.reduced_data import ReducedData
        from pgopinf.data.time.time import Time

        d = self.paths.identified_dataset_dir(split_id)
        try:
            z = np.load(d / "data.npz", allow_pickle=True)
        except BadZipFile:
            # wait for 10s and repeat - this can happen if another process is still writing the file
            import time

            time.sleep(10)
            z = np.load(d / "data.npz", allow_pickle=True)

        t = z["t"]
        X = z["X"]
        U = _optional_npz_value(z, "U")
        Y = _optional_npz_value(z, "Y")
        dXdt = _optional_npz_value(z, "dXdt")
        deriv_method = _optional_npz_value(z, "deriv_method")
        V = _optional_npz_value(z, "V")

        reduced_data = ReducedData(
            time=Time(t),
            X=X,
            U=U,
            Y=Y,
            dXdt=dXdt,
            deriv_method=deriv_method,
            V=V,
        )
        # reduce samples afterwards if requested
        if split_spec.reduce_sample_n is not None:
            reduced_data.reduce_samples(n=split_spec.reduce_sample_n)

        return reduced_data
