from __future__ import annotations

from dataclasses import dataclass
from typing import Union

from pgopinf.data.data import Data
from pgopinf.data.reduced_data import ReducedData

SplitData = Union[Data, ReducedData]


class DataSet:
    """Container for train and optional test data splits.

    Parameters
    ----------
    train : Data or ReducedData
        Training split.
    test : Data or ReducedData, optional
        Test split.
    """

    def __init__(
        self,
        train: SplitData,
        test: SplitData | None = None,
    ) -> None:
        """Initialize the dataset container.

        Parameters
        ----------
        train : Data or ReducedData
            Training split.
        test : Data or ReducedData, optional
            Test split.
        """
        self.TRAIN = train
        self.TEST = test


@dataclass(frozen=True)
class SplitDatasetArtifact:
    """Stored data split and its identifier.

    Attributes
    ----------
    id : str
        Stable split identifier.
    split : str
        Split name, for example ``"train"`` or ``"test"``.
    data : Data or ReducedData
        Split data.
    """

    id: str
    split: str  # "train" or "test"
    data: SplitData


@dataclass(frozen=True)
class DatasetArtifact:
    """Stored train/test dataset artifact.

    Attributes
    ----------
    train : SplitDatasetArtifact
        Training split artifact.
    test : SplitDatasetArtifact
        Test split artifact.
    """

    train: SplitDatasetArtifact
    test: SplitDatasetArtifact

    @property
    def train_id(self) -> str:
        """Stable identifier of the training split.

        Returns
        -------
        str
            Training split identifier.
        """
        return self.train.id

    @property
    def test_id(self) -> str:
        """Stable identifier of the test split.

        Returns
        -------
        str
            Test split identifier.
        """
        return self.test.id

    @property
    def data(self) -> DataSet:
        """Return the dataset as a runtime ``DataSet`` object.

        Returns
        -------
        DataSet
            Container with train and test data.
        """
        return DataSet(
            train=self.train.data,
            test=self.test.data,
        )
