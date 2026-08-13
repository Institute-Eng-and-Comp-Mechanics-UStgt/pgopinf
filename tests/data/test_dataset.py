from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from pgopinf.data.dataset import (
    DataSet,
    DatasetArtifact,
    SplitDatasetArtifact,
)


def test_dataset_stores_train_and_optional_test_splits() -> None:
    train = object()
    test = object()

    dataset = DataSet(train=train, test=test)

    assert dataset.TRAIN is train
    assert dataset.TEST is test


def test_dataset_allows_missing_test_split() -> None:
    train = object()

    dataset = DataSet(train=train)

    assert dataset.TRAIN is train
    assert dataset.TEST is None


def test_split_dataset_artifact_is_frozen_value_object() -> None:
    data = object()
    artifact = SplitDatasetArtifact(id="split-id", split="train", data=data)

    assert artifact.id == "split-id"
    assert artifact.split == "train"
    assert artifact.data is data

    with pytest.raises(FrozenInstanceError):
        artifact.id = "other"  # type: ignore[misc]


def test_dataset_artifact_exposes_ids_and_runtime_dataset() -> None:
    train_data = object()
    test_data = object()
    train = SplitDatasetArtifact(id="train-id", split="train", data=train_data)
    test = SplitDatasetArtifact(id="test-id", split="test", data=test_data)

    artifact = DatasetArtifact(train=train, test=test)
    dataset = artifact.data

    assert artifact.train_id == "train-id"
    assert artifact.test_id == "test-id"
    assert dataset.TRAIN is train_data
    assert dataset.TEST is test_data


def test_dataset_artifact_data_property_returns_fresh_dataset_each_time() -> None:
    train = SplitDatasetArtifact(id="train-id", split="train", data=object())
    test = SplitDatasetArtifact(id="test-id", split="test", data=object())
    artifact = DatasetArtifact(train=train, test=test)

    assert artifact.data is not artifact.data
