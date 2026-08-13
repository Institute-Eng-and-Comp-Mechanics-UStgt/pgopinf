from __future__ import annotations

import json
from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import pytest

from pgopinf.core.system_analysis_store import SystemAnalysisStore
from pgopinf.specs.system_analysis.system_analysis import (
    SystemAnalysisSpec,
)
from pgopinf.systems.analysis.artifact import SystemAnalysisBundle


@dataclass(frozen=True)
class FakeTaskSpec:
    key: str
    value: object
    kind: str = "fake_task"
    calls: ClassVar[list] = []

    def to_dict(self):
        return {"kind": self.kind, "key": self.key, "value": str(self.value)}

    def result_key(self):
        return self.key

    def build(self):
        key = self.key
        value = self.value
        calls = self.calls

        class Task:
            @property
            def task_kind(self):
                return "fake_task"

            def compute(self, *, system):
                FakeTaskSpec.calls.append({"key": key, "system": system})
                return value

        return Task()


def make_store(tmp_path) -> SystemAnalysisStore:
    return SystemAnalysisStore(tmp_path / "system-analysis")


def test_compute_id_is_stable_and_depends_on_inputs(tmp_path) -> None:
    store = make_store(tmp_path)
    spec = SystemAnalysisSpec(tasks=(FakeTaskSpec("a", 1),))

    base_id = store.compute_id(system_id="system-a", analysis_spec=spec)

    assert base_id == store.compute_id(system_id="system-a", analysis_spec=spec)
    assert base_id != store.compute_id(system_id="system-b", analysis_spec=spec)
    assert base_id != store.compute_id(
        system_id="system-a",
        analysis_spec=SystemAnalysisSpec(tasks=(FakeTaskSpec("b", 1),)),
    )


def test_save_and_load_bundle_round_trip(tmp_path) -> None:
    store = make_store(tmp_path)
    bundle = SystemAnalysisBundle(
        task_ids={"eig": "task-id"},
        values={"eig": np.array([1.0, 2.0])},
        meta={"system_id": "system-id"},
    )

    store.save("analysis-id", bundle)

    assert store.exists("analysis-id")
    loaded = store.load("analysis-id")
    assert loaded.task_ids == bundle.task_ids
    np.testing.assert_allclose(loaded.values["eig"], bundle.values["eig"])
    assert loaded.meta == bundle.meta
    assert json.loads((store._dir("analysis-id") / "meta.json").read_text()) == {
        "meta": {"system_id": "system-id"},
        "task_ids": {"eig": "task-id"},
    }


def test_exists_requires_meta_and_values_files(tmp_path) -> None:
    store = make_store(tmp_path)
    d = store._dir("partial")
    d.mkdir(parents=True, exist_ok=True)
    (d / "meta.json").write_text("{}", encoding="utf-8")

    assert store.exists("partial") is False


def test_get_or_create_computes_saves_and_reuses_cached_bundle(tmp_path) -> None:
    store = make_store(tmp_path)
    FakeTaskSpec.calls = []
    analysis_spec = SystemAnalysisSpec(
        tasks=(
            FakeTaskSpec("first", 1),
            FakeTaskSpec("second", {"value": 2}),
        )
    )

    analysis_id, first = store.get_or_create(
        system_id="system-id",
        system="system",
        analysis_spec=analysis_spec,
    )
    second_id, second = store.get_or_create(
        system_id="system-id",
        system="system",
        analysis_spec=analysis_spec,
    )

    assert analysis_id == second_id
    assert FakeTaskSpec.calls == [
        {"key": "first", "system": "system"},
        {"key": "second", "system": "system"},
    ]
    assert first.values == {"first": 1, "second": {"value": 2}}
    assert first.meta == {
        "system_id": "system-id",
        "analysis_spec": analysis_spec.to_dict(),
    }
    assert set(first.task_ids) == {"first", "second"}
    assert second.values == first.values
    assert second.meta == first.meta
    assert second.task_ids == first.task_ids


def test_get_or_create_rejects_duplicate_result_keys(tmp_path) -> None:
    store = make_store(tmp_path)
    analysis_spec = SystemAnalysisSpec(
        tasks=(
            FakeTaskSpec("duplicate", 1),
            FakeTaskSpec("duplicate", 2),
        )
    )

    with pytest.raises(ValueError, match="Duplicate system-analysis result key"):
        store.get_or_create(
            system_id="system-id",
            system="system",
            analysis_spec=analysis_spec,
        )
