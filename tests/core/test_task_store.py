from __future__ import annotations

import json
from dataclasses import dataclass

import pytest

from pgopinf.core.task_store import SystemAnalysisTaskStore
from pgopinf.io.results_path import ResultsPath
from pgopinf.systems.analysis.artifact import SystemTaskArtifact


@dataclass(frozen=True)
class FakeSystemSpec:
    kind: str = "fake_system"
    size: int = 2


@dataclass(frozen=True)
class FakeTaskSpec:
    key: str = "fake_result"
    value: object = 1
    kind: str = "fake_task"

    def to_dict(self):
        return {"kind": self.kind, "key": self.key, "value": str(self.value)}

    def result_key(self):
        return self.key

    def build(self):
        return FakeTask(value=self.value)


class FakeTask:
    calls = []

    def __init__(self, value):
        self.value = value

    def compute(self, *, system):
        FakeTask.calls.append(system)
        return self.value


class CustomTask(FakeTask):
    save_calls = []
    load_calls = []

    def save_result(self, *, task_dir, value):
        CustomTask.save_calls.append({"task_dir": task_dir, "value": value})
        (task_dir / "custom.txt").write_text(str(value), encoding="utf-8")

    def load_result(self, *, task_dir):
        CustomTask.load_calls.append(task_dir)
        return (task_dir / "custom.txt").read_text(encoding="utf-8")


@dataclass(frozen=True)
class CustomTaskSpec(FakeTaskSpec):
    kind: str = "custom_task"

    def build(self):
        return CustomTask(value=self.value)


def make_store(tmp_path, *, force_recompute: bool = False) -> SystemAnalysisTaskStore:
    return SystemAnalysisTaskStore(
        ResultsPath(tmp_path / "results"),
        force_recompute=force_recompute,
    )


def test_compute_id_is_stable_and_depends_on_inputs(tmp_path) -> None:
    store = make_store(tmp_path)
    system_spec = FakeSystemSpec(size=2)
    task_spec = FakeTaskSpec(key="eig", value=1)

    base_id = store.compute_id(system_spec=system_spec, task_spec=task_spec)

    assert base_id == store.compute_id(system_spec=system_spec, task_spec=task_spec)
    assert base_id != store.compute_id(
        system_spec=FakeSystemSpec(size=3),
        task_spec=task_spec,
    )
    assert base_id != store.compute_id(
        system_spec=system_spec,
        task_spec=FakeTaskSpec(key="other", value=1),
    )


def test_save_and_load_pickle_artifact_round_trip(tmp_path) -> None:
    store = make_store(tmp_path)
    system_spec = FakeSystemSpec()
    task_spec = FakeTaskSpec(key="value", value=5)
    task = task_spec.build()
    artifact = SystemTaskArtifact(
        task_key="value",
        task_kind="fake_task",
        value={"answer": 5},
        meta={"source": "test"},
    )

    store.save(system_spec, task_spec, task, artifact)

    assert store.exists(system_spec, task_spec)
    loaded = store.load(system_spec, task_spec, task)
    assert loaded == artifact

    meta = json.loads((store._dir(system_spec, task_spec) / "meta.json").read_text())
    assert meta == {
        "task_key": "value",
        "task_kind": "fake_task",
        "meta": {"source": "test"},
        "storage_mode": "pickle",
    }


def test_exists_rejects_incomplete_pickle_artifact(tmp_path) -> None:
    store = make_store(tmp_path)
    system_spec = FakeSystemSpec()
    task_spec = FakeTaskSpec()
    d = store._dir(system_spec, task_spec)
    d.mkdir(parents=True, exist_ok=True)
    (d / "meta.json").write_text(
        json.dumps(
            {
                "task_key": "fake_result",
                "task_kind": "fake_task",
                "meta": {},
                "storage_mode": "pickle",
            }
        ),
        encoding="utf-8",
    )

    assert store.exists(system_spec, task_spec) is False


def test_save_and_load_custom_artifact_round_trip(tmp_path) -> None:
    CustomTask.save_calls = []
    CustomTask.load_calls = []
    store = make_store(tmp_path)
    system_spec = FakeSystemSpec()
    task_spec = CustomTaskSpec(key="plot", value="figure-data")
    task = task_spec.build()
    artifact = SystemTaskArtifact(
        task_key="plot",
        task_kind="custom_task",
        value="figure-data",
        meta={"format": "custom"},
    )

    store.save(system_spec, task_spec, task, artifact)

    assert store.exists(system_spec, task_spec)
    loaded = store.load(system_spec, task_spec, task)
    assert loaded == SystemTaskArtifact(
        task_key="plot",
        task_kind="custom_task",
        value="figure-data",
        meta={"format": "custom"},
    )
    assert CustomTask.save_calls[0]["value"] == "figure-data"
    assert CustomTask.load_calls == [store._dir(system_spec, task_spec)]
    assert json.loads((store._dir(system_spec, task_spec) / "meta.json").read_text())[
        "storage_mode"
    ] == "custom"


def test_get_or_create_computes_saves_and_reuses_cached_artifact(tmp_path) -> None:
    FakeTask.calls = []
    store = make_store(tmp_path)
    system_spec = FakeSystemSpec()
    task_spec = FakeTaskSpec(key="metric", value=42)

    task_id, first = store.get_or_create(
        system_spec=system_spec,
        system="system",
        task_spec=task_spec,
    )
    second_id, second = store.get_or_create(
        system_spec=system_spec,
        system="system",
        task_spec=task_spec,
    )

    assert task_id == second_id
    assert FakeTask.calls == ["system"]
    assert first.task_key == "metric"
    assert first.task_kind == "fake_task"
    assert first.value == 42
    assert first.meta == {
        "system_spec": {"kind": "fake_system", "size": 2},
        "task_spec": task_spec.to_dict(),
    }
    assert second == first


def test_get_or_create_honors_force_recompute(tmp_path) -> None:
    FakeTask.calls = []
    store = make_store(tmp_path, force_recompute=True)
    system_spec = FakeSystemSpec()
    task_spec = FakeTaskSpec(key="metric", value=42)

    store.get_or_create(system_spec=system_spec, system="system", task_spec=task_spec)
    store.get_or_create(system_spec=system_spec, system="system", task_spec=task_spec)

    assert FakeTask.calls == ["system", "system"]


def test_load_missing_pickle_value_raises(tmp_path) -> None:
    store = make_store(tmp_path)
    system_spec = FakeSystemSpec()
    task_spec = FakeTaskSpec()
    d = store._dir(system_spec, task_spec)
    d.mkdir(parents=True, exist_ok=True)
    (d / "meta.json").write_text(
        json.dumps(
            {
                "task_key": "fake_result",
                "task_kind": "fake_task",
                "meta": {},
                "storage_mode": "pickle",
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(FileNotFoundError):
        store.load(system_spec, task_spec, task_spec.build())
