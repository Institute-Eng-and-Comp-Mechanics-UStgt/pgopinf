from dataclasses import dataclass, field

from typing import Literal, Any

from pgopinf.specs.data.initial_condition.base import (
    ICSpec,
    ZerosIC,
    ic_from_dict,
)
from pgopinf.specs.data.time.base import TimeSpec
from pgopinf.specs.data.inputs.base import (
    InputSpec,
    PresetInputSpec,
    input_from_dict,
)

dXdt_derivation_methods = Literal["return", "finite_difference", "model_derivative"]


@dataclass(frozen=True)
class SplitDataSpec:
    """Specification for one data split."""

    time: TimeSpec = TimeSpec()
    initial_condition: ICSpec = field(default_factory=ZerosIC)
    input: InputSpec = field(default_factory=PresetInputSpec)
    n_sim: int = 1
    dXdt_derivation: dXdt_derivation_methods = (
        "return"  # options: "return", "finite_difference", "model_derivative"
    )
    reduce_sample_n: int | None = (
        None  # only use this many samples (after simulation, before training)
    )

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "SplitDataSpec":
        """Deserialize a split data specification."""
        time = TimeSpec.from_dict(d.get("time", {}))
        initial_condition = ic_from_dict(d.get("initial_condition", {}))
        input = input_from_dict(d.get("input", {}))
        n_sim = int(d.get("n_sim", 1))
        dXdt_derivation = d.get("dXdt_derivation", "return")
        return cls(
            time=time,
            initial_condition=initial_condition,
            input=input,
            n_sim=n_sim,
            dXdt_derivation=dXdt_derivation,
            reduce_sample_n=d.get("reduce_sample_n", None),
        )

    def to_dict(self, exclude: set[str] = None) -> dict[str, Any]:
        """Serialize the split data specification."""
        dict_data_split = {
            "time": self.time.to_dict(),
            "initial_condition": self.initial_condition.to_dict(),
            "input": self.input.to_dict(),
            "n_sim": self.n_sim,
            "dXdt_derivation": self.dXdt_derivation,
            "reduce_sample_n": self.reduce_sample_n,
        }
        if exclude is not None:
            if exclude:
                for key in exclude:
                    del dict_data_split[key]
        return dict_data_split


@dataclass(frozen=True)
class DataSpec:
    """Specification for train and test data generation."""

    train: SplitDataSpec = SplitDataSpec()
    test: SplitDataSpec = SplitDataSpec()
    seed: int = 0

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "DataSpec":
        """Deserialize a data specification."""
        return cls(
            train=SplitDataSpec.from_dict(d.get("train", {})),
            test=SplitDataSpec.from_dict(d.get("test", {})),
            seed=int(d.get("seed", 0)),
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize the data specification."""
        return {
            "train": self.train.to_dict(),
            "test": self.test.to_dict(),
            "seed": self.seed,
        }

    def get_split_spec(self, split: str) -> SplitDataSpec:
        """Return the specification for one split."""
        split_l = split.lower()
        if split_l == "train":
            return self.train
        if split_l == "test":
            return self.test
        raise ValueError(f"Invalid split {split!r}. Must be 'train' or 'test'.")

    def split_to_dict(self, split: str, exclude: set[str] = None) -> dict[str, Any]:
        """Serialize one split together with its name and seed."""
        return {
            "split": split.lower(),
            "split_spec": self.get_split_spec(split).to_dict(exclude=exclude),
            "seed": self.seed,
        }
