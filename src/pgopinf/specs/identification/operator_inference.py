from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Literal, Mapping

from pgopinf.identification.identifier import (
    OperatorInferenceIdentifier,
)
from pgopinf.specs.base import SpecBase
from pgopinf.specs.identification.base import register_identification


@register_identification("opinf")
@dataclass(frozen=True)
class OperatorInferenceSpec(SpecBase):
    """Specification for operator inference."""

    kind: Literal["opinf"] = "opinf"
    use_E: bool = True
    convert_to_ph: bool = True
    seperate_output_inf: bool = False
    lambda_reg: float = 0.0

    def build(self) -> OperatorInferenceIdentifier:
        """Build the operator-inference identifier."""
        from pgopinf.identification.identifier import (
            OperatorInferenceIdentifier,
        )

        return OperatorInferenceIdentifier(
            use_E=self.use_E,
            convert_to_ph=self.convert_to_ph,
            seperate_output_inf=self.seperate_output_inf,
            lambda_reg=self.lambda_reg,
        )

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "OperatorInferenceSpec":
        """Deserialize an operator-inference spec."""
        return cls(
            use_E=bool(d.get("use_E", True)),
            convert_to_ph=bool(d.get("convert_to_ph", True)),
            seperate_output_inf=bool(d.get("seperate_output_inf", False)),
            lambda_reg=float(d.get("lambda_reg", 0.0)),
        )
