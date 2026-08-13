from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass(frozen=True)
class OutputSpec:
    """Output switches for rendered evaluation artifacts."""

    # for series data, e.g. trajectories
    csv: bool = True
    png: bool = True
    pdf: bool = False
    # for scalar data, e.g. metric values
    append_csv: bool = True
    append_jsonl: bool = False
    log_to_terminal: bool = True


@dataclass(frozen=True)
class ReportingSpec:
    """
    default_outputs:
        default behavior for all artifacts

    save_summaries_csv:
        export the summary rows as a table
    """

    default_outputs: OutputSpec = field(default_factory=OutputSpec)
    save_summaries_csv: bool = True
