from __future__ import annotations

from dataclasses import dataclass
from typing import Any


def allow_charged_fragments(total_charge: int, total_radical_electrons: int) -> bool:
    """RDKit and xyz2mol can assign radicals to neutral open-shell inputs.

    Their radical-only mode cannot represent a nonzero net charge. Closed-shell
    and charged inputs use formal-charge assignment.
    """
    return total_charge != 0 or total_radical_electrons <= 0


@dataclass(frozen=True)
class MethodRunOutput:
    status: str
    error: str | None = None
    predicted_smiles: str | None = None
    rdkit_mol: Any | None = None
    equivalent: bool | None = None
    equivalence_method: str | None = None
    timing_ms_breakdown: dict[str, float] | None = None


@dataclass(frozen=True)
class BenchmarkMethod:
    method_id: str

    def run(self, case: dict[str, Any]) -> MethodRunOutput:
        del case
        return MethodRunOutput(
            status="skipped",
            error="not implemented",
            timing_ms_breakdown={"method_ms": 0.0},
        )
