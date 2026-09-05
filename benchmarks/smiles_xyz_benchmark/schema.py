from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal


ResultStatus = Literal["ok", "error", "skipped"]


@dataclass
class BenchmarkResult:
    case_idx: int
    method_id: str
    input_smiles: str
    ground_truth_smiles: str | None
    status: ResultStatus
    error: str | None
    predicted_smiles: str | None
    equivalent: bool | None
    equivalence_method: str | None
    timing_ms_total: float
    timing_ms_breakdown: dict[str, float]
    case_id: str | None = None
    comparison_skipped: bool = False
    comparison_skip_reason: str | None = None
    evaluator_decision: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
