"""Structured diagnostics for XYZ graph reconstruction failures."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, MutableMapping


class ReconstructionFailureCode(str, Enum):
    """Stable machine-readable codes for reconstruction failures."""

    INVALID_XYZ = "INVALID_XYZ"
    INVALID_ELECTRONIC_TARGET = "INVALID_ELECTRONIC_TARGET"
    NO_REACHABLE_METAL_STATE = "NO_REACHABLE_METAL_STATE"
    NO_VALID_ORGANIC_CANDIDATE = "NO_VALID_ORGANIC_CANDIDATE"
    ALL_METAL_CANDIDATES_REJECTED = "ALL_METAL_CANDIDATES_REJECTED"
    OUTPUT_INVARIANT_BROKEN = "OUTPUT_INVARIANT_BROKEN"
    BACKEND_EXCEPTION = "BACKEND_EXCEPTION"
    RDKIT_POSTPROCESS_FAILED = "RDKIT_POSTPROCESS_FAILED"
    SMILES_ROUNDTRIP_FAILED = "SMILES_ROUNDTRIP_FAILED"
    NO_VALID_RECONSTRUCTION = "NO_VALID_RECONSTRUCTION"

    def __str__(self) -> str:
        return self.value


# Keep the existing public name while making it an enum namespace. This lets
# callers use ``RECONSTRUCTION_FAILURE_CODES.INVALID_XYZ`` and iterate over
# the complete set of supported codes.
RECONSTRUCTION_FAILURE_CODES = ReconstructionFailureCode


def _coerce_failure_code(
    value: ReconstructionFailureCode | str,
    fallback: ReconstructionFailureCode = ReconstructionFailureCode.NO_VALID_RECONSTRUCTION,
) -> tuple[ReconstructionFailureCode, str | None]:
    if isinstance(value, ReconstructionFailureCode):
        return value, None
    try:
        return ReconstructionFailureCode(str(value)), None
    except ValueError:
        return fallback, str(value)


@dataclass(frozen=True)
class ReconstructionDiagnostics:
    """Stable, serializable explanation of a failed reconstruction."""

    code: ReconstructionFailureCode
    stage: str
    backend: str
    message: str
    counts: Mapping[str, int] = field(default_factory=dict)
    details: Mapping[str, Any] = field(default_factory=dict)
    cause_type: str = ""
    cause_message: str = ""

    def __post_init__(self) -> None:
        code, unknown_code = _coerce_failure_code(self.code)
        object.__setattr__(self, "code", code)
        if unknown_code is not None:
            details = dict(self.details)
            details.setdefault("unknown_failure_code", unknown_code)
            object.__setattr__(self, "details", details)

    def as_dict(self) -> dict[str, Any]:
        return {
            "code": self.code.value,
            "stage": self.stage,
            "backend": self.backend,
            "message": self.message,
            "counts": dict(self.counts),
            "details": dict(self.details),
            "cause_type": self.cause_type,
            "cause_message": self.cause_message,
        }

    def as_json(self) -> str:
        return json.dumps(self.as_dict(), ensure_ascii=False, sort_keys=True)

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any] | None,
        *,
        backend: str,
        fallback_code: ReconstructionFailureCode
        | str = ReconstructionFailureCode.NO_VALID_RECONSTRUCTION,
    ) -> ReconstructionDiagnostics:
        fallback, _ = _coerce_failure_code(fallback_code)
        if not value:
            return cls(
                code=fallback,
                stage="reconstruction",
                backend=backend,
                message="The reconstruction backend returned no molecule or diagnostics.",
            )
        counts = value.get("counts", {})
        details = value.get("details", {})
        code_value = value.get("code") or fallback
        code, unknown_code = _coerce_failure_code(
            code_value
            if isinstance(code_value, (ReconstructionFailureCode, str))
            else str(code_value),
            fallback,
        )
        normalized_details = dict(details) if isinstance(details, Mapping) else {}
        if unknown_code is not None:
            normalized_details.setdefault("unknown_failure_code", unknown_code)
        return cls(
            code=code,
            stage=str(value.get("stage") or "reconstruction"),
            backend=str(value.get("backend") or backend),
            message=str(value.get("message") or "Reconstruction failed."),
            counts=dict(counts) if isinstance(counts, Mapping) else {},
            details=normalized_details,
            cause_type=str(value.get("cause_type") or ""),
            cause_message=str(value.get("cause_message") or ""),
        )


class ReconstructionError(ValueError):
    """ValueError subclass carrying a machine-readable reconstruction failure."""

    def __init__(self, diagnostics: ReconstructionDiagnostics) -> None:
        self.diagnostics = diagnostics
        super().__init__(self._format_message(diagnostics))

    @staticmethod
    def _format_message(diagnostics: ReconstructionDiagnostics) -> str:
        # Keep the historical token in the human-facing message for callers and
        # logs that still search for it, while exposing the actionable code/stage.
        prefix = (
            "xyz2omol failed"
            if diagnostics.code
            not in {
                ReconstructionFailureCode.INVALID_XYZ,
                ReconstructionFailureCode.INVALID_ELECTRONIC_TARGET,
            }
            else "Invalid reconstruction input"
        )
        message = (
            f"{prefix} [{diagnostics.code.value}] at {diagnostics.stage}: {diagnostics.message}"
        )
        if diagnostics.cause_type and diagnostics.cause_message:
            message += f" ({diagnostics.cause_type}: {diagnostics.cause_message})"
        return message


@dataclass
class ReconstructionDiagnosticCollector:
    """Small bounded accumulator used by the fallback pipeline."""

    backend: str = "python"
    counts: MutableMapping[str, int] = field(default_factory=dict)
    details: MutableMapping[str, Any] = field(default_factory=dict)
    _failure: ReconstructionDiagnostics | None = None

    def count(self, name: str, amount: int = 1) -> None:
        self.counts[name] = int(self.counts.get(name, 0)) + amount

    def set(self, name: str, value: Any) -> None:
        self.details[name] = value

    def fail(
        self,
        code: ReconstructionFailureCode | str,
        stage: str,
        message: str,
        *,
        cause: BaseException | None = None,
        **details: Any,
    ) -> ReconstructionDiagnostics:
        if details:
            self.details.update(details)
        normalized_code, unknown_code = _coerce_failure_code(code)
        if unknown_code is not None:
            self.details.setdefault("unknown_failure_code", unknown_code)
        failure = ReconstructionDiagnostics(
            code=normalized_code,
            stage=stage,
            backend=self.backend,
            message=message,
            counts=dict(self.counts),
            details=dict(self.details),
            cause_type=type(cause).__name__ if cause is not None else "",
            cause_message=str(cause) if cause is not None else "",
        )
        self._failure = failure
        return failure

    def finish(
        self,
        *,
        code: ReconstructionFailureCode | str,
        stage: str,
        message: str,
    ) -> ReconstructionDiagnostics:
        if self._failure is not None:
            return self._failure
        return self.fail(code, stage, message)

    @property
    def failure(self) -> ReconstructionDiagnostics | None:
        return self._failure


__all__ = [
    "RECONSTRUCTION_FAILURE_CODES",
    "ReconstructionFailureCode",
    "ReconstructionDiagnosticCollector",
    "ReconstructionDiagnostics",
    "ReconstructionError",
]
