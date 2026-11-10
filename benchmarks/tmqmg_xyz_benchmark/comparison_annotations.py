from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any


ANNOTATIONS_PATH = Path(__file__).with_name("comparison_annotations.json")
COMPARISON_SKIP_REASON_PREFIX = "tmQMg comparison excluded by project policy"


@dataclass(frozen=True)
class ComparisonAnnotation:
    group_id: str
    status: str
    reason: str
    skip_comparison: bool
    expected_case_count: int
    minimum_element_counts: tuple[tuple[str, int], ...]
    excluded_case_ids: frozenset[str]

    def matches(self, case_id: str, element_counts: Mapping[str, int]) -> bool:
        if not case_id or case_id in self.excluded_case_ids:
            return False
        return all(
            element_counts.get(symbol, 0) >= count for symbol, count in self.minimum_element_counts
        )

    @property
    def comparison_skip_reason(self) -> str:
        return f"{COMPARISON_SKIP_REASON_PREFIX}: {self.status}: {self.reason}"


def _required_string(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"comparison annotation {field} must be a non-empty string")
    return value.strip()


def _csv_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1"}:
            return True
        if normalized in {"false", "0"}:
            return False
    return None


@lru_cache(maxsize=1)
def load_comparison_annotations() -> tuple[ComparisonAnnotation, ...]:
    payload = json.loads(ANNOTATIONS_PATH.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported tmQMg comparison annotation schema_version")
    if payload.get("dataset") != "tmQMg":
        raise ValueError("tmQMg comparison annotations have the wrong dataset")
    _required_string(payload.get("dataset_revision"), field="dataset_revision")
    groups = payload.get("groups")
    if not isinstance(groups, list) or not groups:
        raise ValueError("tmQMg comparison annotations must contain at least one group")

    annotations: list[ComparisonAnnotation] = []
    group_ids: set[str] = set()
    for group in groups:
        if not isinstance(group, dict):
            raise ValueError("tmQMg comparison annotation group must be an object")
        group_id = _required_string(group.get("id"), field="id")
        if group_id in group_ids:
            raise ValueError(f"duplicate tmQMg comparison annotation id: {group_id}")
        group_ids.add(group_id)

        selection = group.get("selection")
        if not isinstance(selection, dict):
            raise ValueError(f"comparison annotation {group_id} selection must be an object")
        raw_counts = selection.get("minimum_element_counts")
        if not isinstance(raw_counts, dict) or not raw_counts:
            raise ValueError(
                f"comparison annotation {group_id} minimum_element_counts must be non-empty"
            )
        minimum_counts: list[tuple[str, int]] = []
        for symbol, count in raw_counts.items():
            if not isinstance(symbol, str) or not symbol or not isinstance(count, int) or count < 1:
                raise ValueError(
                    f"comparison annotation {group_id} has invalid minimum element count"
                )
            minimum_counts.append((symbol, count))

        raw_excluded_ids = selection.get("excluded_case_ids", [])
        if not isinstance(raw_excluded_ids, list) or any(
            not isinstance(case_id, str) or not case_id.strip() for case_id in raw_excluded_ids
        ):
            raise ValueError(f"comparison annotation {group_id} excluded_case_ids are invalid")
        excluded_case_ids = frozenset(case_id.strip() for case_id in raw_excluded_ids)
        if len(excluded_case_ids) != len(raw_excluded_ids):
            raise ValueError(f"comparison annotation {group_id} has duplicate excluded_case_ids")

        expected_case_count = group.get("expected_case_count")
        if not isinstance(expected_case_count, int) or expected_case_count < 1:
            raise ValueError(
                f"comparison annotation {group_id} expected_case_count must be positive"
            )
        skip_comparison = group.get("skip_comparison")
        if skip_comparison is not True:
            raise ValueError(f"comparison annotation {group_id} must skip comparison")

        annotations.append(
            ComparisonAnnotation(
                group_id=group_id,
                status=_required_string(group.get("status"), field="status"),
                reason=_required_string(group.get("reason"), field="reason"),
                skip_comparison=skip_comparison,
                expected_case_count=expected_case_count,
                minimum_element_counts=tuple(sorted(minimum_counts)),
                excluded_case_ids=excluded_case_ids,
            )
        )
    return tuple(annotations)


def find_comparison_annotation(
    case_id: str,
    element_counts: Mapping[str, int],
) -> ComparisonAnnotation | None:
    matches = [
        annotation
        for annotation in load_comparison_annotations()
        if annotation.matches(case_id, element_counts)
    ]
    if len(matches) > 1:
        raise ValueError(f"multiple tmQMg comparison annotations match {case_id}")
    return matches[0] if matches else None


@lru_cache(maxsize=4)
def load_manual_reference_records(path: Path | None = None) -> dict[str, dict[str, Any]]:
    if path is None:
        path = Path(__file__).resolve().parents[2] / "scripts" / "tmqmg_manual_whitelist.json"
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("manual reference records must be keyed by case id")
    records: dict[str, dict[str, Any]] = {}
    for case_id, entry in payload.items():
        if not isinstance(case_id, str) or not isinstance(entry, dict):
            raise ValueError("manual reference records require string ids and object entries")
        records[case_id] = dict(entry)
    return records


def reference_validity(
    row: Mapping[str, Any],
    manual_entry: Mapping[str, Any] | None = None,
) -> tuple[str, str | None]:
    reason = str(row.get("reference_reason") or row.get("reference_answer_reason") or "").strip()
    wrong = row.get("reference_answer_wrong")
    if _csv_bool(wrong) is True:
        return (
            ("excluded", reason)
            if reason
            else ("unresolved", "Missing reference exclusion evidence")
        )
    if _csv_bool(row.get("reference_formula_match")) is False and row.get(
        "reference_formula_mismatch_detail"
    ):
        return "excluded", str(row["reference_formula_mismatch_detail"])
    reference_error = row.get("reference_error")
    if reference_error:
        status = (
            "excluded"
            if str(reference_error).startswith(
                "ValueError: Reference SMILES element counts differ from XYZ:"
            )
            else "unresolved"
        )
        return status, str(reference_error)
    if row.get("reference_status") == "excluded":
        return (
            ("excluded", reason)
            if reason
            else ("unresolved", "Missing reference exclusion evidence")
        )
    parse_status = row.get("reference_parse_status")
    if parse_status and parse_status != "ok":
        return "unresolved", str(parse_status)
    if row.get("reference_status") == "valid":
        return "valid", reason or None
    if row.get("reference_status") == "unresolved":
        return "unresolved", reason or "Reference validity unresolved"
    if manual_entry is not None:
        manual_reason = str(manual_entry.get("reason") or "").strip()
        manual_wrong = manual_entry.get("reference_answer_wrong")
        if (
            manual_entry.get("status") == "accepted_reference_issue"
            and _csv_bool(manual_wrong) is True
            and manual_reason
        ):
            return "excluded", manual_reason
        return "unresolved", manual_reason or "Unconfirmed manual reference-quality record"
    if row.get("reference_status") not in {None, "", "valid"} or row.get(
        "reference_answer_status"
    ) in {
        "accepted_ambiguous",
        "accepted_reference_issue",
        "manual_accepted_ambiguous",
        "manual_confirmed_wrong",
        "unconfirmed_reference_issue",
    }:
        return "unresolved", reason or "Reference validity unresolved"
    return "valid", None
