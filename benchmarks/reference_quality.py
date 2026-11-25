"""Reference-quality records for historical diagnostics, not dataset eligibility."""

from __future__ import annotations

import json
from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path
from typing import Any


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


@lru_cache(maxsize=4)
def load_manual_reference_records(path: Path | None = None) -> dict[str, dict[str, Any]]:
    if path is None:
        path = Path(__file__).resolve().parents[1] / "scripts" / "tmqmg_manual_whitelist.json"
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
