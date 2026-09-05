#!/usr/bin/env python3
"""Local molecule graph review server."""
# ruff: noqa: E402, I001

from __future__ import annotations

import argparse
import html
import json
import mimetypes
import os
import sqlite3
import sys
import traceback
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from collections.abc import Mapping
from typing import Any
from urllib.parse import parse_qs, quote, unquote, urlparse
from urllib.request import Request, urlopen


APP_DIR = Path(__file__).resolve().parent
REPO_ROOT = next(
    (
        candidate
        for candidate in (APP_DIR.parent, Path(__file__).resolve().parents[2])
        if (candidate / "scripts" / "reconstruction_trace.py").is_file()
    ),
    Path(__file__).resolve().parents[2],
)
if __package__ in (None, ""):
    sys.path.insert(0, str(APP_DIR))
    sys.path.insert(0, str(REPO_ROOT))
    sys.path.insert(0, str(REPO_ROOT / "src"))

from rdkit import Chem, RDLogger

from fixture_builder import (
    case_electronic_state,
    load_fixture_records,
    reconstruct_case_mol,
    resolve_xyz_path,
    sync_review_fixture,
)
from molgr.utils.equivalence import evaluate_equivalence
from project_runtime import validate_project_runtime
from scripts.reconstruction_trace import TraceInputCase, render_trace_report


RDLogger.DisableLog("rdApp.*")  # type: ignore[arg-type]

STATIC_DIR = APP_DIR / "static"
DEFAULT_DB = REPO_ROOT / ".local" / "molgr_review" / "review.sqlite"
DEFAULT_XYZ_DIR = os.environ.get("MOLGR_XYZ_DIR", "")
KETCHER_BASE_URL = "https://lifescience.opensource.epam.com"
ALLOWED_STATUSES = {
    "accept_both",
    "accept_candidate",
    "accept_reference",
    "manual_reference",
    "reference_answer_wrong",
    "needs_followup",
    "skip",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--db",
        type=Path,
        default=Path(os.environ.get("MOLGR_REVIEW_DB", DEFAULT_DB)),
        help="SQLite database path.",
    )
    parser.add_argument(
        "--host",
        default=os.environ.get("MOLGR_REVIEW_HOST", "127.0.0.1"),
        help="Bind host.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("MOLGR_REVIEW_PORT", "8765")),
        help="Bind port.",
    )
    parser.add_argument(
        "--xyz-dir",
        type=Path,
        default=Path(DEFAULT_XYZ_DIR) if DEFAULT_XYZ_DIR else None,
        help="Optional directory used to resolve XYZ files by basename.",
    )
    parser.add_argument(
        "--fixtures-dir",
        type=Path,
        required=True,
        help="Directory updated immediately from fixture-producing review decisions.",
    )
    return parser.parse_args()


def _connect(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def _row_dict(
    row: sqlite3.Row | Mapping[str, Any] | None,
    *,
    fixture: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    if row is None:
        return None
    payload = dict(row)
    payload["fixture"] = fixture
    metadata_json = payload.pop("metadata_json", None)
    if isinstance(metadata_json, str) and metadata_json:
        try:
            raw_payload = json.loads(metadata_json)
        except json.JSONDecodeError:
            raw_payload = None
        if isinstance(raw_payload, dict):
            for key, value in raw_payload.items():
                if key not in payload:
                    payload[key] = value

    runtime_label = f"py{sys.version_info.major}{sys.version_info.minor}"
    snapshot_smiles = ""
    snapshot_status = ""
    snapshot_found = False
    for method_id in ("candidate_cpp", "molgr_cpp"):
        smiles_key = f"{runtime_label}_{method_id}_smiles"
        status_key = f"{runtime_label}_{method_id}_status"
        if smiles_key in payload or status_key in payload:
            snapshot_smiles = str(payload.get(smiles_key) or "")
            snapshot_status = str(payload.get(status_key) or "")
            snapshot_found = True
            break
    payload["candidate_snapshot_runtime"] = runtime_label
    if snapshot_found:
        payload["candidate_snapshot_smiles"] = snapshot_smiles
        payload["candidate_snapshot_status"] = snapshot_status
    else:
        payload["candidate_snapshot_smiles"] = str(payload.get("candidate_smiles") or "")
        payload["candidate_snapshot_status"] = str(payload.get("candidate_status") or "")
    return payload


def _json_response(
    handler: BaseHTTPRequestHandler,
    payload: Any,
    *,
    status: HTTPStatus = HTTPStatus.OK,
) -> None:
    body = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def _text_response(
    handler: BaseHTTPRequestHandler,
    text: str,
    *,
    status: HTTPStatus = HTTPStatus.OK,
    content_type: str = "text/plain; charset=utf-8",
) -> None:
    body = text.encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", content_type)
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def _trace_error_page(case_id: str, error: Exception) -> str:
    escaped_case = html.escape(case_id)
    escaped_error = html.escape(f"{type(error).__name__}: {error}")
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><title>Molecule Trace - {escaped_case}</title>
<style>body{{font-family:system-ui,sans-serif;margin:2rem;color:#172033}}code{{white-space:pre-wrap}}</style>
</head><body><h1>Trace 生成失败: {escaped_case}</h1><code>{escaped_error}</code></body></html>"""


def _svg_fragment_from_image(image: Any) -> str:
    data = getattr(image, "data", None)
    if isinstance(data, str):
        return data
    repr_svg = getattr(image, "_repr_svg_", None)
    if callable(repr_svg):
        rendered = repr_svg()
        if isinstance(rendered, str):
            return rendered
    if isinstance(image, str):
        return image
    return str(image)


def _safe_smiles(mol: Chem.Mol) -> str:
    clone = Chem.RemoveHs(Chem.Mol(mol), sanitize=False)
    try:
        return Chem.MolToSmiles(clone, canonical=True, isomericSmiles=True)
    except Chem.KekulizeException:
        rw_mol = Chem.RWMol(clone)
        for atom in rw_mol.GetAtoms():  # pyright: ignore[reportCallIssue]
            atom.SetIsAromatic(False)
        for bond in rw_mol.GetBonds():  # pyright: ignore[reportCallIssue]
            bond.SetIsAromatic(False)
        fallback = rw_mol.GetMol()
        fallback.UpdatePropertyCache(strict=False)
        return Chem.MolToSmiles(fallback, canonical=True, isomericSmiles=True)


def _benchmark_candidate_mol(mol: Chem.Mol) -> Chem.Mol:
    """Apply the same successful-result postprocessing as the tmQMg benchmark."""

    # Candidate aromatic flags may be inconsistent after reconstruction edits;
    # RemoveHs must not trigger a fresh sanitization/Kekule pass here.
    return Chem.RemoveHs(Chem.Mol(mol), sanitize=False)


def _mol_from_smiles(smiles: str) -> Chem.Mol | None:
    if not smiles.strip():
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.AddHs(mol)


def _mol_from_smiles_without_sanitize(smiles: str) -> Chem.Mol | None:
    if not smiles.strip():
        return None
    mol = Chem.MolFromSmiles(smiles, sanitize=False)
    if mol is not None:
        mol.UpdatePropertyCache(strict=False)
    return mol


def _live_candidate_comparison(mol: Chem.Mol, snapshot_smiles: str) -> dict[str, Any]:
    live_smiles = _safe_smiles(mol)
    comparison: dict[str, Any] = {
        "candidate_snapshot_smiles": snapshot_smiles,
        "live_candidate_smiles_exact_match": (
            live_smiles == snapshot_smiles if snapshot_smiles else None
        ),
        "live_matches_candidate_snapshot": None,
        "live_candidate_equivalence_decision": "",
        "live_candidate_equivalence_method": "",
        "live_candidate_equivalence_reason": "",
    }
    if not snapshot_smiles:
        comparison["live_candidate_equivalence_reason"] = "candidate_snapshot_smiles_unavailable"
        return comparison

    snapshot = _mol_from_smiles_without_sanitize(snapshot_smiles)
    if snapshot is None:
        comparison["live_candidate_equivalence_reason"] = "candidate_snapshot_smiles_invalid"
        return comparison

    info = evaluate_equivalence(mol, snapshot, use_chirality=False)
    decision = info.decision.value
    comparison["live_candidate_equivalence_decision"] = decision
    comparison["live_matches_candidate_snapshot"] = (
        True if decision == "equivalent" else False if decision == "not_equivalent" else None
    )
    comparison["live_candidate_equivalence_method"] = info.method.value if info.method else ""
    comparison["live_candidate_equivalence_reason"] = info.reason
    return comparison


def _render_mol_svg(mol: Chem.Mol, *, legend: str) -> str:
    from rdkit_dof import MolToDofImage

    image = MolToDofImage(mol, size=(520, 360), legend=legend, use_svg=True, return_image=False)
    return _svg_fragment_from_image(image)


def _mol_to_sdf_block(mol: Chem.Mol) -> str:
    # Candidate graphs may carry aromatic flags that are inconsistent after
    # charge/bond edits. Preserve the explicit graph without forcing Kekule.
    block = Chem.MolToMolBlock(mol, kekulize=False, forceV3000=True)
    if not block.endswith("\n"):
        block += "\n"
    return block + "$$$$\n"


def _mol_from_sdf_block(sdf: str) -> Chem.Mol:
    molblock = sdf.partition("$$$$")[0]
    mol = Chem.MolFromMolBlock(
        molblock,
        sanitize=False,
        removeHs=False,
        strictParsing=False,
    )
    if mol is None:
        raise ValueError("invalid_sdf")
    mol.UpdatePropertyCache(strict=False)
    return mol


def _dof_size(payload: dict[str, Any], key: str, default: tuple[int, int]) -> tuple[int, int]:
    raw = payload.get(key)
    if raw is None:
        return default
    if not isinstance(raw, list) or len(raw) != 2:
        raise ValueError(f"invalid_{key}")
    width, height = int(raw[0]), int(raw[1])
    if not (1 <= width <= 2000 and 1 <= height <= 2000):
        raise ValueError(f"invalid_{key}")
    return width, height


def _render_deferred_dof(payload: dict[str, Any]) -> str:
    render_type = str(payload.get("render_type") or "")
    legends = payload.get("legends") or []
    if not isinstance(legends, list) or not all(isinstance(item, str) for item in legends):
        raise ValueError("invalid_legends")

    if render_type == "single":
        sdf = payload.get("sdf")
        if not isinstance(sdf, str) or not sdf:
            raise ValueError("missing_sdf")
        from rdkit_dof import MolToDofImage

        image = MolToDofImage(
            _mol_from_sdf_block(sdf),
            size=_dof_size(payload, "size", (360, 300)),
            legend=legends[0] if legends else "",
            use_svg=True,
            return_image=False,
        )
    else:
        sdfs = payload.get("sdfs")
        if (
            not isinstance(sdfs, list)
            or not sdfs
            or not all(isinstance(item, str) for item in sdfs)
        ):
            raise ValueError("missing_sdfs")
        mols = [_mol_from_sdf_block(sdf) for sdf in sdfs]
        if legends and len(legends) != len(mols):
            raise ValueError("legend_count_mismatch")
        if render_type == "grid":
            from rdkit_dof import MolsToGridDofImage

            mols_per_row = int(payload.get("mols_per_row") or 3)
            if not 1 <= mols_per_row <= 12:
                raise ValueError("invalid_mols_per_row")
            image = MolsToGridDofImage(
                mols,
                molsPerRow=mols_per_row,
                subImgSize=_dof_size(payload, "sub_image_size", (320, 260)),
                legends=legends,
                use_svg=True,
                return_image=False,
            )
        elif render_type == "animation":
            from rdkit_dof import MolsToDofSvgAnimation

            duration = int(payload.get("duration") or 650)
            if not 50 <= duration <= 10_000:
                raise ValueError("invalid_duration")
            image = MolsToDofSvgAnimation(
                mols,
                size=_dof_size(payload, "size", (360, 300)),
                legends=legends,
                duration=duration,
                loop=0,
                return_image=False,
            )
        else:
            raise ValueError("invalid_render_type")
    svg = _svg_fragment_from_image(image)
    svg_start = svg.find("<svg")
    return svg[svg_start:] if svg_start >= 0 else svg


def _case_query(select_extra: str = "") -> str:
    return f"""
        SELECT
            c.case_id, c.row_index, c.source, c.category, c.xyz_path,
            c.total_charge, c.total_radical_electrons, c.spin_multiplicity,
            c.reference_smiles, c.candidate_smiles,
            c.candidate_organic_smiles, c.reference_organic_smiles,
            c.candidate_status, c.metadata_json,
            r.status AS review_status, r.corrected_smiles, r.corrected_molblock,
            r.notes, r.reviewer, r.updated_at
            {select_extra}
        FROM cases c
        LEFT JOIN reviews r ON r.case_id = c.case_id
    """


class ReviewServer(ThreadingHTTPServer):
    db_path: Path
    xyz_dir: Path | None
    fixtures_dir: Path
    runtime_info: dict[str, Any]


class ReviewHandler(BaseHTTPRequestHandler):
    server: ReviewServer

    def log_message(self, format: str, *args: object) -> None:
        sys.stderr.write("[molgr-review] " + format % args + "\n")

    def do_GET(self) -> None:  # noqa: N802
        try:
            self._handle_get()
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            _json_response(
                self,
                {"error": f"{type(exc).__name__}: {exc}"},
                status=HTTPStatus.INTERNAL_SERVER_ERROR,
            )

    def do_POST(self) -> None:  # noqa: N802
        try:
            self._handle_post()
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            _json_response(
                self,
                {"error": f"{type(exc).__name__}: {exc}"},
                status=HTTPStatus.INTERNAL_SERVER_ERROR,
            )

    def _handle_get(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        query = parse_qs(parsed.query)
        if path in ("", "/"):
            self._serve_static("index.html")
        elif path.startswith("/static/"):
            self._serve_static(path[len("/static/") :])
        elif path.startswith("/KetcherDemoSA/"):
            self._proxy_ketcher(path, parsed.query)
        elif path.startswith("/ketcher/"):
            self._proxy_ketcher("/KetcherDemoSA/" + path[len("/ketcher/") :], parsed.query)
        elif path.startswith("/trace/"):
            self._trace_page(unquote(path[len("/trace/") :]))
        elif path == "/api/stats":
            self._api_stats()
        elif path == "/api/cases":
            self._api_cases(query)
        elif path.startswith("/api/cases/") and path.endswith("/render"):
            case_id = unquote(path.split("/")[3])
            self._api_render(case_id, query)
        elif path.startswith("/api/cases/") and path.endswith("/xyz"):
            case_id = unquote(path.split("/")[3])
            self._api_xyz(case_id)
        elif path.startswith("/api/cases/") and path.endswith("/candidate-sdf"):
            case_id = unquote(path.split("/")[3])
            self._api_candidate_sdf(case_id)
        elif path.startswith("/api/cases/"):
            case_id = unquote(path.split("/")[3])
            self._api_case(case_id)
        else:
            _json_response(self, {"error": "not_found"}, status=HTTPStatus.NOT_FOUND)

    def _trace_page(self, case_id: str) -> None:
        if not case_id:
            _text_response(self, "case_id is required", status=HTTPStatus.BAD_REQUEST)
            return
        with _connect(self.server.db_path) as conn:
            row = conn.execute("SELECT * FROM cases WHERE case_id = ?", (case_id,)).fetchone()
        if row is None:
            _text_response(self, "case_not_found", status=HTTPStatus.NOT_FOUND)
            return

        case = dict(row)
        xyz_path = resolve_xyz_path(str(case.get("xyz_path") or ""), self.server.xyz_dir)
        try:
            if not xyz_path.is_file():
                raise FileNotFoundError(xyz_path)
            total_charge, total_radicals, _ = case_electronic_state(case)
            fixture = load_fixture_records(self.server.fixtures_dir).get(case_id) or {}
            input_case = TraceInputCase(
                id=case_id,
                xyz_block=xyz_path.read_text(encoding="utf-8"),
                total_charge=total_charge,
                total_radical_electrons=total_radicals,
                xyz_path=xyz_path,
                xyz_source="review_page",
                fixture_kind=str(fixture.get("kind") or ""),
                fixture_structure_file=str(fixture.get("structure_file") or ""),
                expected_smiles=str(fixture.get("approved_smiles") or ""),
                expected_smiles_options=tuple(
                    str(value).strip()
                    for value in fixture.get("accepted_smiles", [])
                    if str(value).strip()
                ),
                reference_smiles=str(case.get("reference_smiles") or ""),
            )

            report = render_trace_report(
                [input_case],
                score_all_candidates=False,
                dof_max_images=None,
                defer_dof_images=True,
            )
        except Exception as exc:  # noqa: BLE001
            _text_response(
                self,
                _trace_error_page(case_id, exc),
                status=HTTPStatus.INTERNAL_SERVER_ERROR,
                content_type="text/html; charset=utf-8",
            )
            return
        _text_response(self, report, content_type="text/html; charset=utf-8")

    def _handle_post(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        if path == "/api/render-dof":
            self._api_render_dof()
        elif path.startswith("/api/cases/") and path.endswith("/review"):
            case_id = unquote(path.split("/")[3])
            self._api_review(case_id)
        else:
            _json_response(self, {"error": "not_found"}, status=HTTPStatus.NOT_FOUND)

    def _api_render_dof(self) -> None:
        content_length = int(self.headers.get("Content-Length") or 0)
        if not 0 < content_length <= 16 * 1024 * 1024:
            _json_response(self, {"error": "invalid_content_length"}, status=HTTPStatus.BAD_REQUEST)
            return
        try:
            payload = json.loads(self.rfile.read(content_length))
            if not isinstance(payload, dict):
                raise ValueError("payload_must_be_an_object")
            svg = _render_deferred_dof(payload)
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            _json_response(self, {"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
            return
        _text_response(self, svg, content_type="image/svg+xml; charset=utf-8")

    def _serve_static(self, name: str) -> None:
        requested = (STATIC_DIR / name).resolve()
        if STATIC_DIR.resolve() not in requested.parents and requested != STATIC_DIR.resolve():
            _json_response(self, {"error": "invalid_static_path"}, status=HTTPStatus.BAD_REQUEST)
            return
        if not requested.exists() or not requested.is_file():
            _json_response(self, {"error": "static_not_found"}, status=HTTPStatus.NOT_FOUND)
            return
        body = requested.read_bytes()
        content_type = mimetypes.guess_type(str(requested))[0] or "application/octet-stream"
        if requested.suffix in {".html", ".js", ".css"}:
            content_type += "; charset=utf-8"
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _proxy_ketcher(self, path: str, query: str) -> None:
        if ".." in path:
            _json_response(self, {"error": "invalid_ketcher_path"}, status=HTTPStatus.BAD_REQUEST)
            return
        encoded_path = quote(path, safe="/._-")
        url = KETCHER_BASE_URL + encoded_path
        if query:
            url += "?" + query
        request = Request(
            url,
            headers={
                "User-Agent": "Molecule graph review client",
                "Accept": self.headers.get("Accept", "*/*"),
            },
        )
        with urlopen(request, timeout=30) as response:
            body = response.read()
            content_type = response.headers.get("Content-Type") or mimetypes.guess_type(path)[0]
            content_type = content_type or "application/octet-stream"
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "public, max-age=3600")
            self.end_headers()
            self.wfile.write(body)

    def _read_json_body(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length)
        if not body:
            return {}
        value = json.loads(body.decode("utf-8"))
        if not isinstance(value, dict):
            raise ValueError("JSON request body must be an object")
        return value

    def _api_stats(self) -> None:
        with _connect(self.server.db_path) as conn:
            category_rows = conn.execute(
                "SELECT category, COUNT(*) AS count FROM cases GROUP BY category"
            ).fetchall()
            status_rows = conn.execute(
                """
                SELECT COALESCE(r.status, 'unreviewed') AS status, COUNT(*) AS count
                FROM cases c
                LEFT JOIN reviews r ON r.case_id = c.case_id
                GROUP BY COALESCE(r.status, 'unreviewed')
                """
            ).fetchall()
            metadata_rows = conn.execute("SELECT key, value FROM metadata").fetchall()
        _json_response(
            self,
            {
                "categories": {row["category"]: row["count"] for row in category_rows},
                "review_statuses": {row["status"]: row["count"] for row in status_rows},
                "metadata": {row["key"]: row["value"] for row in metadata_rows},
                "runtime": getattr(self.server, "runtime_info", {}),
            },
        )

    def _api_cases(self, query: dict[str, list[str]]) -> None:
        category = query.get("category", [""])[0]
        status = query.get("status", [""])[0]
        search = query.get("q", [""])[0].strip()
        limit = max(1, min(int(query.get("limit", ["100"])[0]), 500))
        offset = max(0, int(query.get("offset", ["0"])[0]))

        where: list[str] = []
        params: list[Any] = []
        if category:
            where.append("c.category = ?")
            params.append(category)
        if status:
            if status == "unreviewed":
                where.append("r.status IS NULL")
            else:
                where.append("r.status = ?")
                params.append(status)
        if search:
            where.append("(c.case_id LIKE ? OR CAST(c.row_index AS TEXT) = ?)")
            params.extend((f"%{search}%", search))
        where_sql = (" WHERE " + " AND ".join(where)) if where else ""

        with _connect(self.server.db_path) as conn:
            rows = conn.execute(
                _case_query() + where_sql + " ORDER BY c.row_index LIMIT ? OFFSET ?",
                (*params, limit, offset),
            ).fetchall()
            total = conn.execute(
                """
                SELECT COUNT(*) AS count
                FROM cases c
                LEFT JOIN reviews r ON r.case_id = c.case_id
                """
                + where_sql,
                params,
            ).fetchone()["count"]
        fixture_records = load_fixture_records(self.server.fixtures_dir)
        _json_response(
            self,
            {
                "total": total,
                "items": [
                    _row_dict(row, fixture=fixture_records.get(str(row["case_id"]))) for row in rows
                ],
            },
        )

    def _api_case(self, case_id: str) -> None:
        with _connect(self.server.db_path) as conn:
            row = conn.execute(_case_query() + " WHERE c.case_id = ?", (case_id,)).fetchone()
        fixture_records = load_fixture_records(self.server.fixtures_dir)
        payload = _row_dict(
            row,
            fixture=fixture_records.get(case_id),
        )
        if payload is None:
            _json_response(self, {"error": "case_not_found"}, status=HTTPStatus.NOT_FOUND)
            return
        _json_response(self, payload)

    def _api_xyz(self, case_id: str) -> None:
        with _connect(self.server.db_path) as conn:
            row = conn.execute(
                "SELECT xyz_path FROM cases WHERE case_id = ?", (case_id,)
            ).fetchone()
        if row is None:
            _json_response(self, {"error": "case_not_found"}, status=HTTPStatus.NOT_FOUND)
            return
        path = resolve_xyz_path(row["xyz_path"], self.server.xyz_dir)
        if not path.exists():
            _json_response(
                self,
                {"error": "xyz_not_found", "xyz_path": str(path)},
                status=HTTPStatus.NOT_FOUND,
            )
            return
        _text_response(self, path.read_text(encoding="utf-8"), content_type="chemical/x-xyz")

    def _api_candidate_sdf(self, case_id: str) -> None:
        with _connect(self.server.db_path) as conn:
            row = conn.execute(
                """
                SELECT *
                FROM cases
                WHERE case_id = ?
                """,
                (case_id,),
            ).fetchone()
        if row is None:
            _json_response(self, {"error": "case_not_found"}, status=HTTPStatus.NOT_FOUND)
            return

        case = _row_dict(row)
        assert case is not None
        snapshot_smiles = str(case["candidate_snapshot_smiles"] or "")

        try:
            reconstructed_mol = reconstruct_case_mol(dict(row), xyz_dir=self.server.xyz_dir)
            candidate_mol = _benchmark_candidate_mol(reconstructed_mol)
            sdf = _mol_to_sdf_block(reconstructed_mol)
        except Exception as exc:  # noqa: BLE001
            _json_response(
                self,
                {
                    "sdf": "",
                    "svg": "",
                    "smiles": "",
                    "available": False,
                    "source": "live_reconstruction",
                    "live_candidate_status": "failed",
                    "live_candidate_smiles": "",
                    "live_candidate_smiles_exact_match": None,
                    "live_matches_candidate_snapshot": None,
                    "candidate_snapshot_smiles": snapshot_smiles,
                    "live_candidate_equivalence_decision": "",
                    "live_candidate_equivalence_reason": "live_reconstruction_failed",
                    "error": f"{type(exc).__name__}: {exc}",
                    "render_error": "",
                },
            )
            return

        smiles = _safe_smiles(candidate_mol)
        svg = ""
        render_error = ""
        try:
            svg = _render_mol_svg(
                reconstructed_mol,
                legend=f"{case_id} candidate current",
            )
        except Exception as exc:  # noqa: BLE001
            render_error = f"{type(exc).__name__}: {exc}"
        _json_response(
            self,
            {
                "sdf": sdf,
                "svg": svg,
                "smiles": smiles,
                "available": True,
                "source": "live_reconstruction",
                "live_candidate_status": "ok",
                "live_candidate_smiles": smiles,
                "error": "",
                "render_error": render_error,
                **_live_candidate_comparison(candidate_mol, snapshot_smiles),
            },
        )

    def _api_render(self, case_id: str, query: dict[str, list[str]]) -> None:
        kind = query.get("kind", ["candidate"])[0]
        if kind not in {"candidate", "reference", "candidate_organic", "reference_organic"}:
            _json_response(self, {"error": "invalid_render_kind"}, status=HTTPStatus.BAD_REQUEST)
            return

        with _connect(self.server.db_path) as conn:
            if kind != "candidate":
                cached = conn.execute(
                    "SELECT svg, smiles, error FROM render_cache WHERE case_id = ? AND kind = ?",
                    (case_id, kind),
                ).fetchone()
                if cached is not None and not cached["error"]:
                    _json_response(self, {"kind": kind, **dict(cached)})
                    return

            case = conn.execute("SELECT * FROM cases WHERE case_id = ?", (case_id,)).fetchone()
            if case is None:
                _json_response(self, {"error": "case_not_found"}, status=HTTPStatus.NOT_FOUND)
                return

            svg = ""
            smiles = ""
            error = ""
            try:
                if kind == "candidate":
                    mol = reconstruct_case_mol(dict(case), xyz_dir=self.server.xyz_dir)
                    smiles = _safe_smiles(mol)
                    svg = _render_mol_svg(mol, legend=f"{case_id} candidate")
                elif kind == "reference":
                    mol = _mol_from_smiles(case["reference_smiles"] or "")
                    if mol is None:
                        raise ValueError("reference_smiles_missing_or_invalid")
                    smiles = _safe_smiles(mol)
                    svg = _render_mol_svg(mol, legend=f"{case_id} Reference")
                elif kind == "candidate_organic":
                    mol = _mol_from_smiles(case["candidate_organic_smiles"] or "")
                    if mol is None:
                        raise ValueError("candidate_organic_smiles_missing_or_invalid")
                    smiles = _safe_smiles(mol)
                    svg = _render_mol_svg(mol, legend=f"{case_id} candidate organic")
                else:
                    mol = _mol_from_smiles(case["reference_organic_smiles"] or "")
                    if mol is None:
                        raise ValueError("reference_organic_smiles_missing_or_invalid")
                    smiles = _safe_smiles(mol)
                    svg = _render_mol_svg(mol, legend=f"{case_id} Reference organic")
            except Exception as exc:  # noqa: BLE001
                error = f"{type(exc).__name__}: {exc}"

            if kind != "candidate":
                conn.execute(
                    """
                    INSERT OR REPLACE INTO render_cache(
                        case_id, kind, svg, smiles, error, generated_at
                    )
                    VALUES(?, ?, ?, ?, ?, ?)
                    """,
                    (case_id, kind, svg, smiles, error, datetime.now(timezone.utc).isoformat()),
                )
                conn.commit()
        _json_response(self, {"kind": kind, "svg": svg, "smiles": smiles, "error": error})

    def _api_review(self, case_id: str) -> None:
        payload = self._read_json_body()
        status = str(payload.get("status", "")).strip()
        if status not in ALLOWED_STATUSES:
            _json_response(
                self,
                {"error": "invalid_status", "allowed": sorted(ALLOWED_STATUSES)},
                status=HTTPStatus.BAD_REQUEST,
            )
            return

        corrected_smiles = str(payload.get("corrected_smiles", "")).strip()
        corrected_molblock = str(payload.get("corrected_molblock", ""))
        if status == "manual_reference" and not corrected_smiles:
            _json_response(
                self,
                {"error": "corrected_smiles_required_for_manual_reference"},
                status=HTTPStatus.BAD_REQUEST,
            )
            return

        updated_at = datetime.now(timezone.utc).isoformat()
        review = {
            "status": status,
            "corrected_smiles": corrected_smiles,
            "corrected_molblock": corrected_molblock,
            "notes": str(payload.get("notes", "")).strip(),
            "reviewer": str(payload.get("reviewer", "")).strip(),
            "updated_at": updated_at,
        }
        with _connect(self.server.db_path) as conn:
            case = conn.execute("SELECT * FROM cases WHERE case_id = ?", (case_id,)).fetchone()
            if case is None:
                _json_response(self, {"error": "case_not_found"}, status=HTTPStatus.NOT_FOUND)
                return
            try:
                fixture = sync_review_fixture(
                    dict(case),
                    review,
                    fixtures_dir=self.server.fixtures_dir,
                    xyz_dir=self.server.xyz_dir,
                )
            except (FileNotFoundError, ValueError) as exc:
                _json_response(
                    self,
                    {"error": str(exc)},
                    status=HTTPStatus.BAD_REQUEST,
                )
                return
            conn.execute(
                """
                INSERT INTO reviews(
                    case_id, status, corrected_smiles, corrected_molblock,
                    notes, reviewer, updated_at
                )
                VALUES(?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(case_id) DO UPDATE SET
                    status = excluded.status,
                    corrected_smiles = excluded.corrected_smiles,
                    corrected_molblock = excluded.corrected_molblock,
                    notes = excluded.notes,
                    reviewer = excluded.reviewer,
                    updated_at = excluded.updated_at
                """,
                (
                    case_id,
                    status,
                    corrected_smiles,
                    corrected_molblock,
                    review["notes"],
                    review["reviewer"],
                    updated_at,
                ),
            )
            conn.commit()
        _json_response(self, {"ok": True, "fixture": fixture})


def main() -> None:
    args = _parse_args()
    if not args.db.exists():
        raise SystemExit(
            f"Database does not exist: {args.db}\n"
            "Create it first with: uv run python tools/molgr_review/import_cases.py --input <queue.csv> --db <path>"
        )
    try:
        runtime_info = validate_project_runtime(REPO_ROOT)
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc
    server = ReviewServer((args.host, args.port), ReviewHandler)
    server.db_path = args.db
    server.xyz_dir = args.xyz_dir
    server.fixtures_dir = args.fixtures_dir
    server.runtime_info = runtime_info
    server.fixtures_dir.mkdir(parents=True, exist_ok=True)
    print(f"Molecule review server: http://{args.host}:{args.port}")
    print(f"database: {args.db}")
    if args.xyz_dir is not None:
        print(f"xyz directory: {args.xyz_dir}")
    print(f"review fixtures: {args.fixtures_dir}")
    print(f"molgr source: {runtime_info['molgr_source']}")
    print(f"C++ extension: {runtime_info['cpp_extension']}")
    print(
        "checkout: "
        f"{runtime_info['git_revision'] or 'unknown'}"
        f"{' (dirty)' if runtime_info['git_dirty'] else ''}"
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
