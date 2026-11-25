from __future__ import annotations

import io
import json
import sqlite3
import sys
import threading
from http.client import HTTPConnection
from pathlib import Path
from typing import Any

import pytest
from rdkit import Chem
from rdkit.Chem import AllChem


APP_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP_DIR))

import fixture_builder  # noqa: E402
import server as review_server  # noqa: E402
from fixture_builder import (  # noqa: E402
    ACCEPT_BOTH_STATUS,
    _sdf_text,
    case_electronic_state,
    load_fixture_records,
    sync_review_fixture,
)
from server import ReviewHandler, ReviewServer  # noqa: E402

from molgr.utils.converter import (  # noqa: E402
    METAL_UNPAIRED_ELECTRONS_PROP,
    get_atom_unpaired_electrons,
)


def test_review_page_exposes_fixture_removal_action() -> None:
    html = (APP_DIR / "static" / "index.html").read_text(encoding="utf-8")
    javascript = (APP_DIR / "static" / "app.js").read_text(encoding="utf-8")
    stylesheet = (APP_DIR / "static" / "style.css").read_text(encoding="utf-8")

    assert 'id="removeFixture"' in html
    assert 'id="languageToggle"' in html
    assert 'data-i18n="languageToggle"' in html
    assert 'id="openTrace"' in html
    assert '$("removeFixture").addEventListener("click", removeCurrentFixture)' in javascript
    assert 'localStorage.setItem("moleculeReviewLanguage", state.language)' in javascript
    assert "function applyLanguage()" in javascript
    assert "function renderKetcherStatus()" in javascript
    assert 'setKetcherStatus("ketcherReady")' in javascript
    assert "window.open(" in javascript
    assert "`/trace/${encodeURIComponent(state.current.case_id)}`" in javascript
    assert 'await saveReview("needs_followup")' in javascript
    assert 'data-status="accept_both"' in html
    assert 'accept_both: "接受两者"' in javascript
    assert 'id="imageLightbox"' in html
    assert 'class="viewer-grid"' in html
    assert 'loading="lazy"' in html
    assert html.index('id="reviewControls"') < html.index('id="mainLayout"')
    assert html.index('id="mainLayout"') < html.index('id="reviewPanel"')
    assert "function openImageLightbox(box)" in javascript
    assert "dialog.showModal()" in javascript
    assert "if (workspace) workspace.scrollTop = 0;" in javascript
    assert '<code title="${escapeHtml(value)}">' in javascript
    assert (
        "grid-template-columns: clamp(260px, 34vw, var(--sidebar-width)) "
        "minmax(0, 1fr);" in stylesheet
    )
    assert "min-width: 1200px;" in stylesheet
    assert "@media (max-width: 1700px)" in stylesheet
    assert "contain: layout paint;" in stylesheet
    assert "flex-direction: column;" in stylesheet
    assert "grid-template-columns: repeat(2, minmax(0, 1fr));" in stylesheet
    assert "text-overflow: ellipsis;" in stylesheet
    assert "body {\n    display: block;" not in stylesheet


def test_fixture_manifest_rejects_unconfirmed_answers(tmp_path: Path) -> None:
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "fixtures": [
                    {
                        "case_id": "PENDING",
                        "kind": "pending_algorithm",
                        "structure_file": "reference_graph/PENDING.xyz",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="non-confirmed answers"):
        load_fixture_records(tmp_path)


def test_review_trace_page_renders_current_case_in_new_document(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    xyz_path = tmp_path / "TRACE.xyz"
    xyz_path.write_text(
        """1
carbon
C 0.0 0.0 0.0
""",
        encoding="utf-8",
    )
    db_path = tmp_path / "review.sqlite"
    with sqlite3.connect(db_path) as conn:
        conn.executescript((APP_DIR / "schema.sql").read_text(encoding="utf-8"))
        conn.execute(
            """
            INSERT INTO cases(
                    case_id, row_index, category, xyz_path, total_charge,
                    total_radical_electrons, spin_multiplicity,
                    reference_smiles, candidate_status, metadata_json
                )
                VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "TRACE",
                1,
                "graph_not_equivalent",
                str(xyz_path),
                2,
                2,
                3,
                "C",
                "ok",
                json.dumps(
                    {
                        "spin_multiplicity_used": "3",
                        "total_radical_electrons_used": "2",
                    }
                ),
            ),
        )
        conn.commit()

    captured: dict[str, object] = {}

    def fake_render(
        cases: object,
        *,
        score_all_candidates: bool,
        dof_max_images: int | None,
        defer_dof_images: bool,
    ) -> str:
        captured["cases"] = cases
        captured["score_all_candidates"] = score_all_candidates
        captured["dof_max_images"] = dof_max_images
        captured["defer_dof_images"] = defer_dof_images
        return "<!doctype html><title>TRACE</title>"

    monkeypatch.setattr(review_server, "render_trace_report", fake_render)
    server = ReviewServer(("127.0.0.1", 0), ReviewHandler)
    server.db_path = db_path
    server.xyz_dir = None
    server.fixtures_dir = tmp_path / "reviewed"
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    def get(path: str) -> tuple[int, str, str]:
        connection = HTTPConnection("127.0.0.1", server.server_port, timeout=10)
        connection.request("GET", path)
        response = connection.getresponse()
        body = response.read().decode("utf-8")
        content_type = response.getheader("Content-Type") or ""
        connection.close()
        return response.status, body, content_type

    def post(path: str, payload: object) -> tuple[int, str, str]:
        connection = HTTPConnection("127.0.0.1", server.server_port, timeout=10)
        connection.request(
            "POST",
            path,
            body=json.dumps(payload),
            headers={"Content-Type": "application/json"},
        )
        response = connection.getresponse()
        body = response.read().decode("utf-8")
        content_type = response.getheader("Content-Type") or ""
        connection.close()
        return response.status, body, content_type

    try:
        status, body, content_type = get("/trace/TRACE")
        methane = Chem.AddHs(Chem.MolFromSmiles("C"))
        dof_status, dof_body, dof_content_type = post(
            "/api/render-dof",
            {
                "render_type": "single",
                "sdf": Chem.MolToMolBlock(methane) + "\n$$$$\n",
                "legends": ["methane"],
                "size": [360, 300],
            },
        )
        invalid_dof_status, _, _ = post("/api/render-dof", {"render_type": "single"})
        missing_status, _, _ = get("/trace/MISSING")
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=10)

    assert status == 200
    assert body == "<!doctype html><title>TRACE</title>"
    assert content_type == "text/html; charset=utf-8"
    assert captured["score_all_candidates"] is False
    assert captured["dof_max_images"] is None
    assert captured["defer_dof_images"] is True
    assert dof_status == 200
    assert dof_body.startswith("<svg")
    assert dof_content_type == "image/svg+xml; charset=utf-8"
    assert invalid_dof_status == 400
    traced_cases = captured["cases"]
    assert isinstance(traced_cases, list)
    traced_case = traced_cases[0]
    assert traced_case.id == "TRACE"
    assert traced_case.total_charge == 2
    assert traced_case.total_radical_electrons == 2
    assert traced_case.xyz_source == "review_page"
    assert missing_status == 404


def test_case_electronic_state_does_not_sanitize_reference_smiles() -> None:
    case = {
        "total_charge": 0,
        "reference_smiles": (
            "CO->[La]123456(ON(O)O)(<-OC)(<-N(O)(O1)O2)(<-N(O)(O3)O4)"
            "N1CCCCC1C1NC(C2CCCCN2)NC(C2CCCCN25)N16"
        ),
        "metadata_json": "{}",
    }

    assert case_electronic_state(case) == (0, 0, 1)


def test_review_sdf_preserves_metal_unpaired_electron_property() -> None:
    mol = Chem.MolFromSmiles("[Fe+2]", sanitize=False)
    assert mol is not None
    iron = mol.GetAtomWithIdx(0)
    iron.SetIntProp(METAL_UNPAIRED_ELECTRONS_PROP, 2)
    iron.SetNumRadicalElectrons(0)
    Chem.CreateAtomIntPropertyList(mol, METAL_UNPAIRED_ELECTRONS_PROP)

    restored = next(
        Chem.ForwardSDMolSupplier(
            io.BytesIO(_sdf_text(mol, {}).encode()),
            sanitize=False,
            removeHs=False,
        )
    )

    assert restored is not None
    restored_iron = restored.GetAtomWithIdx(0)
    assert restored_iron.GetNumRadicalElectrons() == 0
    assert get_atom_unpaired_electrons(restored_iron) == 2


def test_review_sdf_uses_case_spin_not_metal_unpaired_electrons(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mol = Chem.MolFromSmiles("[Fe+2]", sanitize=False)
    assert mol is not None
    mol.GetAtomWithIdx(0).SetIntProp(METAL_UNPAIRED_ELECTRONS_PROP, 2)
    Chem.CreateAtomIntPropertyList(mol, METAL_UNPAIRED_ELECTRONS_PROP)
    monkeypatch.setattr(fixture_builder, "reconstruct_case_mol", lambda *args, **kwargs: mol)

    xyz_path = tmp_path / "SPIN.xyz"
    xyz_path.write_text("1\nFe\nFe 0.0 0.0 0.0\n", encoding="utf-8")
    fixtures_dir = tmp_path / "fixtures"
    (fixtures_dir / "manifest.json").parent.mkdir(parents=True)
    (fixtures_dir / "manifest.json").write_text(
        json.dumps({"schema_version": 1, "fixtures": []}), encoding="utf-8"
    )
    case = {
        "case_id": "SPIN",
        "row_index": 1,
        "xyz_path": str(xyz_path),
        "total_charge": 0,
        "total_radical_electrons": 0,
        "spin_multiplicity": 1,
        "reference_smiles": "[Fe+2]",
        "metadata_json": "{}",
    }

    record = sync_review_fixture(
        case,
        {"status": ACCEPT_BOTH_STATUS},
        fixtures_dir=fixtures_dir,
    )

    assert record is not None
    assert record["total_radical_electrons"] == 0
    assert record["spin_multiplicity"] == 1
    sdf = next(
        mol
        for mol in Chem.SDMolSupplier(
            str(fixtures_dir / str(record["structure_file"])),
            sanitize=False,
            removeHs=False,
        )
        if mol is not None
    )
    assert sdf.GetProp("TOTAL_RADICAL_ELECTRONS") == "0"
    assert sdf.GetProp("SPIN_MULTIPLICITY") == "1"
    assert get_atom_unpaired_electrons(sdf.GetAtomWithIdx(0)) == 2


def test_review_fixture_sync_stores_manual_xyz_smiles_and_removes_stale_files(
    tmp_path: Path,
) -> None:
    source = Chem.AddHs(Chem.MolFromSmiles("C"))
    assert AllChem.EmbedMolecule(source, randomSeed=7) == 0  # pyright: ignore[reportAttributeAccessIssue]
    xyz_path = tmp_path / "TEST.xyz"
    xyz_path.write_text(Chem.MolToXYZBlock(source), encoding="utf-8")

    case = {
        "case_id": "TEST",
        "row_index": 1,
        "xyz_path": str(xyz_path),
        "total_charge": 0,
        "reference_smiles": "C",
        "metadata_json": json.dumps(
            {"spin_multiplicity_used": "1", "total_radical_electrons_used": "0"}
        ),
    }
    review = {
        "status": "manual_reference",
        "corrected_smiles": "[CH4]",
        "corrected_molblock": "deliberately not a MolBlock",
        "reviewer": "test",
        "notes": "approved correction",
        "updated_at": "2026-07-17T00:00:00+00:00",
    }
    fixtures_dir = tmp_path / "fixtures"
    fixtures_dir.mkdir()
    (fixtures_dir / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "source_dataset": {"name": "example", "revision": "pinned"},
                "fixtures": [],
            }
        ),
        encoding="utf-8",
    )

    record = sync_review_fixture(case, review, fixtures_dir=fixtures_dir)

    assert record is not None
    assert record["kind"] == "manual_reference"
    assert record["structure_file"] == "manual_reference/TEST.xyz"
    assert record["approved_smiles"] == "[CH4]"
    assert record["reference_smiles"] == "C"
    assert record["source"] == "corrected_smiles"
    assert record["total_charge"] == 0
    assert record["total_radical_electrons"] == 0
    assert record["spin_multiplicity"] == 1
    assert {"reviewer", "notes", "updated_at"}.isdisjoint(record)
    assert json.loads((fixtures_dir / "manifest.json").read_text())["source_dataset"] == {
        "name": "example",
        "revision": "pinned",
    }
    frozen_xyz = fixtures_dir / str(record["structure_file"])
    assert frozen_xyz.read_bytes() == xyz_path.read_bytes()

    removed = sync_review_fixture(
        case,
        {**review, "status": "needs_followup"},
        fixtures_dir=fixtures_dir,
    )
    assert removed is None
    assert json.loads((fixtures_dir / "manifest.json").read_text())["fixtures"] == []
    assert not (fixtures_dir / "manual_reference" / "TEST.xyz").exists()

    reference_record = sync_review_fixture(
        case,
        {
            **review,
            "status": "accept_reference",
            "corrected_smiles": "",
            "corrected_molblock": "",
        },
        fixtures_dir=fixtures_dir,
    )
    assert reference_record is not None
    assert reference_record["kind"] == "reference_graph"
    assert reference_record["reference_smiles"] == "C"
    assert reference_record["total_charge"] == 0
    assert reference_record["spin_multiplicity"] == 1
    frozen_xyz = fixtures_dir / str(reference_record["structure_file"])
    assert frozen_xyz.read_text(encoding="utf-8") == xyz_path.read_text(encoding="utf-8")


def test_reference_answer_wrong_status_does_not_modify_fixture(tmp_path: Path) -> None:
    fixtures_dir = tmp_path / "fixtures"
    approved_dir = fixtures_dir / "approved_graph"
    approved_dir.mkdir(parents=True)
    approved_path = approved_dir / "WRONG.xyz"
    approved_path.write_text("fixture\n", encoding="utf-8")
    record = {
        "case_id": "WRONG",
        "kind": "approved_graph",
        "structure_file": "approved_graph/WRONG.xyz",
        "row_index": 1,
        "total_charge": 0,
        "total_radical_electrons": 0,
        "spin_multiplicity": 1,
        "reference_smiles": "C",
        "approved_smiles": "C",
        "accepted_smiles": ["C"],
        "source": "molgr_reconstruction",
        "review_status": "accept_candidate",
    }
    (fixtures_dir / "manifest.json").write_text(
        json.dumps({"schema_version": 1, "fixtures": [record]}),
        encoding="utf-8",
    )

    result = sync_review_fixture(
        {"case_id": "WRONG"},
        {"status": "reference_answer_wrong"},
        fixtures_dir=fixtures_dir,
    )

    assert result == record
    assert approved_path.read_text(encoding="utf-8") == "fixture\n"
    assert json.loads((fixtures_dir / "manifest.json").read_text(encoding="utf-8"))["fixtures"] == [
        record
    ]


def test_review_api_writes_fixture_with_review_decision(tmp_path: Path) -> None:
    xyz_path = tmp_path / "API.xyz"
    xyz_path.write_text(
        """3
water
O 0.000000 0.000000 0.000000
H 0.957200 0.000000 0.000000
H -0.239987 0.927297 0.000000
""",
        encoding="utf-8",
    )
    db_path = tmp_path / "review.sqlite"
    with sqlite3.connect(db_path) as conn:
        conn.executescript((APP_DIR / "schema.sql").read_text(encoding="utf-8"))
        conn.execute(
            """
            INSERT INTO cases(
                case_id, row_index, category, xyz_path, total_charge,
                reference_smiles, candidate_status, metadata_json
            )
            VALUES(?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "API",
                1,
                "graph_not_equivalent",
                str(xyz_path),
                0,
                "O",
                "ok",
                json.dumps(
                    {
                        "spin_multiplicity_used": "1",
                        "total_radical_electrons_used": "0",
                    }
                ),
            ),
        )
        conn.commit()

    server = ReviewServer(("127.0.0.1", 0), ReviewHandler)
    server.db_path = db_path
    server.xyz_dir = None
    server.fixtures_dir = tmp_path / "reviewed"
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    def post_review(review: dict[str, str]) -> tuple[int, dict[str, Any]]:
        connection = HTTPConnection("127.0.0.1", server.server_port, timeout=10)
        connection.request(
            "POST",
            "/api/cases/API/review",
            body=json.dumps(review),
            headers={"Content-Type": "application/json"},
        )
        response = connection.getresponse()
        response_payload = json.loads(response.read())
        connection.close()
        return response.status, response_payload

    try:
        missing_smiles_status, missing_smiles_payload = post_review(
            {
                "status": "manual_reference",
                "corrected_smiles": "",
                "corrected_molblock": "a MolBlock is not fixture-authoritative",
                "notes": "",
                "reviewer": "tester",
            }
        )
        reference_status, reference_payload = post_review(
            {
                "status": "accept_reference",
                "corrected_smiles": "",
                "corrected_molblock": "",
                "notes": "reference graph approved",
                "reviewer": "tester",
            }
        )
        both_status, both_payload = post_review(
            {
                "status": "accept_both",
                "corrected_smiles": "",
                "corrected_molblock": "",
                "notes": "candidate and reference are both acceptable",
                "reviewer": "tester",
            }
        )
        manual_status, manual_payload = post_review(
            {
                "status": "manual_reference",
                "corrected_smiles": "O",
                "corrected_molblock": "deliberately not a MolBlock",
                "notes": "manually approved graph",
                "reviewer": "tester",
            }
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=10)

    assert missing_smiles_status == 400
    assert missing_smiles_payload == {"error": "corrected_smiles_required_for_manual_reference"}
    assert reference_status == 200
    assert reference_payload["fixture"]["kind"] == "reference_graph"
    assert both_status == 200
    assert both_payload["fixture"]["kind"] == "accepted_both"
    assert len(both_payload["fixture"]["accepted_smiles"]) == 2
    assert both_payload["fixture"]["reference_smiles"] == "O"
    assert manual_status == 200
    assert manual_payload["fixture"]["kind"] == "manual_reference"
    assert (server.fixtures_dir / manual_payload["fixture"]["structure_file"]).is_file()
    assert not (server.fixtures_dir / "reference_graph" / "API.xyz").exists()
    with sqlite3.connect(db_path) as conn:
        assert conn.execute("SELECT status FROM reviews WHERE case_id = 'API'").fetchone() == (
            "manual_reference",
        )


def test_review_api_reports_and_removes_existing_fixture(tmp_path: Path) -> None:
    xyz_path = tmp_path / "SYNC.xyz"
    xyz_path.write_text(
        """1
carbon
C 0.0 0.0 0.0
""",
        encoding="utf-8",
    )
    fixtures_dir = tmp_path / "reviewed"
    case = {
        "case_id": "SYNC",
        "row_index": 1,
        "xyz_path": str(xyz_path),
        "total_charge": 0,
        "reference_smiles": "C",
        "metadata_json": json.dumps(
            {"spin_multiplicity_used": "1", "total_radical_electrons_used": "0"}
        ),
    }
    review = {
        "status": "accept_reference",
        "corrected_smiles": "",
        "corrected_molblock": "",
        "reviewer": "tester",
        "notes": "",
        "updated_at": "2026-07-18T00:00:00+00:00",
    }

    record = sync_review_fixture(case, review, fixtures_dir=fixtures_dir)
    assert record is not None
    assert load_fixture_records(fixtures_dir)["SYNC"]["kind"] == "reference_graph"

    db_path = tmp_path / "review.sqlite"
    with sqlite3.connect(db_path) as conn:
        conn.executescript((APP_DIR / "schema.sql").read_text(encoding="utf-8"))
        conn.execute(
            """
            INSERT INTO cases(
                case_id, row_index, category, xyz_path, total_charge,
                reference_smiles, candidate_status, metadata_json
            )
            VALUES(?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "SYNC",
                1,
                "graph_not_equivalent",
                str(xyz_path),
                0,
                "C",
                "ok",
                case["metadata_json"],
            ),
        )
        conn.commit()

    server = ReviewServer(("127.0.0.1", 0), ReviewHandler)
    server.db_path = db_path
    server.xyz_dir = None
    server.fixtures_dir = fixtures_dir
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    def request(path: str, body: dict[str, str] | None = None) -> tuple[int, dict[str, Any]]:
        connection = HTTPConnection("127.0.0.1", server.server_port, timeout=10)
        if body is None:
            connection.request("GET", path)
        else:
            connection.request(
                "POST",
                path,
                body=json.dumps(body),
                headers={"Content-Type": "application/json"},
            )
        response = connection.getresponse()
        payload = json.loads(response.read())
        connection.close()
        return response.status, payload

    try:
        status, payload = request("/api/cases/SYNC")
        assert status == 200
        assert payload["fixture"]["kind"] == "reference_graph"
        status, payload = request("/api/cases?q=SYNC")
        assert status == 200
        assert payload["items"][0]["fixture"]["kind"] == "reference_graph"

        status, payload = request(
            "/api/cases/SYNC/review",
            {
                "status": "needs_followup",
                "corrected_smiles": "",
                "corrected_molblock": "",
                "reviewer": "tester",
                "notes": "deferred",
            },
        )
        assert status == 200
        assert payload["fixture"] is None

        status, payload = request("/api/cases/SYNC")
        assert status == 200
        assert payload["fixture"] is None
        assert load_fixture_records(fixtures_dir) == {}
        assert not (fixtures_dir / "reference_graph" / "SYNC.xyz").exists()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=10)


def test_live_candidate_payload_is_self_consistent_and_does_not_use_snapshot_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db_path = tmp_path / "review.sqlite"
    with sqlite3.connect(db_path) as conn:
        conn.executescript((APP_DIR / "schema.sql").read_text(encoding="utf-8"))
        conn.execute(
            """
            INSERT INTO cases(
                case_id, row_index, category, xyz_path, total_charge,
                candidate_smiles, candidate_status, metadata_json
            )
            VALUES(?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "LIVE",
                1,
                "candidate_failed",
                str(tmp_path / "not-needed.xyz"),
                0,
                "C",
                "failed",
                "{}",
            ),
        )
        conn.execute(
            """
            INSERT INTO render_cache(case_id, kind, svg, smiles, error, generated_at)
            VALUES(?, ?, ?, ?, ?, ?)
            """,
            ("LIVE", "candidate", "<svg>stale</svg>", "N", "", "2026-07-17T00:00:00Z"),
        )
        conn.commit()

    reconstruct_calls = 0

    def reconstruct_live(*args: object, **kwargs: object) -> Chem.Mol:
        nonlocal reconstruct_calls
        reconstruct_calls += 1
        mol = Chem.MolFromSmiles("CC")
        assert mol is not None
        return Chem.AddHs(mol)

    monkeypatch.setattr(review_server, "reconstruct_case_mol", reconstruct_live)
    monkeypatch.setattr(
        review_server,
        "_render_mol_svg",
        lambda mol, *, legend: "<svg>live</svg>",
    )

    server = ReviewServer(("127.0.0.1", 0), ReviewHandler)
    server.db_path = db_path
    server.xyz_dir = None
    server.fixtures_dir = tmp_path / "reviewed"
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    def get_json(path: str) -> dict[str, object]:
        connection = HTTPConnection("127.0.0.1", server.server_port, timeout=10)
        connection.request("GET", path)
        response = connection.getresponse()
        assert response.status == 200
        payload = json.loads(response.read())
        connection.close()
        return payload

    try:
        case_payload = get_json("/api/cases/LIVE")
        live_payload = get_json("/api/cases/LIVE/candidate-sdf")
        first_render = get_json("/api/cases/LIVE/render?kind=candidate")
        second_render = get_json("/api/cases/LIVE/render?kind=candidate")
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=10)

    assert case_payload["candidate_snapshot_status"] == "failed"
    assert case_payload["candidate_snapshot_smiles"] == "C"
    assert live_payload["available"] is True
    assert live_payload["source"] == "live_reconstruction"
    assert live_payload["live_candidate_status"] == "ok"
    assert live_payload["live_candidate_smiles"] == "CC"
    assert live_payload["live_candidate_smiles_exact_match"] is False
    assert live_payload["smiles"] == "CC"
    sdf_mol = Chem.MolFromMolBlock(
        str(live_payload["sdf"]).partition("$$$$")[0],
        sanitize=False,
        removeHs=False,
    )
    assert sdf_mol is not None
    assert sdf_mol.GetNumAtoms() == 8
    assert sum(atom.GetAtomicNum() == 1 for atom in sdf_mol.GetAtoms()) == 6  # pyright: ignore[reportCallIssue]
    assert live_payload["candidate_snapshot_smiles"] == "C"
    assert live_payload["live_matches_candidate_snapshot"] is False
    assert live_payload["live_candidate_equivalence_reason"] == (
        "Heavy-atom formulas differ."
    )
    assert live_payload["svg"] == "<svg>live</svg>"
    assert first_render["svg"] == "<svg>live</svg>"
    assert second_render["svg"] == "<svg>live</svg>"
    assert reconstruct_calls == 3


def test_case_payload_selects_candidate_snapshot_for_server_python() -> None:
    runtime_label = f"py{sys.version_info.major}{sys.version_info.minor}"
    other_label = "py310" if runtime_label != "py310" else "py38"
    payload = review_server._row_dict(
        {
            "case_id": "RUNTIME",
            "candidate_smiles": "fallback",
            "candidate_status": "fallback_status",
            "metadata_json": json.dumps(
                {
                    f"{runtime_label}_molgr_cpp_smiles": "runtime",
                    f"{runtime_label}_molgr_cpp_status": "ok",
                    f"{other_label}_molgr_cpp_smiles": "other",
                    f"{other_label}_molgr_cpp_status": "error",
                }
            ),
        }
    )

    assert payload is not None
    assert payload["candidate_snapshot_runtime"] == runtime_label
    assert payload["candidate_snapshot_smiles"] == "runtime"
    assert payload["candidate_snapshot_status"] == "ok"

    failed_payload = review_server._row_dict(
        {
            "case_id": "FAILED",
            "candidate_smiles": "other-version-result",
            "candidate_status": "ok",
            "metadata_json": json.dumps(
                {
                    f"{runtime_label}_molgr_cpp_smiles": "",
                    f"{runtime_label}_molgr_cpp_status": "error",
                }
            ),
        }
    )
    assert failed_payload is not None
    assert failed_payload["candidate_snapshot_smiles"] == ""
    assert failed_payload["candidate_snapshot_status"] == "error"
