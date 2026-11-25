from pathlib import Path
from types import SimpleNamespace

from openbabel import pybel
from rdkit import Chem

import scripts.reconstruction_trace as reconstruction_trace
from molgr.fallback.state import OmolTraceRecorder
from molgr.fallback.utils.dataclasses import MetalAtomPosition
from scripts.reconstruction_trace import (
    DofRenderContext,
    TraceInputCase,
    _additional_negative_charge_visibility,
    _assign_global_trace_node_indices,
    _deferred_dof_record,
    _extract_deferred_dof_payloads,
    _metal_selection_basis,
    _oxidative_addition_valence_deltas,
    _render_html_browser_report,
    _reserve_dof_render_slot,
    _resonance_selection_report,
    _review_fixture_trace_check,
    dof_rendering_summary,
)


def _metal_state(idx: int, valence: int) -> MetalAtomPosition:
    return MetalAtomPosition(idx, "Fe", 26, valence, 0, 0.0, 0.0, 0.0)


def test_failed_no_metal_target_trace_remains_renderable(monkeypatch) -> None:
    prototype = SimpleNamespace(
        no_metal_charge_target=-1,
        no_metal_radical_target=2,
    )
    base_state = SimpleNamespace(
        available_valence_radical_states=[[_metal_state(1, 2)]],
        no_metal_xyz_block="organic xyz",
        metal_sites=(_metal_state(1, 2).site,),
        phase_history=["prepare_metal_state"],
        metadata={"metal_atom_count": 1},
    )
    failed_trace = {
        "status": "no_valid_no_metal_candidate",
        "target": {"total_charge": -1, "total_radical_electrons": 2},
        "trace_node_count": 1,
        "trace_nodes": [
            {
                "node_id": 0,
                "tree_parent_id": -1,
                "tree_depth": 0,
                "phase": "reject_no_metal_candidate_validation",
                "kind": "checkpoint",
                "event": {"kind": "checkpoint"},
                "metadata": {},
                "state": {},
            }
        ],
    }

    monkeypatch.setattr(
        reconstruction_trace.preparation,
        "prepare_metal_state",
        lambda *args, **kwargs: base_state,
    )
    monkeypatch.setattr(
        reconstruction_trace,
        "_metal_field_analysis_by_state",
        lambda *args, **kwargs: {},
    )
    monkeypatch.setattr(
        reconstruction_trace.search,
        "_build_metal_state_search_groups",
        lambda *args, **kwargs: [[object()]],
    )
    monkeypatch.setattr(
        reconstruction_trace.search,
        "_build_layered_metal_state_search_groups",
        lambda *args, **kwargs: [[[object()]]],
    )
    monkeypatch.setattr(
        reconstruction_trace.search,
        "_group_candidates_by_target_dp",
        lambda *args, **kwargs: {(-1, 2): [prototype]},
    )
    monkeypatch.setattr(
        reconstruction_trace,
        "_trace_no_metal_reconstruction",
        lambda *args, **kwargs: failed_trace,
    )
    monkeypatch.setattr(
        reconstruction_trace.reconstruct_without_metals,
        "xyz_to_omol_no_metal_state",
        lambda *args, **kwargs: None,
    )

    trace = reconstruction_trace._trace_candidates(
        "metal xyz",
        total_charge=0,
        total_radical_electrons=0,
        score_all_candidates=False,
        case_id="FAILED",
    )
    layer = trace["search"]["layer_summaries"][0]
    assert "unprepared_target_buckets" not in layer
    assert len(layer["target_buckets"]) == 1
    bucket = layer["target_buckets"][0]
    assert bucket["status"] == "no_metal_reconstruction_none"
    assert bucket["no_metal_trace"]["trace_nodes"][0]["phase"] == (
        "reject_no_metal_candidate_validation"
    )

    case = {"id": "FAILED", "trace_kind": "metal", **trace}
    _assign_global_trace_node_indices([case])
    node = bucket["no_metal_trace"]["trace_nodes"][0]
    assert node["global_node_locator"] == "FAILED:N000000"

    html = _render_html_browser_report({"case_count": 1, "cases": [case]})
    assert "no_metal_reconstruction_none" in html
    assert "reject_no_metal_candidate_validation" in html


def test_oxidative_addition_valence_deltas_require_same_sites_and_plus_or_minus_two() -> None:
    selected = SimpleNamespace(metal_states=(_metal_state(1, 2), _metal_state(2, 1)))

    assert _oxidative_addition_valence_deltas(
        selected,
        SimpleNamespace(metal_states=(_metal_state(1, 4), _metal_state(2, 1))),
    ) == {1: 2, 2: 0}
    assert _oxidative_addition_valence_deltas(
        selected,
        SimpleNamespace(metal_states=(_metal_state(1, 0), _metal_state(2, 1))),
    ) == {1: -2, 2: 0}
    assert (
        _oxidative_addition_valence_deltas(
            selected,
            SimpleNamespace(metal_states=(_metal_state(1, 3), _metal_state(2, 1))),
        )
        is None
    )
    assert (
        _oxidative_addition_valence_deltas(
            selected,
            SimpleNamespace(metal_states=(_metal_state(1, 4), _metal_state(3, 1))),
        )
        is None
    )


def _charged_carbon_candidate(charge: int, metal_x: float) -> SimpleNamespace:
    mol = pybel.readstring(
        "xyz",
        """1
C
C 0.0 0.0 0.0
""",
    )
    mol.OBMol.GetAtom(1).SetFormalCharge(charge)
    return SimpleNamespace(
        no_metal_state=SimpleNamespace(omol=mol),
        metal_states=(MetalAtomPosition(2, "Fe", 26, 2, 0, metal_x, 0.0, 0.0),),
    )


def test_additional_negative_charge_must_be_visible_to_metal() -> None:
    candidate = _charged_carbon_candidate(0, 1.8)

    visible = _additional_negative_charge_visibility(
        _charged_carbon_candidate(-1, 1.8),
        candidate,
        config=None,
    )
    assert visible["required"] is True
    assert visible["passed"] is True
    assert visible["visible_additional_negative_charge_atom_indices"] == [1]

    invisible = _additional_negative_charge_visibility(
        _charged_carbon_candidate(-1, 10.0),
        candidate,
        config=None,
    )
    assert invisible["required"] is True
    assert invisible["passed"] is False
    assert invisible["invisible_additional_negative_charge_atom_indices"] == [1]


def test_equal_negative_charge_does_not_require_metal_visibility() -> None:
    result = _additional_negative_charge_visibility(
        _charged_carbon_candidate(-1, 10.0),
        _charged_carbon_candidate(-1, 10.0),
        config=None,
    )

    assert result["required"] is False
    assert result["passed"] is True


def test_metal_selection_basis_explains_filter_and_lexicographic_rejection() -> None:
    selected_key = [0, 0, 0, 0, 0, 0, 0.0, 0.0, 0, 0.0, 0, 10.0, 0]
    candidates = [
        {
            "candidate_index": 0,
            "combination_index": 0,
            "search_layer_index": 0,
            "selected": True,
            "in_production_selection_layer": True,
            "metal_states": [{"symbol": "Fe", "valence": 2, "radical_num": 0}],
            "score": 10.0,
            "score_details": {
                "metal_assignment_rank": 0.0,
                "metal_discordance_count": 0,
                "passes_metal_discordance_filter": True,
                "organic_charge_localization_penalty": 1.0,
                "organic_charge_localization_margin_exceeded": False,
                "selection_key": selected_key,
            },
        },
        {
            "candidate_index": 1,
            "combination_index": 1,
            "search_layer_index": 0,
            "selected": False,
            "in_production_selection_layer": True,
            "metal_states": [{"symbol": "Fe", "valence": 3, "radical_num": 0}],
            "score": 9.0,
            "score_details": {
                "metal_assignment_rank": 0.0,
                "metal_discordance_count": 2,
                "passes_metal_discordance_filter": False,
            },
        },
        {
            "candidate_index": 2,
            "combination_index": 2,
            "search_layer_index": 0,
            "selected": False,
            "in_production_selection_layer": True,
            "metal_states": [{"symbol": "Fe", "valence": 1, "radical_num": 0}],
            "score": 8.0,
            "score_details": {
                "metal_assignment_rank": 10.0,
                "metal_discordance_count": 0,
                "passes_metal_discordance_filter": True,
                "organic_charge_localization_penalty": 2.0,
                "organic_charge_localization_margin_exceeded": True,
                "selection_key": [0, 1, 0, 0, 0, 0, 0.0, 0.0, 0, 0.0, 0, 8.0, 2],
            },
        },
    ]

    basis = _metal_selection_basis(candidates, selected_layer_index=0)

    assert basis["minimum_metal_discordance_count"] == 0
    assert basis["selected_selection_key"] == selected_key
    decisions = basis["candidate_decisions"]
    assert decisions[1]["decision"] == "rejected_by_discordance_filter"
    assert decisions[1]["decisive_field"] == "metal_discordance_count"
    assert decisions[2]["decision"] == "rejected_by_lexicographic_key"
    assert decisions[2]["decisive_field"] == "organic_charge_localization_margin_exceeded"
    assert basis["assignment_rank_policy"]["participates_in_final_selection_key"] is False


def test_nitrosyl_selection_evidence_survives_trace_export() -> None:
    from scripts.reconstruction_trace import _score_details

    evidence = {
        "nitrosyl_geometry_penalty": 0.5,
        "nitrosyl_geometry_site_count": 1,
        "nitrosonium_redistribution_exception": 0,
    }
    assert _score_details(evidence) == evidence


def test_review_fixture_accepts_any_approved_answer() -> None:
    input_case = TraceInputCase(
        id="BOTH",
        xyz_block="",
        total_charge=0,
        total_radical_electrons=0,
        fixture_kind="accepted_both",
        expected_smiles="C",
        expected_smiles_options=("C", "N"),
    )
    trace = {
        "trace_kind": "metal",
        "selected_candidate": {"graph": {"canonical_smiles": "N"}},
    }

    result = _review_fixture_trace_check(input_case, trace)

    assert result["equivalent"] is True
    assert result["matched_answer_index"] == 1


def test_resonance_selection_report_compares_every_scored_candidate() -> None:
    recorder = OmolTraceRecorder()
    for node_id, score in enumerate((1.0, 2.0)):
        recorder.records[node_id] = {
            "phase": "accept_no_metal_candidate",
            "parent_id": -1,
            "omol": pybel.readstring("smi", "C"),
            "given_charge": 0,
            "total_charge": 0,
            "total_radical_electrons": 0,
            "event": {"kind": "checkpoint", "score": score},
            "metadata": {
                "resonance_seed_index": 0,
                "resonance_index": node_id,
                "resonance_raw_index": node_id,
                "resonance_normalization": "full_resonance_normalization",
            },
        }
    trace_nodes = [
        {"node_id": node_id, "dof_image": {"render_id": f"dof-{node_id}"}} for node_id in range(2)
    ]

    report = _resonance_selection_report(
        recorder,
        selected_ids={0},
        trace_nodes=trace_nodes,
        target_charge=0,
        target_radical_electrons=0,
    )

    assert report["candidate_count"] == 2
    assert report["selected_candidate_index"] == 0
    assert report["candidates"][0]["selected"] is True
    assert report["candidates"][0]["decision"] == "selected_lexicographic_minimum"
    assert report["candidates"][1]["decision"] == "rejected_by_selection_key"
    assert report["candidates"][1]["decisive_field"] == "score"
    assert report["candidates"][1]["dof_image"] == {"render_id": "dof-1"}
    assert [field["key"] for field in report["selection_key_fields"]] == [
        "organic_azinium_geometry_conflict_count",
        "organic_formal_charge_absolute_sum",
        "organic_long_carbonyl_multiple_bond_count",
        "organic_aromatic_atom_count",
        "organic_aromatic_ring_count",
        "organic_aromatic_stability_score",
        "organic_adjusted_max_conjugated_component_size",
        "organic_adjusted_conjugated_atom_count",
        "organic_adjusted_conjugated_bond_count",
        "organic_excess_radical_labels",
        "organic_hyperconjugation_score",
        "score",
    ]


def _deferred_context(tmp_path: Path) -> DofRenderContext:
    return DofRenderContext(
        image_dir=tmp_path,
        display_base_dir=None,
        defer_images=True,
        max_images=None,
    )


def test_unlimited_dof_render_context_never_skips_images(tmp_path: Path) -> None:
    context = _deferred_context(tmp_path)

    slots = [
        _reserve_dof_render_slot(render_context=context, label=str(index), kind="state")
        for index in range(1_250)
    ]

    assert slots == list(range(1_250))
    assert context.image_count == 1_250
    assert context.skipped_count == 0


def test_deferred_dof_sdf_is_deduplicated_and_rendered_on_activation(tmp_path: Path) -> None:
    context = _deferred_context(tmp_path)
    mol = Chem.MolFromSmiles("C")
    assert mol is not None
    image = _deferred_dof_record(
        [mol],
        render_context=context,
        label="state",
        kind="no_metal_trace_node",
        render_type="single",
        legends=["state"],
    )
    cases = [
        {
            "id": "CASE",
            "trace_kind": "no_metal",
            "no_metal_trace": {
                "target": {},
                "selected_candidate": {"state": {"dof_image": image}},
                "pipeline_steps": [{"dof_image": image}],
            },
        }
    ]
    normalized_cases, payloads = _extract_deferred_dof_payloads(cases)
    output = {
        "case_count": 1,
        "cases": normalized_cases,
        "dof_payloads": payloads,
        "dof_rendering": dof_rendering_summary(context),
    }

    report = _render_html_browser_report(output)

    assert len(payloads) == 1
    assert report.count("$$$$") == 1
    assert 'fetch("/api/render-dof"' in report
    assert "void renderDeferredDof(node, activePanel)" in report
    assert "detailRoot.replaceChildren(renderPanel(node))" in report
    assert 'lazyDetails("完整 JSON", node.metadata)' in report
    assert 'id="image-lightbox"' in report
    assert 'id="language-toggle"' in report
    assert 'localStorage.setItem("moleculeReviewLanguage", language)' in report
    assert "function renderApplication(activeId" in report
    assert 'src="https://3Dmol.csb.pitt.edu/build/3Dmol-min.js"' in report
    assert ".dof-visual-row" in report
    assert "function renderDofMol3d" in report
    assert "function syncDofVisualHeight" in report
    assert "function openImageLightbox" in report
    assert "setImageZoomState(box, node.label)" in report
    assert "showModal()" in report
    assert 'label: "金属价态候选"' in report
    assert 'kind: "金属价态选择依据"' not in report
    assert "node.metadata.selection_basis" in report
    assert "function renderMetalSelectionBasis" in report
    assert "function renderResonanceSelectionBasis" in report
    assert "function renderDeferredResonanceGrids" in report
    assert 'kind: "共振候选对比"' in report
    assert "resonanceBasis: candidateResonanceBasis" in report
    assert 'node.kind === "金属电子态候选" && node.resonanceBasis' in report
    assert "该金属价态对应有机目标的共振候选" in report
    assert 'render_type: "grid"' in report
    assert "sdfs: items.map(item => item.payload.sdf)" in report
    assert 'details("共振候选逐指标横向对比"' in report
    assert 'details("候选选择结论"' in report
    assert 'details("生产层候选逐项对比"' in report
    assert "@media (max-width:1400px)" in report
    assert "@media (max-width:720px)" in report
    assert ".global-info, main { grid-template-columns:1fr; }" not in report
    assert "main { grid-template-columns:300px minmax(0,1fr);" in report
    assert "html { width:100%; overflow-x:auto; }" in report
    assert "body { margin:0; width:100%; min-width:1000px;" in report
    assert "align-self:start; position:relative;" in report
    assert ".tree { flex:1 1 0; height:0; min-height:0; overflow:auto;" in report
    assert ".content { align-self:start; min-height:0; overflow:auto;" in report
    assert report.count("scrollbar-gutter:stable") == 2
    assert "function syncMainColumnHeights()" in report
    assert "window.visualViewport?.height || window.innerHeight" in report
    assert "viewportHeight - stickyHeaderHeight - 24" in report
    assert "main.style.height = height;" in report
    assert "sidebar.style.height = height;" in report
    assert "content.style.height = height;" in report
    assert "ResizeObserver" not in report
    assert 'window.addEventListener("resize", scheduleMainColumnHeightSync)' in report
    assert (
        'window.visualViewport?.addEventListener("resize", scheduleMainColumnHeightSync)' in report
    )
    assert "max-height:calc(100vh" not in report
    assert ".wide-table { width:100%; min-width:0; max-width:100%; overflow-x:auto" in report
    assert "table-layout:fixed" in report
    assert 'window.matchMedia("(max-width: 720px)")' in report
    assert "activePanel.scrollIntoView" in report
    assert "<svg" not in report
