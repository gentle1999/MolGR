# pyright: reportMissingImports=false

from __future__ import annotations

from pathlib import Path

import pytest


pytest.importorskip("openbabel")

from molgr import _core as core
from molgr.interface import xyz_to_rdmol
from molgr.utils.equivalence import check_equivalence
from scripts.reconstruction_trace import (
    DofRenderContext,
    TraceInputCase,
    _html_no_metal_trace,
    _render_html_browser_report,
    trace_reconstruction_case,
)
from scripts.tmqmg_fixture_data import reference_graph


def _adoqep_xyz_block() -> str:
    return (Path(__file__).parent / "data" / "xyz" / "ADOQEP.xyz").read_text(encoding="utf-8")


def test_adoqep_cpp_molecule_data_is_closed_shell() -> None:
    mol_data = core.pipeline.reconstruct_with_metals.xyz2omol(
        _adoqep_xyz_block(),
        1,
        0,
    )

    assert mol_data is not None
    assert mol_data.total_radical_num == 0
    assert [
        (atom_idx, atom.atomic_num, atom.formal_charge, atom.radical_num)
        for atom_idx, atom in enumerate(mol_data.atoms, start=1)
        if atom.radical_num
    ] == []


def test_adoqep_cpp_matches_reference_under_nonchiral_tmqmg_equivalence() -> None:
    reference = reference_graph("ADOQEP")
    assert reference is not None

    reconstructed = xyz_to_rdmol(
        _adoqep_xyz_block(),
        1,
        1,
        backend="cpp",
        make_dative_bonds=True,
    )

    assert (
        sum(
            atom.GetNumRadicalElectrons()  # pyright: ignore[reportCallIssue]
            for atom in reconstructed.GetAtoms()  # pyright: ignore[reportCallIssue]
        )
        == 0
    )
    equivalent, info = check_equivalence(
        reference,
        reconstructed,
        use_chirality=False,
        ignore_metal_coordination_bonds=True,
        max_resonance=100000,
    )
    assert equivalent is True, info.reason


def test_adoqep_trace_uses_production_no_metal_phase_history() -> None:
    render_context = DofRenderContext(
        image_dir=Path("molgr_trace_dof_images"),
        display_base_dir=None,
        max_images=1000,
    )
    trace = trace_reconstruction_case(
        TraceInputCase(
            id="ADOQEP",
            xyz_block=_adoqep_xyz_block(),
            total_charge=1,
            total_radical_electrons=0,
        ),
        score_all_candidates=False,
        render_context=render_context,
    )

    buckets = [
        bucket
        for layer in trace.get("search", {}).get("layer_summaries", [])
        for bucket in layer.get("target_buckets", [])
    ]
    assert buckets
    for bucket in buckets:
        no_metal_trace = bucket.get("no_metal_trace") or {}
        assert no_metal_trace.get("status") != "trace_error"
        assert no_metal_trace.get("status") == "selected"
        phases = [step["phase"] for step in no_metal_trace.get("pipeline_steps", [])]
        pipeline_steps = no_metal_trace.get("pipeline_steps", [])
        assert all(step.get("dof_image", {}).get("svg_fragment") for step in pipeline_steps)
        assert "prepare_no_metal_seed" in phases
        assert phases[-1] == "select_best_no_metal_candidate"
        assert not no_metal_trace.get("linear_branches")
        trace_nodes = no_metal_trace.get("trace_nodes", [])
        trace_phases = {node["phase"] for node in trace_nodes}
        assert no_metal_trace.get("trace_node_count") == len(trace_nodes)
        assert all(node.get("dof_image", {}).get("svg_fragment") for node in trace_nodes)
        linear_nodes = [
            node
            for node in trace_nodes
            if node["phase"]
            in {
                "make_connections",
                "pre_clean",
                "fresh_omol_charge_radical_initial",
                "initialize_charge_budget",
                "eliminate_NNN_negative",
            }
        ]
        assert linear_nodes
        assert {node["tree_depth"] for node in linear_nodes} == {0}
        assert "relocate_carbene_radical_for_resonance" in trace_phases
        assert "branch_resonance_candidate" in trace_phases
        assert "full_resonance_normalization" in trace_phases
        resonance_branches = {
            node["node_id"] for node in trace_nodes if node["phase"] == "branch_resonance_candidate"
        }
        assert resonance_branches
        full_nodes = [
            node for node in trace_nodes if node["phase"] == "full_resonance_normalization"
        ]
        assert full_nodes
        nodes_by_id = {node["node_id"]: node for node in trace_nodes}
        first_process_nodes = [
            node
            for node in trace_nodes
            if node["phase"] == "process_resonance_eliminate_1_3_dipole_postive"
        ]
        assert first_process_nodes
        assert all(node["parent_id"] in resonance_branches for node in first_process_nodes)
        assert all(
            nodes_by_id[node["parent_id"]]["phase"] == "process_resonance_clean_resonances"
            for node in full_nodes
        )
        assert all(
            node["expansion"]
            for node in trace_nodes
            if node["phase"] in {"recover_deformed_pi_bonds", "recover_by_breaking_bonds"}
            and node.get("event", {}).get("hit")
        )
        assert {
            "accept_no_metal_candidate",
            "reject_no_metal_candidate_validation",
            "discard_duplicate_processed_resonance_candidate",
        } & trace_phases
        resonance = no_metal_trace.get("resonance") or {}
        assert resonance.get("normalization") == "full_resonance_normalization"

    assert render_context.errors == []

    inspectable_nodes = []
    for layer in trace.get("search", {}).get("layer_summaries", []):
        for bucket in layer.get("target_buckets", []):
            inspectable_nodes.extend(
                node
                for node in (bucket.get("no_metal_trace") or {}).get("trace_nodes", [])
                if isinstance(node, dict)
            )
    inspectable_nodes.extend(
        candidate for candidate in trace.get("candidates", []) if isinstance(candidate, dict)
    )
    global_indices = [node["global_node_index"] for node in inspectable_nodes]
    assert global_indices == list(range(len(global_indices)))
    assert all(node.get("global_node_locator") for node in inspectable_nodes)
    for node in inspectable_nodes:
        parent_index = node.get("global_tree_parent_index")
        if parent_index is not None:
            assert parent_index in global_indices

    static_html = _html_no_metal_trace(buckets[0]["no_metal_trace"])
    assert "生产管线阶段" in static_html
    assert "完整状态机分支树" in static_html
    assert "线性分支" not in static_html

    html = _render_html_browser_report(
        {
            "input": {
                "source": "generic",
                "ids": ["ADOQEP"],
                "total_charge": 1,
                "total_radical_electrons": 0,
            },
            "case_count": 1,
            "cases": [trace],
        }
    )
    assert "tree-toggle" in html
    assert "tree-children" in html
    assert "is-collapsed" in html
    assert "无金属完整 trace" in html
    assert "完整状态机分支树" in html
    assert "无金属线性分支" not in html
