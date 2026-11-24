"""Check actual widening control flow, budget bounds, and early success."""

from dataclasses import asdict
from types import SimpleNamespace

import pytest

from molgr.config import MolGRConfig
from molgr.fallback.pipeline import reconstruct_without_metals as pipeline


@pytest.mark.parametrize("depth", [0, 1, 2, 3, 4])
@pytest.mark.parametrize("budget", [0, 1, 2, 3])
def test_widening_obeys_requested_limits_and_preserves_original_search(monkeypatch, depth, budget):
    cfg = MolGRConfig()
    cfg.resonance.max_depth = depth
    cfg.resonance.limited_discrepancy_max_discrepancy = budget
    before = asdict(cfg)
    seed = object()
    calls = []

    def primary(prepared, *, max_discrepancy, traversal_policy, config):
        assert (
            prepared is seed and max_discrepancy == cfg.resonance.neighbor_radical_max_discrepancy
        )
        calls.append(("primary", config.resonance.max_depth, traversal_policy.max_discrepancy))
        return SimpleNamespace(attempted_resonance_seeds=[seed], validated_candidates=[])

    def search(seeds, *, resonance_traversal_policy, config):
        assert seeds == [seed]
        calls.append(
            ("search", config.resonance.max_depth, resonance_traversal_policy.max_discrepancy)
        )
        return []

    def recovery(seeds, *, traversal_policy, config):
        assert seeds == [seed]
        calls.append(("recovery", config.resonance.max_depth, traversal_policy.max_discrepancy))
        return []

    monkeypatch.setattr(pipeline, "_search_discrepancy_layers", primary)
    monkeypatch.setattr(pipeline.no_metal_resonance, "search_resonance_candidates", search)
    monkeypatch.setattr(pipeline, "_search_recovery_tiers", recovery)
    assert pipeline._find_no_metal_candidates(seed, config=cfg) == []
    expected = [("primary", min(2, depth), min(1, budget))]
    if depth > 2:
        expected.append(("search", depth, min(1, budget)))
    if budget > 1:
        expected.append(("search", depth, budget))
    expected.append(("recovery", depth, budget))
    assert calls == expected
    assert all(d <= depth and b <= budget for _, d, b in calls)
    assert asdict(cfg) == before


@pytest.mark.parametrize("success", ["primary", "deeper", "wider"])
def test_first_success_stops_widening_without_replacing_candidates(monkeypatch, success):
    cfg = MolGRConfig()
    cfg.resonance.max_depth = 3
    cfg.resonance.limited_discrepancy_max_discrepancy = 2
    candidate = object()
    seed = object()
    calls = []

    def primary(*args, **kwargs):
        calls.append("primary")
        return SimpleNamespace(
            attempted_resonance_seeds=[seed],
            validated_candidates=[candidate] if success == "primary" else [],
        )

    def search(seeds, *, resonance_traversal_policy, config):
        name = "deeper" if resonance_traversal_policy.max_discrepancy == 1 else "wider"
        calls.append(name)
        return [candidate] if success == name else []

    def recovery(*args, **kwargs):
        raise AssertionError("No destructive recovery after successful ordinary reconstruction")

    monkeypatch.setattr(pipeline, "_search_discrepancy_layers", primary)
    monkeypatch.setattr(pipeline.no_metal_resonance, "search_resonance_candidates", search)
    monkeypatch.setattr(pipeline, "_search_recovery_tiers", recovery)
    assert pipeline._find_no_metal_candidates(seed, config=cfg) == [candidate]
    assert (
        calls == ["primary", "deeper", "wider"][: ["primary", "deeper", "wider"].index(success) + 1]
    )
