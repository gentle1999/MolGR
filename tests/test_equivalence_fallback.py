# pyright: reportCallIssue=false
from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest
from rdkit import Chem

# Frozen Python-engine regression coverage. The public API is exercised by
# test_equivalence_native.py and the fidelity/electronic/coordination suites.
from molgr.utils import _legacy_equivalence as equivalence


@pytest.mark.parametrize(
    ("multiple_bond", "octet_form"),
    [
        ("CP(=O)(C)C", "C[P+](C)(C)[O-]"),
        ("CS(=O)(=O)C", "C[S+2](C)([O-])[O-]"),
    ],
)
def test_equivalence_normalizes_supported_hypervalent_nonmetals_to_octets(
    multiple_bond: str,
    octet_form: str,
) -> None:
    mol1 = Chem.MolFromSmiles(multiple_bond)
    mol2 = Chem.MolFromSmiles(octet_form)
    assert mol1 is not None
    assert mol2 is not None

    equivalent, _ = equivalence.check_equivalence(mol1, mol2, use_chirality=False)

    assert equivalent is True


def test_equivalence_accepts_resonance_forms_after_standardization() -> None:
    mol1 = Chem.MolFromSmiles("CC1=[N+]=C(OC=[N-])[N]N1")
    mol2 = Chem.MolFromSmiles("Cc1nc(OC=[N])n[nH]1")
    assert mol1 is not None
    assert mol2 is not None

    equivalent, info = equivalence.check_equivalence(mol1, mol2, use_chirality=False)

    assert equivalent is True
    assert info.method == equivalence.EquivalenceMethod.RESONANCE


def test_open_shell_forms_without_explicit_resonance_witness_are_inconclusive() -> None:
    mol1 = Chem.MolFromSmiles("[NH3+]CC1=N[N-][N]C1=CO")
    mol2 = Chem.MolFromSmiles("[NH3+]Cc1[n-]nnc1[CH]O")
    assert mol1 is not None
    assert mol2 is not None

    equivalent, info = equivalence.check_equivalence(mol1, mol2, use_chirality=False)

    assert equivalent is False
    assert info.decision == equivalence.EquivalenceDecision.INCONCLUSIVE
    assert info.open_shell_heuristic_triggered is True
    assert info.open_shell_three_center is None


def test_equivalence_rejects_constitutional_isomers_with_same_formula() -> None:
    ethanol = Chem.MolFromSmiles("CCO")
    dimethyl_ether = Chem.MolFromSmiles("COC")
    assert ethanol is not None
    assert dimethyl_ether is not None

    equivalent, info = equivalence.check_equivalence(
        ethanol,
        dimethyl_ether,
        use_chirality=False,
    )

    assert equivalent is False
    assert info.method is None
    assert info.reason == "Not equivalent: non-metal connectivity differs."


def test_equivalence_rejects_charge_transfer_between_components() -> None:
    mol1 = Chem.MolFromSmiles("[OH+].[SH-]")
    mol2 = Chem.MolFromSmiles("[OH-].[SH+]")
    assert mol1 is not None
    assert mol2 is not None

    equivalent, info = equivalence.check_equivalence(mol1, mol2, use_chirality=False)

    assert equivalent is False
    assert info.method is None
    assert info.reason == "Not equivalent: no standardized comparison path matched."


def test_resonance_match_compares_the_two_generated_sets() -> None:
    mol1 = Chem.MolFromSmiles("CC1=[N+]=C(OC=[N-])[N]N1")
    mol2 = Chem.MolFromSmiles("Cc1nc(OC=[N])n[nH]1")
    assert mol1 is not None
    assert mol2 is not None
    standardized_1 = equivalence._standardize_metal_bonds(mol1)
    standardized_2 = equivalence._standardize_metal_bonds(mol2)
    organic_1 = equivalence._prepare_organic_mol(standardized_1, already_standardized=True)
    organic_2 = equivalence._prepare_organic_mol(standardized_2, already_standardized=True)

    matched, mol1_count, mol2_count, hit_smiles = equivalence._resonance_match(
        organic_1,
        organic_2,
        use_chirality=False,
        max_resonance=50,
        resonance_flags=Chem.ResonanceFlags.UNCONSTRAINED_CATIONS
        | Chem.ResonanceFlags.UNCONSTRAINED_ANIONS,
    )

    assert matched is True
    assert mol1_count > 0
    assert mol2_count > 0
    assert hit_smiles is not None


def test_resonance_match_stops_before_enumerating_second_side_on_direct_hit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    implicit_mol = Chem.MolFromSmiles("C")
    assert implicit_mol is not None
    mol = Chem.AddHs(implicit_mol)
    original = equivalence._prepare_resonance_source
    calls = 0

    def tracked_prepare(source: Chem.Mol, **kwargs) -> Chem.Mol:
        nonlocal calls
        calls += 1
        return original(source, **kwargs)

    monkeypatch.setattr(equivalence, "_prepare_resonance_source", tracked_prepare)
    matched, mol1_count, mol2_count, hit_smiles = equivalence._resonance_match(
        mol,
        mol,
        use_chirality=False,
        max_resonance=50,
        resonance_flags=Chem.ResonanceFlags.UNCONSTRAINED_CATIONS
        | Chem.ResonanceFlags.UNCONSTRAINED_ANIONS,
    )

    assert (matched, mol1_count, mol2_count, hit_smiles) == (
        True,
        1,
        0,
        "[H]C([H])([H])[H]",
    )
    assert calls == 1


def test_resonance_supplier_includes_generic_anion_charge_shift() -> None:
    equivalence._cached_resonance_smiles.cache_clear()

    forms = equivalence._enumerate_resonance_forms(
        Chem.MolFromSmiles("CC=C[O-]"),
        use_chirality=False,
        max_resonance=1,
        resonance_flags=int(Chem.ResonanceFlags.UNCONSTRAINED_ANIONS),
    )

    assert forms
    assert any("C(=O)[C-]" in form for form in forms)
    equivalence._cached_resonance_smiles.cache_clear()


def test_resonance_charge_shift_survives_special_normalization() -> None:
    # The two fragments differ by two local migrations:
    # [C-]-N=N-C(=S) -> C=N-[N-]-C(=S) -> C=N-N=C([S-]).
    candidate = Chem.MolFromSmiles("S=C(N)N=N[CH-]c1ccccc1.[S-]C(=NN=Cc1ccccc1)N")
    reference = Chem.MolFromSmiles("[S-]C(=NN=Cc1ccccc1)N.[S-]C(=NN=Cc1ccccc1)N")
    assert candidate is not None
    assert reference is not None

    equivalent, info = equivalence.check_equivalence(
        candidate,
        reference,
        use_chirality=False,
    )

    assert equivalent is True
    assert info.method == equivalence.EquivalenceMethod.RESONANCE


@pytest.mark.parametrize("reverse", [False, True])
def test_equivalence_rejects_total_charge_erased_by_resonance_normalization(
    reverse: bool,
) -> None:
    neutral = Chem.MolFromSmiles("[S]C(=NN=Cc1ccccc1)N")
    anion = Chem.MolFromSmiles("[S-]C(=NN=Cc1ccccc1)N")
    assert neutral is not None
    assert anion is not None
    mol1, mol2 = (anion, neutral) if reverse else (neutral, anion)

    result = equivalence.evaluate_equivalence(mol1, mol2, use_chirality=False)

    assert result.decision == equivalence.EquivalenceDecision.NOT_EQUIVALENT
    assert result.equivalent is False
    assert result.relation == equivalence.EquivalenceRelation.NONE
    assert result.method is None
    charge = result.invariants["raw_total_formal_charge"]
    assert charge.status == equivalence.InvariantStatus.FAILED
    assert (charge.mol1, charge.mol2) == ((-1, 0) if reverse else (0, -1))


def test_resonance_timeout_is_not_swallowed(monkeypatch: pytest.MonkeyPatch) -> None:
    class TimeoutSupplier:
        def __init__(self, *args, **kwargs) -> None:  # noqa: ANN002, ANN003
            pass

        def __iter__(self):
            raise TimeoutError("resonance timeout")

    mol1 = Chem.MolFromSmiles("CCO")
    mol2 = Chem.MolFromSmiles("COC")
    assert mol1 is not None
    assert mol2 is not None
    monkeypatch.setattr(equivalence, "ResonanceMolSupplier", TimeoutSupplier)

    with pytest.raises(TimeoutError, match="resonance timeout"):
        equivalence._resonance_match(
            mol1,
            mol2,
            use_chirality=False,
            max_resonance=50,
            resonance_flags=Chem.ResonanceFlags.UNCONSTRAINED_CATIONS
            | Chem.ResonanceFlags.UNCONSTRAINED_ANIONS,
        )


def test_resonance_sets_are_cached(monkeypatch: pytest.MonkeyPatch) -> None:
    real_supplier = equivalence.ResonanceMolSupplier
    supplier_calls = 0

    def counting_supplier(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
        nonlocal supplier_calls
        supplier_calls += 1
        return real_supplier(*args, **kwargs)

    mol1 = Chem.MolFromSmiles("CCO")
    mol2 = Chem.MolFromSmiles("COC")
    assert mol1 is not None
    assert mol2 is not None
    equivalence._cached_resonance_smiles.cache_clear()
    monkeypatch.setattr(equivalence, "ResonanceMolSupplier", counting_supplier)

    for _ in range(2):
        for source in ("CCO", "COC"):
            equivalence._cached_resonance_smiles(
                source,
                use_chirality=False,
                max_resonance=50,
                resonance_flags=int(
                    Chem.ResonanceFlags.UNCONSTRAINED_CATIONS
                    | Chem.ResonanceFlags.UNCONSTRAINED_ANIONS
                ),
            )

    assert supplier_calls == 2
    equivalence._cached_resonance_smiles.cache_clear()


@pytest.mark.parametrize("failure_stage", ["prepare", "supplier_init", "supplier_iter"])
def test_resonance_search_recovers_after_temporary_failure(
    monkeypatch: pytest.MonkeyPatch,
    failure_stage: str,
) -> None:
    mol1 = Chem.MolFromSmiles("CC1=[N+]=C(OC=[N-])[N]N1")
    mol2 = Chem.MolFromSmiles("Cc1nc(OC=[N])n[nH]1")
    assert mol1 is not None
    assert mol2 is not None
    equivalence._cached_resonance_smiles.cache_clear()
    real_supplier = equivalence.ResonanceMolSupplier

    def fail(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
        raise RuntimeError("temporary resonance failure")

    def failing_forms(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
        yield next(iter(real_supplier(*args, **kwargs)))
        raise RuntimeError("temporary resonance failure")

    try:
        with monkeypatch.context() as patch:
            if failure_stage == "prepare":
                patch.setattr(equivalence, "_prepare_resonance_source", fail)
            else:
                patch.setattr(
                    equivalence,
                    "ResonanceMolSupplier",
                    failing_forms if failure_stage == "supplier_iter" else fail,
                )
            failed = equivalence.evaluate_equivalence(mol1, mol2, use_chirality=False)
        assert failed.decision == equivalence.EquivalenceDecision.INCONCLUSIVE
        assert failed.bounded_search is not None
        assert failed.bounded_search.exhaustive is False
        recovered = equivalence.evaluate_equivalence(mol1, mol2, use_chirality=False)
        assert recovered.decision == equivalence.EquivalenceDecision.EQUIVALENT
        assert recovered.method == equivalence.EquivalenceMethod.RESONANCE
    finally:
        equivalence._cached_resonance_smiles.cache_clear()


def test_failed_resonance_search_cannot_establish_non_equivalence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mol1 = Chem.MolFromSmiles("[OH+].[SH-]")
    mol2 = Chem.MolFromSmiles("[OH-].[SH+]")
    assert mol1 is not None
    assert mol2 is not None

    def fail(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
        raise RuntimeError("temporary resonance failure")

    with monkeypatch.context() as patch:
        patch.setattr(equivalence, "_prepare_resonance_source", fail)
        result = equivalence.evaluate_equivalence(mol1, mol2, use_chirality=False)
    assert result.decision == equivalence.EquivalenceDecision.INCONCLUSIVE
    assert result.bounded_search is not None
    assert result.bounded_search.exhaustive is False
    recovered = equivalence.evaluate_equivalence(mol1, mol2, use_chirality=False)
    assert recovered.decision == equivalence.EquivalenceDecision.NOT_EQUIVALENT


def test_resonance_hit_is_stable_across_hash_seeds() -> None:
    script = """
import json
from rdkit import Chem
from molgr.utils._legacy_equivalence import evaluate_equivalence
r = evaluate_equivalence(
    Chem.MolFromSmiles("CC1=[N+]=C(OC=[N-])[N]N1"),
    Chem.MolFromSmiles("Cc1nc(OC=[N])n[nH]1"),
    use_chirality=False,
)
assert r.equivalent and r.resonance is not None
print(json.dumps(r.resonance.hit_smiles))
"""
    hits = []
    for seed in ("1", "2"):
        result = subprocess.run(
            [sys.executable, "-B", "-c", script],
            env={**os.environ, "PYTHONHASHSEED": seed},
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        hits.append(json.loads(result.stdout))
    assert hits[0] == hits[1]


def test_equivalence_rejects_metal_valence_mismatch() -> None:
    mol1 = Chem.MolFromSmiles("C[Cu]N")
    mol2 = Chem.MolFromSmiles("C->[Cu+]<-N")
    assert mol1 is not None
    assert mol2 is not None

    equivalent, info = equivalence.check_equivalence(mol1, mol2, use_chirality=False)

    assert equivalent is False
    assert info.method is None
    assert info.reason == "Not equivalent: metal valence assignment differs."


def test_metal_double_bond_becomes_carbene_after_metal_removal() -> None:
    mol = Chem.MolFromSmiles("[CH2]=[Fe]")
    assert mol is not None

    standardized = equivalence._standardize_metal_bonds(mol)
    organic = equivalence._prepare_organic_mol(standardized, already_standardized=True)
    carbon = organic.GetAtomWithIdx(0)

    assert carbon.GetAtomicNum() == 6
    assert carbon.GetFormalCharge() == 0
    assert carbon.GetNumRadicalElectrons() == 2


def test_equivalence_normalizes_metal_double_bond_to_explicit_carbene_complex() -> None:
    metal_double_bond = Chem.MolFromSmiles("[CH2]=[Fe]")
    carbene_complex = Chem.MolFromSmiles("[CH2]->[Fe]")
    assert metal_double_bond is not None
    assert carbene_complex is not None

    equivalent, info = equivalence.check_equivalence(
        metal_double_bond,
        carbene_complex,
        use_chirality=False,
    )

    assert equivalent is True
    assert info.method == equivalence.EquivalenceMethod.IDEAL


def test_equivalence_normalizes_aqowuy_metal_double_bond_to_carbene_coordination() -> None:
    reference = Chem.MolFromSmiles(
        "COc1ccc(C=C[C](OC)->[Rh+]234(<-[C-]#[O+])<-C5=C->2CCC->3=C->4CC5)cc1"
    )
    candidate = Chem.MolFromSmiles(
        "COC(C=Cc1ccc(OC)cc1)=[Rh+]123(<-[C-]#[O+])<-C4=C->1CCC->2=C->3CC4"
    )
    assert reference is not None
    assert candidate is not None

    equivalent, info = equivalence.check_equivalence(
        reference,
        candidate,
        use_chirality=False,
    )

    assert equivalent is True
    assert info.method == equivalence.EquivalenceMethod.IDEAL


def test_equivalence_ignores_redundant_metal_radicals() -> None:
    mol1 = Chem.MolFromSmiles("C->[Ru+]<-N", sanitize=False)
    mol2 = Chem.MolFromSmiles("C->[Ru+]<-N", sanitize=False)
    assert mol1 is not None
    assert mol2 is not None
    ru_atom = next(atom for atom in mol1.GetAtoms() if atom.GetSymbol() == "Ru")
    ru_atom.SetNumRadicalElectrons(1)
    mol1.UpdatePropertyCache(strict=False)
    mol2.UpdatePropertyCache(strict=False)

    equivalent, info = equivalence.check_equivalence(mol1, mol2, use_chirality=False)

    assert equivalent is True
    assert info.method == equivalence.EquivalenceMethod.IDEAL


def test_equivalence_treats_metals_as_isolated_ions() -> None:
    mol1 = Chem.MolFromSmiles("C->[Ru+]<-N", sanitize=False)
    mol2 = Chem.MolFromSmiles("C.[Ru+].N", sanitize=False)
    assert mol1 is not None
    assert mol2 is not None
    mol1.UpdatePropertyCache(strict=False)
    mol2.UpdatePropertyCache(strict=False)

    equivalent, info = equivalence.check_equivalence(mol1, mol2, use_chirality=False)

    assert equivalent is True
    assert info.method == equivalence.EquivalenceMethod.IDEAL


def test_equivalence_suppresses_rdkit_warnings_for_metal_hydrides(capfd) -> None:
    mol1 = Chem.MolFromSmiles("C->[IrH+2]<-N", sanitize=False)
    mol2 = Chem.MolFromSmiles("C->[Ir+2]([H])<-N", sanitize=False)
    assert mol1 is not None
    assert mol2 is not None
    mol1.UpdatePropertyCache(strict=False)
    mol2.UpdatePropertyCache(strict=False)

    equivalent, info = equivalence.check_equivalence(mol1, mol2, use_chirality=False)

    captured = capfd.readouterr()
    assert equivalent is True
    assert info.method == equivalence.EquivalenceMethod.IDEAL
    assert captured.err == ""


def test_equivalence_rejects_hydrogen_count_mismatch_after_expansion() -> None:
    mol1 = Chem.MolFromSmiles("C=C")
    mol2 = Chem.MolFromSmiles("CC")
    assert mol1 is not None
    assert mol2 is not None

    equivalent, info = equivalence.check_equivalence(mol1, mol2, use_chirality=False)

    assert equivalent is False
    assert info.method is None
    assert info.checks is not None
    assert info.checks.heavy_atom_formula.passed is True
    assert info.checks.explicit_h_formula.passed is False
    assert info.reason == "Not equivalent: explicit-hydrogen element counts differ."


def test_equivalence_rejects_stereoisomers_with_same_inchi_connectivity() -> None:
    mol1 = Chem.MolFromSmiles("C/C=C/C")
    mol2 = Chem.MolFromSmiles("C/C=C\\C")
    assert mol1 is not None
    assert mol2 is not None

    equivalent, info = equivalence.check_equivalence(mol1, mol2)

    assert equivalent is False
    assert info.method is None
    assert info.reason == "Not equivalent: stereochemistry differs."


def test_equivalence_rejects_local_charge_mismatch_after_standardization() -> None:
    mol1 = Chem.MolFromSmiles("CC(C)(C)[N+]#[C-]->[Cu+]1<-[O-]C=C2C=CC=N->12")
    mol2 = Chem.MolFromSmiles("CC(C)(C)[N][C][Cu]1[O][CH][C]2[CH][CH][CH][N]21")
    assert mol1 is not None
    assert mol2 is not None

    equivalent, info = equivalence.check_equivalence(mol1, mol2, use_chirality=False)

    assert equivalent is False
    assert info.method is None
    assert info.reason


def test_structured_result_records_normalized_graph_identity() -> None:
    mol1 = Chem.MolFromSmiles("CCO")
    mol2 = Chem.MolFromSmiles("CCO")
    assert mol1 is not None
    assert mol2 is not None

    result = equivalence.evaluate_equivalence(mol1, mol2, use_chirality=False)

    assert result.decision == equivalence.EquivalenceDecision.EQUIVALENT
    assert result.relation == equivalence.EquivalenceRelation.NORMALIZED_GRAPH_IDENTITY
    assert result.method == equivalence.EquivalenceMethod.IDEAL
    assert result.invariants["nonmetal_connectivity"].status == equivalence.InvariantStatus.PASSED
    assert result.contradictions == []


def test_inchi_respects_disabled_chirality() -> None:
    trans = Chem.MolFromSmiles("C/C=C/C")
    cis = Chem.MolFromSmiles("C/C=C\\C")
    assert trans is not None
    assert cis is not None

    assert equivalence._inchi_key(trans, use_chirality=True) != equivalence._inchi_key(
        cis, use_chirality=True
    )
    assert equivalence._inchi_key(trans, use_chirality=False) == equivalence._inchi_key(
        cis, use_chirality=False
    )


def test_inchi_ignores_3d_stereo_only_when_chirality_disabled() -> None:
    mol = Chem.MolFromSmiles("C/C=C/C")
    assert mol is not None
    conformer = Chem.Conformer(mol.GetNumAtoms())
    conformer.Set3D(True)
    for idx, position in enumerate(
        ((-1.0, 1.0, 0.0), (0.0, 0.0, 0.0), (1.3, 0.0, 0.0), (2.3, -1.0, 0.2))
    ):
        conformer.SetAtomPosition(idx, position)
    mol.AddConformer(conformer)
    Chem.RemoveStereochemistry(mol)
    coordinate_free = Chem.Mol(mol)
    coordinate_free.RemoveAllConformers()

    enabled_key = equivalence._inchi_key(mol, use_chirality=True)
    coordinate_free_key = equivalence._inchi_key(coordinate_free, use_chirality=True)
    assert enabled_key is not None
    assert coordinate_free_key is not None
    assert enabled_key != coordinate_free_key
    assert equivalence._inchi_key(mol, use_chirality=False) == coordinate_free_key
    assert mol.GetNumConformers() == 1
    assert mol.GetConformer().Is3D()
    assert equivalence._inchi_key(mol, use_chirality=True) == enabled_key


def test_normalized_identifier_match_does_not_override_component_electron_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mol1 = Chem.MolFromSmiles("[OH+].[SH-]")
    mol2 = Chem.MolFromSmiles("[OH-].[SH+]")
    assert mol1 is not None
    assert mol2 is not None
    keys = iter((None, None, "normalized-key", "normalized-key"))
    monkeypatch.setattr(equivalence, "_inchi_key", lambda *args, **kwargs: next(keys))

    result = equivalence.evaluate_equivalence(mol1, mol2, use_chirality=False)

    assert result.decision == equivalence.EquivalenceDecision.INCONCLUSIVE
    assert result.relation == equivalence.EquivalenceRelation.IDENTIFIER_EQUIVALENCE
    assert result.method == equivalence.EquivalenceMethod.INCHI_KEY
    assert result.invariants["component_electrons"].status == equivalence.InvariantStatus.FAILED
    assert result.contradictions == ["identifier_match_despite_component_electron_mismatch"]


def test_carbene_match_overrides_component_electron_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mol1 = Chem.MolFromSmiles("[OH+].[SH-]")
    mol2 = Chem.MolFromSmiles("[OH-].[SH+]")
    assert mol1 is not None
    assert mol2 is not None
    monkeypatch.setattr(equivalence, "_inchi_key", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        equivalence,
        "_carbene_zwitterion_normalized_smiles",
        lambda *args, **kwargs: "same-normalized-form",
    )

    result = equivalence.evaluate_equivalence(mol1, mol2, use_chirality=False)

    assert result.decision == equivalence.EquivalenceDecision.EQUIVALENT
    assert result.relation == equivalence.EquivalenceRelation.CARBENE_ZWITTERION_EQUIVALENCE
    assert result.method == equivalence.EquivalenceMethod.CARBENE_ZWITTERION
    assert result.invariants["component_electrons"].status == equivalence.InvariantStatus.FAILED
    assert result.contradictions == [
        "carbene_normalization_match_despite_component_electron_mismatch"
    ]


def test_normalized_identifier_match_is_inconclusive_without_structural_witness(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mol1 = Chem.MolFromSmiles("[OH+].[SH-]")
    mol2 = Chem.MolFromSmiles("[OH-].[SH+]")
    assert mol1 is not None
    assert mol2 is not None
    keys = iter((None, None, "normalized-key", "normalized-key"))
    monkeypatch.setattr(equivalence, "_inchi_key", lambda *args, **kwargs: next(keys))
    monkeypatch.setattr(
        equivalence,
        "_component_electron_signature",
        lambda *args, **kwargs: (("same-topology", 0, 0),),
    )
    monkeypatch.setattr(
        equivalence,
        "_resonance_match",
        lambda *args, **kwargs: equivalence._ResonanceMatchResult(False, 1, 1, None),
    )
    carbene_forms = iter(("form-1", "form-2"))
    monkeypatch.setattr(
        equivalence,
        "_carbene_zwitterion_normalized_smiles",
        lambda *args, **kwargs: next(carbene_forms),
    )
    monkeypatch.setattr(equivalence, "_total_radical_electrons", lambda *args, **kwargs: 0)

    result = equivalence.evaluate_equivalence(mol1, mol2, use_chirality=False)

    assert result.decision == equivalence.EquivalenceDecision.INCONCLUSIVE
    assert result.relation == equivalence.EquivalenceRelation.IDENTIFIER_EQUIVALENCE
    assert result.method == equivalence.EquivalenceMethod.INCHI_KEY


def test_both_identifier_sources_match_is_still_inconclusive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mol1 = Chem.MolFromSmiles("[OH+].[SH-]")
    mol2 = Chem.MolFromSmiles("[OH-].[SH+]")
    assert mol1 is not None
    assert mol2 is not None
    keys = iter(("prepared-key", "prepared-key", "normalized-key", "normalized-key"))
    monkeypatch.setattr(equivalence, "_inchi_key", lambda *args, **kwargs: next(keys))

    result = equivalence.evaluate_equivalence(mol1, mol2, use_chirality=False)

    assert result.decision == equivalence.EquivalenceDecision.INCONCLUSIVE
    assert result.relation == equivalence.EquivalenceRelation.IDENTIFIER_EQUIVALENCE
    assert result.method == equivalence.EquivalenceMethod.INCHI_KEY


def test_component_electron_mismatch_without_positive_evidence_remains_not_equivalent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mol1 = Chem.MolFromSmiles("[OH+].[SH-]")
    mol2 = Chem.MolFromSmiles("[OH-].[SH+]")
    assert mol1 is not None
    assert mol2 is not None
    monkeypatch.setattr(equivalence, "_inchi_key", lambda *args, **kwargs: None)

    result = equivalence.evaluate_equivalence(mol1, mol2, use_chirality=False)

    assert result.decision == equivalence.EquivalenceDecision.NOT_EQUIVALENT
    assert result.relation == equivalence.EquivalenceRelation.NONE
    assert result.method is None
    assert result.reason == "Not equivalent: no standardized comparison path matched."
    assert result.invariants["component_electrons"].status == equivalence.InvariantStatus.FAILED


def test_resonance_match_overrides_component_electron_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mol1 = Chem.MolFromSmiles("S=C(N)N=N[CH-]c1ccccc1.[S-]C(=NN=Cc1ccccc1)N")
    mol2 = Chem.MolFromSmiles("[S-]C(=NN=Cc1ccccc1)N.[S-]C(=NN=Cc1ccccc1)N")
    assert mol1 is not None
    assert mol2 is not None
    monkeypatch.setattr(equivalence, "_inchi_key", lambda *args, **kwargs: None)
    signatures = iter(
        (
            (("same-topology", -1, 0), ("same-topology", -1, 0)),
            (("same-topology", -1, 0), ("same-topology", -1, 1)),
        )
    )
    monkeypatch.setattr(
        equivalence,
        "_component_electron_signature",
        lambda *args, **kwargs: next(signatures),
    )

    result = equivalence.evaluate_equivalence(mol1, mol2, use_chirality=False)

    assert result.decision == equivalence.EquivalenceDecision.EQUIVALENT
    assert result.relation == equivalence.EquivalenceRelation.RESONANCE_EQUIVALENCE
    assert result.method == equivalence.EquivalenceMethod.RESONANCE
    assert result.invariants["component_electrons"].status == equivalence.InvariantStatus.FAILED


def test_atom_count_mismatch_remains_a_hard_veto(monkeypatch: pytest.MonkeyPatch) -> None:
    mol1 = Chem.MolFromSmiles("C")
    mol2 = Chem.MolFromSmiles("CC")
    assert mol1 is not None
    assert mol2 is not None
    formula_keys = iter(
        ("same-heavy-formula", "same-heavy-formula", "same-h-formula", "same-h-formula")
    )
    monkeypatch.setattr(equivalence, "_formula_key", lambda *args, **kwargs: next(formula_keys))

    result = equivalence.evaluate_equivalence(mol1, mol2, use_chirality=False)

    assert result.decision == equivalence.EquivalenceDecision.NOT_EQUIVALENT
    assert result.reason == "Not equivalent: explicit-hydrogen atom counts differ."
    assert result.invariants["atom_count"].status == equivalence.InvariantStatus.FAILED


def test_normalized_graph_identity_implies_weaker_invariants(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mol1 = Chem.MolFromSmiles("CCO")
    mol2 = Chem.MolFromSmiles("CCO")
    assert mol1 is not None
    assert mol2 is not None

    def redundant_check(*args, **kwargs):
        raise AssertionError("Graph identity already proves this invariant")

    monkeypatch.setattr(equivalence, "_resonance_topology_key", redundant_check)
    monkeypatch.setattr(equivalence, "_component_electron_signature", redundant_check)
    result = equivalence.evaluate_equivalence(mol1, mol2, use_chirality=False)

    assert result.decision == equivalence.EquivalenceDecision.EQUIVALENT
    assert result.relation == equivalence.EquivalenceRelation.NORMALIZED_GRAPH_IDENTITY
    for name in ("nonmetal_connectivity", "component_electrons", "stereochemistry"):
        invariant = result.invariants[name]
        assert invariant.status == equivalence.InvariantStatus.PASSED
        assert invariant.mol1 is None and invariant.mol2 is None
        assert "Implied" in invariant.reason
    assert result.contradictions == []


def test_normalized_graph_identity_skips_identifier_and_carbene_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mol1 = Chem.MolFromSmiles("CCO")
    mol2 = Chem.MolFromSmiles("CCO")
    assert mol1 is not None
    assert mol2 is not None

    def unexpected_call(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
        raise AssertionError("an exact graph match needs no fallback normalization")

    monkeypatch.setattr(equivalence, "_inchi_key", unexpected_call)
    monkeypatch.setattr(equivalence, "_carbene_zwitterion_normalized_smiles", unexpected_call)

    result = equivalence.evaluate_equivalence(mol1, mol2, use_chirality=False)

    assert result.decision == equivalence.EquivalenceDecision.EQUIVALENT
    assert result.method == equivalence.EquivalenceMethod.IDEAL
    assert result.carbene_zwitterion is None


@pytest.mark.parametrize(
    ("smiles_1", "smiles_2", "invariant"),
    [
        ("C", "CC", "heavy_atom_formula"),
        ("C=C", "CC", "explicit_h_formula"),
        ("C", "[CH3-]", "explicit_h_formula"),
        ("[CH3-]", "[CH3]", "raw_total_formal_charge"),
        ("C[Cu]", "C->[Cu]", "metal_state"),
        ("CCO", "COC", "nonmetal_connectivity"),
    ],
)
def test_hard_structural_invariants_remain_vetoes(
    smiles_1: str,
    smiles_2: str,
    invariant: str,
) -> None:
    mol1 = Chem.MolFromSmiles(smiles_1)
    mol2 = Chem.MolFromSmiles(smiles_2)
    assert mol1 is not None
    assert mol2 is not None

    result = equivalence.evaluate_equivalence(mol1, mol2, use_chirality=False)

    assert result.decision == equivalence.EquivalenceDecision.NOT_EQUIVALENT
    assert result.invariants[invariant].status == equivalence.InvariantStatus.FAILED


def test_prepared_identifier_only_remains_inconclusive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mol1 = Chem.MolFromSmiles("[OH+].[SH-]")
    mol2 = Chem.MolFromSmiles("[OH-].[SH+]")
    assert mol1 is not None
    assert mol2 is not None
    keys = iter(("prepared-key", "prepared-key", None, None))
    monkeypatch.setattr(equivalence, "_inchi_key", lambda *args, **kwargs: next(keys))

    result = equivalence.evaluate_equivalence(mol1, mol2, use_chirality=False)

    assert result.decision == equivalence.EquivalenceDecision.INCONCLUSIVE
    assert result.relation == equivalence.EquivalenceRelation.IDENTIFIER_EQUIVALENCE
    assert result.method == equivalence.EquivalenceMethod.INCHI_KEY
    assert result.invariants["component_electrons"].status == equivalence.InvariantStatus.FAILED


@pytest.mark.parametrize("search_fails", [False, True])
def test_open_shell_explicit_three_center_witness_is_equivalent(
    monkeypatch: pytest.MonkeyPatch,
    search_fails: bool,
) -> None:
    candidate = Chem.MolFromSmiles("[CH2]C=CO")
    reference = Chem.MolFromSmiles("C=C[CH]O")
    assert candidate is not None
    assert reference is not None
    monkeypatch.setattr(equivalence, "_inchi_key", lambda *args, **kwargs: None)

    def resonance_search(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
        if search_fails:
            raise RuntimeError("temporary resonance failure")
        return equivalence._ResonanceMatchResult(False, 1, 1, None)

    monkeypatch.setattr(equivalence, "_resonance_match", resonance_search)

    result = equivalence.evaluate_equivalence(candidate, reference, use_chirality=False)

    assert result.decision == equivalence.EquivalenceDecision.EQUIVALENT
    assert result.method == equivalence.EquivalenceMethod.OPEN_SHELL_THREE_CENTER
    assert result.open_shell_three_center is not None
    assert sorted(result.open_shell_three_center.mol1_radical_electrons) == [0, 0, 1]
    assert sorted(result.open_shell_three_center.mol2_radical_electrons) == [0, 0, 1]
    assert result.open_shell_three_center_mapping_search is not None
    assert result.open_shell_three_center_mapping_search.limit_reached is False
    assert result.bounded_search is not None
    assert result.bounded_search.exhaustive is not search_fails


def test_open_shell_without_witness_remains_inconclusive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate = Chem.MolFromSmiles("[CH2]C=CO")
    reference = Chem.MolFromSmiles("C=C[CH]O")
    assert candidate is not None
    assert reference is not None
    monkeypatch.setattr(equivalence, "_inchi_key", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        equivalence,
        "_resonance_match",
        lambda *args, **kwargs: equivalence._ResonanceMatchResult(False, 1, 1, None),
    )
    monkeypatch.setattr(
        equivalence,
        "_elementary_open_shell_resonance_witness",
        lambda *args, **kwargs: (
            None,
            equivalence.TopologyMappingSearchMetadata(),
        ),
    )

    result = equivalence.evaluate_equivalence(candidate, reference, use_chirality=False)

    assert result.open_shell_heuristic_triggered is True
    assert result.decision == equivalence.EquivalenceDecision.INCONCLUSIVE
    assert "no explicit resonance witness" in result.reason


def test_open_shell_mapping_bound_remains_inconclusive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate = Chem.MolFromSmiles("[CH2]C=CO")
    reference = Chem.MolFromSmiles("C=C[CH]O")
    assert candidate is not None
    assert reference is not None
    monkeypatch.setattr(equivalence, "_inchi_key", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        equivalence,
        "_resonance_match",
        lambda *args, **kwargs: equivalence._ResonanceMatchResult(False, 1, 1, None),
    )
    monkeypatch.setattr(
        equivalence,
        "_elementary_open_shell_resonance_witness",
        lambda *args, **kwargs: (
            None,
            equivalence.TopologyMappingSearchMetadata(
                attempted=True,
                mappings_examined=10000,
                limit_reached=True,
            ),
        ),
    )

    result = equivalence.evaluate_equivalence(candidate, reference, use_chirality=False)

    assert result.decision == equivalence.EquivalenceDecision.INCONCLUSIVE
    assert result.open_shell_three_center_mapping_search is not None
    assert result.open_shell_three_center_mapping_search.limit_reached is True
    assert "topology mapping search reached its limit" in result.reason


@pytest.mark.parametrize(
    ("candidate_smiles", "reference_smiles"),
    [
        ("C=C1[CH]CCC=C1", "[CH2]C1=CCCC=C1"),
        ("[O]C=CO", "O=C[CH]O"),
        ("C=C([N]O)C(C)=O", "[CH2]/C(=N\\O)C(C)=O"),
        ("[H]N=C1[CH]C=CCO1", "[NH]C1=CC=CCO1"),
        ("[H]N=C([N]C)N(C)C", "C/N=C(/[NH])N(C)C"),
    ],
)
def test_elementary_open_shell_three_center_witness_is_explicit(
    candidate_smiles: str,
    reference_smiles: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate = Chem.MolFromSmiles(candidate_smiles)
    reference = Chem.MolFromSmiles(reference_smiles)
    assert candidate is not None
    assert reference is not None

    # Identifier evidence is intentionally absent; the positive proof is the
    # mapped electron shift itself.
    monkeypatch.setattr(equivalence, "_inchi_key", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        equivalence,
        "_resonance_match",
        lambda *args, **kwargs: equivalence._ResonanceMatchResult(False, 1, 1, None),
    )

    result = equivalence.evaluate_equivalence(candidate, reference, use_chirality=False)

    assert result.decision == equivalence.EquivalenceDecision.EQUIVALENT
    assert result.relation == equivalence.EquivalenceRelation.RESONANCE_EQUIVALENCE
    assert result.method == equivalence.EquivalenceMethod.OPEN_SHELL_THREE_CENTER
    assert result.open_shell_three_center is not None
    assert result.open_shell_three_center.mol1_bond_orders in {(1.0, 2.0), (2.0, 1.0)}
    assert result.open_shell_three_center.mol2_bond_orders in {(1.0, 2.0), (2.0, 1.0)}
    assert result.open_shell_three_center.mol1_bond_orders != (
        result.open_shell_three_center.mol2_bond_orders
    )
    assert sorted(result.open_shell_three_center.mol1_radical_electrons) == [0, 0, 1]
    assert sorted(result.open_shell_three_center.mol2_radical_electrons) == [0, 0, 1]
    assert result.open_shell_three_center_mapping_search is not None
    assert result.open_shell_three_center_mapping_search.mappings_examined >= 1
    assert result.open_shell_three_center_mapping_search.limit_reached is False
    assert result.bounded_search is not None
    assert result.bounded_search.attempted is True


def test_nonconjugated_radical_cation_relocation_is_not_equivalent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate = Chem.MolFromSmiles("C[N+](C)CCCN(CC)CC")
    reference = Chem.MolFromSmiles("CN(C)CCC[N+](CC)CC")
    assert candidate is not None
    assert reference is not None
    monkeypatch.setattr(equivalence, "_inchi_key", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        equivalence,
        "_resonance_match",
        lambda *args, **kwargs: equivalence._ResonanceMatchResult(False, 1, 1, None),
    )

    result = equivalence.evaluate_equivalence(candidate, reference, use_chirality=False)

    assert result.open_shell_heuristic_triggered is True
    assert result.decision == equivalence.EquivalenceDecision.INCONCLUSIVE
    assert result.equivalent is False


def test_saturated_chain_radical_relocation_has_no_elementary_witness(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate = Chem.MolFromSmiles("O[CH]CCCN")
    reference = Chem.MolFromSmiles("OCC[CH]CN")
    assert candidate is not None
    assert reference is not None
    monkeypatch.setattr(equivalence, "_inchi_key", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        equivalence,
        "_resonance_match",
        lambda *args, **kwargs: equivalence._ResonanceMatchResult(False, 1, 1, None),
    )

    result = equivalence.evaluate_equivalence(candidate, reference, use_chirality=False)

    assert result.decision == equivalence.EquivalenceDecision.NOT_EQUIVALENT
    assert result.method is None
    assert result.open_shell_three_center is None


def test_radical_shift_with_unrelated_charge_redistribution_has_no_elementary_witness(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate = Chem.MolFromSmiles("C=[NH+]CC[C]([O-])OC")
    reference = Chem.MolFromSmiles("[CH2]NCCC(=O)OC")
    assert candidate is not None
    assert reference is not None
    monkeypatch.setattr(equivalence, "_inchi_key", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        equivalence,
        "_resonance_match",
        lambda *args, **kwargs: equivalence._ResonanceMatchResult(False, 1, 1, None),
    )

    result = equivalence.evaluate_equivalence(candidate, reference, use_chirality=False)

    assert result.decision != equivalence.EquivalenceDecision.EQUIVALENT
    assert result.open_shell_three_center is None


def test_aromatic_noninteger_delta_has_no_elementary_witness(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate = Chem.MolFromSmiles("[O]c1ccco1")
    reference = Chem.MolFromSmiles("O=C1C=C[CH]O1")
    assert candidate is not None
    assert reference is not None
    monkeypatch.setattr(equivalence, "_inchi_key", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        equivalence,
        "_resonance_match",
        lambda *args, **kwargs: equivalence._ResonanceMatchResult(False, 1, 1, None),
    )

    result = equivalence.evaluate_equivalence(candidate, reference, use_chirality=False)

    assert result.decision == equivalence.EquivalenceDecision.INCONCLUSIVE
    assert result.open_shell_three_center is None


def test_open_shell_explicit_resonance_witness_is_equivalent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate = Chem.MolFromSmiles("[CH2]C=CO")
    reference = Chem.MolFromSmiles("C=C[CH]O")
    assert candidate is not None
    assert reference is not None
    calls = 0

    def explicit_resonance_witness(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
        nonlocal calls
        calls += 1
        return equivalence._ResonanceMatchResult(True, 2, 2, "[CH2]C=CO")

    monkeypatch.setattr(equivalence, "_resonance_match", explicit_resonance_witness)
    monkeypatch.setattr(
        equivalence,
        "_elementary_open_shell_resonance_witness",
        lambda *args, **kwargs: (None, equivalence.TopologyMappingSearchMetadata()),
    )

    result = equivalence.evaluate_equivalence(candidate, reference, use_chirality=False)

    assert calls == 1
    assert result.open_shell_heuristic_triggered is True
    assert result.decision == equivalence.EquivalenceDecision.EQUIVALENT
    assert result.relation == equivalence.EquivalenceRelation.RESONANCE_EQUIVALENCE
    assert result.method == equivalence.EquivalenceMethod.RESONANCE


def test_bounded_resonance_limit_is_inconclusive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mol1 = Chem.MolFromSmiles("S=C(N)N=N[CH-]c1ccccc1.[S-]C(=NN=Cc1ccccc1)N")
    mol2 = Chem.MolFromSmiles("[S-]C(=NN=Cc1ccccc1)N.[S-]C(=NN=Cc1ccccc1)N")
    assert mol1 is not None
    assert mol2 is not None
    monkeypatch.setattr(equivalence, "_inchi_key", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        equivalence,
        "_resonance_match",
        lambda *args, **kwargs: equivalence._ResonanceMatchResult(
            False, 10, 10, None, local_limit_reached=True
        ),
    )

    result = equivalence.evaluate_equivalence(
        mol1,
        mol2,
        use_chirality=False,
        max_resonance=10,
    )
    compatible, wrapper_result = equivalence.check_equivalence(
        mol1,
        mol2,
        use_chirality=False,
        max_resonance=10,
    )

    assert result.decision == equivalence.EquivalenceDecision.INCONCLUSIVE
    assert result.bounded_search is not None
    assert result.bounded_search.limit_reached is True
    assert result.bounded_search.exhaustive is False
    assert compatible is False
    assert wrapper_result.decision == equivalence.EquivalenceDecision.INCONCLUSIVE


@pytest.mark.parametrize(
    ("left", "right"),
    [
        ("C.C.C.C.C.C.C.C", "C.C.C.C.C.C.C.C"),
        ("[CH3].C.C.C.C.C.C.C", "[CH2].C.C.C.C.C.C.C"),
        ("[CH3].C.C.C.C.C.C.C", "C.C.C.C.C.C.C.C"),
        ("C.C.C.C.C.C.C.C", "[CH3].C.C.C.C.C.C.C"),
    ],
)
def test_impossible_radical_shift_does_not_exhaust_mapping_budget(
    left: str,
    right: str,
) -> None:
    mol1 = Chem.MolFromSmiles(left)
    mol2 = Chem.MolFromSmiles(right)
    assert mol1 is not None
    assert mol2 is not None

    witness, search = equivalence._elementary_open_shell_resonance_witness(
        mol1, mol2, use_chirality=False
    )

    assert witness is None
    assert search.attempted is False
    assert search.mappings_examined == 0
    assert search.limit_reached is False
