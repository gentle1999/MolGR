# pyright: reportCallIssue=false
from __future__ import annotations

import pytest
from rdkit import Chem

from molgr.utils import equivalence


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
    assert info.reason == "Not equivalent: charge or radical count differs within a component."


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


def test_resonance_supplier_includes_generic_anion_charge_shift() -> None:
    equivalence._cached_resonance_smiles.cache_clear()

    forms = equivalence._cached_resonance_smiles(
        "CC=C[O-]",
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
        equivalence._resonance_match(
            mol1,
            mol2,
            use_chirality=False,
            max_resonance=50,
            resonance_flags=Chem.ResonanceFlags.UNCONSTRAINED_CATIONS
            | Chem.ResonanceFlags.UNCONSTRAINED_ANIONS,
        )

    assert supplier_calls == 2
    equivalence._cached_resonance_smiles.cache_clear()


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


def test_identifier_match_cannot_override_component_electron_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mol1 = Chem.MolFromSmiles("[OH+].[SH-]")
    mol2 = Chem.MolFromSmiles("[OH-].[SH+]")
    assert mol1 is not None
    assert mol2 is not None
    monkeypatch.setattr(equivalence, "_inchi_key", lambda *args, **kwargs: "same-key")

    result = equivalence.evaluate_equivalence(mol1, mol2, use_chirality=False)

    assert result.decision == equivalence.EquivalenceDecision.NOT_EQUIVALENT
    assert result.relation == equivalence.EquivalenceRelation.NONE
    assert result.method is None
    assert result.invariants["component_electrons"].status == equivalence.InvariantStatus.FAILED
    assert result.contradictions == ["identifier_match_despite_component_electron_mismatch"]


def test_identifier_only_evidence_is_inconclusive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate = Chem.MolFromSmiles("S=C(N)N=N[CH-]c1ccccc1.[S-]C(=NN=Cc1ccccc1)N")
    reference = Chem.MolFromSmiles("[S-]C(=NN=Cc1ccccc1)N.[S-]C(=NN=Cc1ccccc1)N")
    assert candidate is not None
    assert reference is not None
    monkeypatch.setattr(equivalence, "_inchi_key", lambda *args, **kwargs: "same-key")
    monkeypatch.setattr(
        equivalence,
        "_resonance_match",
        lambda *args, **kwargs: (False, 1, 1, None),
    )

    result = equivalence.evaluate_equivalence(
        candidate,
        reference,
        use_chirality=False,
        max_resonance=10,
    )

    assert result.decision == equivalence.EquivalenceDecision.INCONCLUSIVE
    assert result.relation == equivalence.EquivalenceRelation.IDENTIFIER_EQUIVALENCE
    assert result.method == equivalence.EquivalenceMethod.INCHI_KEY
    assert result.bounded_search is not None
    assert result.bounded_search.attempted is True
    assert result.bounded_search.limit_reached is False


def test_open_shell_heuristic_without_resonance_witness_is_inconclusive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate = Chem.MolFromSmiles("[CH2]C=CO")
    reference = Chem.MolFromSmiles("C=C[CH]O")
    assert candidate is not None
    assert reference is not None
    calls = 0

    def no_resonance_witness(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
        nonlocal calls
        calls += 1
        return False, 2, 1, None

    monkeypatch.setattr(equivalence, "_inchi_key", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        equivalence,
        "_elementary_open_shell_resonance_witness",
        lambda *args, **kwargs: (None, equivalence.TopologyMappingSearchMetadata()),
    )
    monkeypatch.setattr(equivalence, "_resonance_match", no_resonance_witness)

    result = equivalence.evaluate_equivalence(candidate, reference, use_chirality=False)

    assert calls == 1
    assert result.open_shell_heuristic_triggered is True
    assert result.decision == equivalence.EquivalenceDecision.INCONCLUSIVE
    assert result.relation == equivalence.EquivalenceRelation.NONE
    assert "no explicit resonance witness" in result.reason


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
        lambda *args, **kwargs: (False, 1, 1, None),
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
        lambda *args, **kwargs: (False, 1, 1, None),
    )

    result = equivalence.evaluate_equivalence(candidate, reference, use_chirality=False)

    assert result.open_shell_heuristic_triggered is True
    assert result.decision == equivalence.EquivalenceDecision.INCONCLUSIVE
    assert result.equivalent is False


def test_elementary_mapping_search_exhaustion_is_inconclusive(
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
        lambda *args, **kwargs: (False, 1, 1, None),
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
        lambda *args, **kwargs: (False, 1, 1, None),
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
        lambda *args, **kwargs: (False, 1, 1, None),
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
        lambda *args, **kwargs: (False, 1, 1, None),
    )

    result = equivalence.evaluate_equivalence(candidate, reference, use_chirality=False)

    assert result.decision == equivalence.EquivalenceDecision.INCONCLUSIVE
    assert result.open_shell_three_center is None


def test_unrelated_radical_component_does_not_activate_open_shell_heuristic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate = Chem.MolFromSmiles("CC(=O)CC[C+]([O-])CC.[CH3]")
    reference = Chem.MolFromSmiles("C[C+]([O-])CCC(=O)CC.[CH3]")
    assert candidate is not None
    assert reference is not None
    monkeypatch.setattr(equivalence, "_inchi_key", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        equivalence,
        "_resonance_match",
        lambda *args, **kwargs: (False, 1, 1, None),
    )

    result = equivalence.evaluate_equivalence(candidate, reference, use_chirality=False)

    assert result.open_shell_heuristic_triggered is False
    assert result.decision == equivalence.EquivalenceDecision.NOT_EQUIVALENT
    assert result.equivalent is False


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
        return True, 2, 2, "[CH2]C=CO"

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
        lambda *args, **kwargs: (False, 10, 10, None),
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
