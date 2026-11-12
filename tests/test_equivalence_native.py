"""Native proofs are replayed independently on the explicit-H input graphs."""

import pytest
from rdkit import Chem

from molgr import _core
from molgr.utils import equivalence as eq
from molgr.utils.converter import rdkit_to_mol_data


def _mol(smiles):
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None
    return mol


def _replay(left, right, result):
    """Replay both paths to a meeting state, without the native rule generator.

    These fixtures have no metal or hypervalent preprocessing. The recorded
    carbene convention is explicitly distinguished from electron-conserving
    resonance moves. Atom indices in both paths use the left input's frame.
    """
    inverse = {j: i for i, j in enumerate(result["mapping"])}
    states = []
    for side, data in enumerate((left, right)):
        indices = inverse if side else {i: i for i in range(len(data.atoms))}
        q, r = {}, {}
        for i, atom in enumerate(data.atoms):
            q[indices[i]], r[indices[i]] = atom.formal_charge, atom.radical_num
        bonds = {
            tuple(sorted((indices[b.begin_atom_idx - 1], indices[b.end_atom_idx - 1]))): b.order
            for b in data.bonds
        }
        states.append((q, r, bonds))
    for move in result["witness"]:
        q, r, bonds = states[move["side"]]
        a, b, c = move["atoms"]
        charge, radicals = sum(q.values()), sum(r.values())
        kind = move["kind"]

        def bond(u, v, delta, bonds=bonds):
            key = tuple(sorted((u, v)))
            assert key in bonds
            bonds[key] += delta
            assert 1 <= bonds[key] <= 3

        if kind == "kekule_cycle":
            cycle = move["cycle"]
            assert cycle[0] == cycle[-1] and len(cycle) % 2 == 1
            for i, (u, v) in enumerate(zip(cycle, cycle[1:])):
                bond(u, v, c if i % 2 == 0 else -c)
        elif kind == "carbene_normalization":
            bond(a, b, -1)
            q[a] += 1
            q[b] -= 1
            r[a] = 2
        elif kind in ("polarization", "pair_donation"):
            bond(a, b, -1 if kind == "polarization" else 1)
            q[a] += 1
            q[b] -= 1
        else:
            assert kind in ("electron_pair_shift", "cation_shift", "radical_shift")
            bond(a, b, 1)
            bond(b, c, -1)
            if kind == "radical_shift":
                r[a] -= 1
                r[c] += 1
                assert r[a] >= 0
            else:
                delta = 1 if kind == "electron_pair_shift" else -1
                q[a] += delta
                q[c] -= delta
        assert sum(q.values()) == charge
        if kind != "carbene_normalization":
            assert sum(r.values()) == radicals
    assert states[0] == states[1]


PAIRS = [
    ("[CH2-]C=CC=CC", "C=CC=C[CH-]C"),
    ("[CH2+]C=CC=CC", "C=CC=C[CH+]C"),
    ("[CH2]C=CC=CC", "C=CC=C[CH]C"),
    ("CC(=O)N", "CC([O-])=[NH2+]"),
    ("C[N+](=[18O])[O-]", "C[N+]([18O-])=O"),
    ("N#C[CH2-]", "[N-]=C=C"),
    ("CS(C)=O", "C[S+](C)[O-]"),
    ("[C]1N(C)C=CN1C", "[c-]1[n+](C)ccn1C"),
    ("[NH3+]CC1=N[N-][N]C1=CO", "[NH3+]Cc1[n-]nnc1[CH]O"),
    ("COC(=O)[CH-]C(=O)OC", "COC(=O)C=C([O-])OC"),
]


@pytest.mark.parametrize("smiles1,smiles2", PAIRS)
@pytest.mark.parametrize("reverse", [False, True])
def test_native_witness_reaches_same_state(smiles1, smiles2, reverse):
    left, right = (rdkit_to_mol_data(_mol(s)) for s in (smiles1, smiles2))
    if reverse:
        left, right = right, left
    result = _core.equivalence.evaluate(left, right, use_chirality=False, max_resonance=200)
    assert result["decision"] == "equivalent", result
    assert result["witness"]
    _replay(left, right, result)


def test_public_entry_never_serializes_or_calls_legacy(monkeypatch):
    from molgr.utils import _legacy_equivalence as legacy

    def forbidden(*args, **kwargs):
        raise AssertionError("Native evaluation must not call the Python search/serializer")

    left, right = _mol(PAIRS[0][0]), _mol(PAIRS[0][1])
    monkeypatch.setattr(Chem, "MolToSmiles", forbidden)
    monkeypatch.setattr(Chem, "ResonanceMolSupplier", forbidden)
    monkeypatch.setattr(legacy, "evaluate_equivalence", forbidden)
    assert eq.evaluate_equivalence(left, right).equivalent


@pytest.mark.parametrize(
    "options",
    [
        {"max_resonance": 0},
        {"max_resonance": 1},
        {"max_resonance_steps": 1},
        {"max_mappings": 0},
        {"max_mapping_nodes": 1},
    ],
)
def test_bounded_search_never_claims_inequivalence(options):
    left, right = (_mol(s) for s in PAIRS[0])
    result = eq.evaluate_equivalence(left, right, **options)
    assert result.decision == eq.EquivalenceDecision.INCONCLUSIVE
    assert not result.resonance_witness
    assert result.bounded_search.limit_reached
    assert result.bounded_search.regional_state_count <= 2 * options.get("max_resonance", 5000)


@pytest.mark.parametrize(
    "smiles1,smiles2",
    [
        ("C[CH]CC", "[CH2]CCC"),
        ("[CH2-]CC=CC", "C=CC[CH-]C"),
    ],
)
def test_no_migration_across_saturated_barrier(smiles1, smiles2):
    result = eq.evaluate_equivalence(_mol(smiles1), _mol(smiles2))
    assert result.decision != eq.EquivalenceDecision.EQUIVALENT


@pytest.mark.parametrize("count", [1, 4, 8])
def test_independent_regions_use_sum_not_product_of_states(count):
    # A saturated methylene separates the amide sites in one connected graph.
    left = _mol("CC(=O)N" + "CC(=O)N" * (count - 1))
    # Terminal N has two explicit hydrogens, internal N has one.
    right = _mol(("CC([O-])=[NH+]" * (count - 1)) + "CC([O-])=[NH2+]")
    a, b = rdkit_to_mol_data(left), rdkit_to_mol_data(right)
    result = _core.equivalence.evaluate(a, b)
    assert result["decision"] == "equivalent", result
    assert result["regions"] == count
    assert result["states_left"] + result["states_right"] <= 4 * count
    _replay(a, b, result)


def test_kekule_cycle_needs_no_search_and_does_not_depend_on_aromatic_flags():
    left = rdkit_to_mol_data(_mol("[13cH]1ccccc1"))
    right = rdkit_to_mol_data(_mol("[13cH]1ccccc1"))
    bonds = right.bonds
    for b in bonds:
        if b.aromatic:
            b.order = 3 - b.order
            b.aromatic = False
    right.bonds = bonds
    result = _core.equivalence.evaluate(left, right, max_resonance=0)
    assert result["decision"] == "equivalent"
    assert result["states_left"] + result["states_right"] == 0
    assert {m["kind"] for m in result["witness"]} == {"kekule_cycle"}
    _replay(left, right, result)


@pytest.mark.parametrize(
    "count,template",
    [
        (3, "F[Pt@SP{n}](Cl)(Br)I"),
        (20, "F[As@TB{n}](Cl)(Br)(I)N"),
        (30, "F[Co@OH{n}](Cl)(Br)(I)(N)P"),
    ],
)
def test_all_native_coordination_stereo_permutations(count, template):
    first = None
    for p in range(1, count + 1):
        mol = _mol(template.format(n=p))
        reordered = Chem.RenumberAtoms(mol, list(reversed(range(mol.GetNumAtoms()))))
        left, right = rdkit_to_mol_data(mol), rdkit_to_mol_data(reordered)
        assert (
            _core.equivalence.evaluate(
                left, right, use_chirality=True, ignore_metal_coordination_bonds=False
            )["decision"]
            == "equivalent"
        )
        if first is None:
            first = left
        else:
            assert (
                _core.equivalence.evaluate(
                    first, right, use_chirality=True, ignore_metal_coordination_bonds=False
                )["decision"]
                == "not_equivalent"
            )


@pytest.mark.parametrize("group", ["&1:1,3", "o1:1,3"])
def test_relative_stereo_requires_consistent_group_inversion(group):
    a = _mol(f"C[C@H](O)[C@@H](F)Cl |{group}|")
    b = _mol(f"C[C@@H](O)[C@H](F)Cl |{group}|")
    c = _mol(f"C[C@@H](O)[C@@H](F)Cl |{group}|")
    assert eq.evaluate_equivalence(a, b, use_chirality=True).equivalent
    assert (
        eq.evaluate_equivalence(a, c, use_chirality=True).decision
        == eq.EquivalenceDecision.NOT_EQUIVALENT
    )


def test_unspecified_stereo_is_inconclusive_against_specified():
    left, right = _mol("FC=CF"), _mol("F/C=C/F")
    assert eq.evaluate_equivalence(left, right).equivalent
    assert eq.check_equivalence(left, right)[0]
    a, b = rdkit_to_mol_data(left), rdkit_to_mol_data(right)
    assert _core.equivalence.evaluate(a, b)["decision"] == "equivalent"
    assert _core.equivalence.evaluate(a, b, use_chirality=True)["decision"] == "inconclusive"
    assert (
        eq.evaluate_equivalence(left, right, use_chirality=True).decision
        == eq.EquivalenceDecision.INCONCLUSIVE
    )


def test_invalid_direct_data_is_inconclusive_and_inputs_are_read_only():
    left, right = rdkit_to_mol_data(_mol("CO")), rdkit_to_mol_data(_mol("CO"))
    right.total_charge = 1
    result = _core.equivalence.evaluate(left, right)
    assert result["decision"] == "inconclusive" and result["stop_reason"] == "error"
    assert left.total_charge == 0 and right.total_charge == 1


def test_unsanitized_radical_perception_preserves_known_zero():
    mol = _mol("C[C]C")
    center = mol.GetAtomWithIdx(1)
    center.SetNumRadicalElectrons(0)
    data, _ = eq._prepare_input(mol, False)
    assert data.atoms[1].radical_num == 2
    center.SetIntProp("MOLGR_UNPAIRED_ELECTRON_COUNT", 0)
    data, _ = eq._prepare_input(mol, False)
    assert data.atoms[1].radical_num == 0
    assert center.GetNumRadicalElectrons() == 0


@pytest.mark.parametrize(
    "left,right",
    [
        ("p1(Cl)(->[Au+])oc2ccccc2c2ccccc2o1", "[p]1(Cl)(->[Au+])oc2ccccc2c2ccccc2o1"),
        ("s1(->[Cu+])cccc1", "[s]1(->[Cu+])cccc1"),
    ],
)
def test_aromatic_bracket_syntax_does_not_create_radicals(left, right):
    a, b = _mol(left), _mol(right)
    for mol in (a, b):
        data, _ = eq._prepare_input(mol, False)
        assert all(atom.radical_num == 0 for atom in data.atoms)
    assert eq.evaluate_equivalence(a, b).equivalent


@pytest.mark.parametrize(
    "left,right",
    [
        ("CCCC", "CC(C)C"),
        ("C1CCCCC1", "C1CC1.C1CC1"),
        ("CC=O", "C=CO"),  # Moving explicit H is not resonance.
    ],
)
def test_sigma_skeleton_mismatch_never_enters_resonance(left, right):
    result = _core.equivalence.evaluate(
        rdkit_to_mol_data(_mol(left)),
        rdkit_to_mol_data(_mol(right)),
        use_chirality=False,
    )
    assert result["decision"] == "not_equivalent", result
    assert result["invariants"]["nonmetal_connectivity"] == -1
    assert not result["resonance_attempted"]
    assert result["states_left"] + result["states_right"] == 0


@pytest.mark.parametrize("missing_stereo", [False, True])
def test_equal_topology_colors_are_not_an_isomorphism_proof(missing_stereo):
    # Both connected graphs are 3-regular on six C atoms (one H per C).
    # The prism contains triangles; K3,3 is bipartite. Color refinement alone
    # cannot distinguish them, so adjacency constraints must reject the pair.
    prism = [(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3), (0, 3), (1, 4), (2, 5)]
    bipartite = [(i, j) for i in range(3) for j in range(3, 6)]

    def graph(edges):
        mol = Chem.RWMol()
        for _ in range(6):
            mol.AddAtom(Chem.Atom(6))
        for u, v in edges:
            mol.AddBond(u, v, Chem.BondType.SINGLE)
        Chem.SanitizeMol(mol)
        return rdkit_to_mol_data(mol)

    left, right = graph(prism), graph(bipartite)
    if missing_stereo:
        record = _core.utils.StereoData()
        record.center, record.configuration = 1, 1
        record.refs = [
            b.end_atom_idx if b.begin_atom_idx == 1 else b.begin_atom_idx
            for b in right.bonds
            if 1 in (b.begin_atom_idx, b.end_atom_idx)
        ]
        right.stereo = [record]
    result = _core.equivalence.evaluate(left, right, use_chirality=missing_stereo)
    assert result["decision"] == "not_equivalent", result
    assert result["mapping_nodes"] > 0
    assert result["mappings"] == 0
    assert not result["resonance_attempted"]


@pytest.mark.parametrize("reverse", [False, True])
def test_component_charge_transfer_is_rejected_before_mapping(reverse):
    inputs = [rdkit_to_mol_data(_mol(s)) for s in ("[CH3-].[NH3+]", "[CH3].[NH3]")]
    if reverse:
        inputs.reverse()
    result = _core.equivalence.evaluate(*inputs, use_chirality=False)
    assert result["decision"] == "not_equivalent", result
    assert result["invariants"]["component_electrons"] == -1
    assert result["mapping_nodes"] == 0
    assert not result["resonance_attempted"]


def test_equal_components_can_swap_to_preserve_charge():
    left, right = [rdkit_to_mol_data(_mol(s)) for s in ("[CH3-].[CH3+]", "[CH3+].[CH3-]")]
    result = _core.equivalence.evaluate(left, right, use_chirality=False, max_mappings=1)
    assert result["decision"] == "equivalent", result
    assert result["method"] == "ideal"
    _replay(left, right, result)


@pytest.mark.parametrize("reverse", [False, True])
def test_annotation_candidates_find_symmetric_witness_with_one_mapping(reverse):
    left, right = _mol("OCCO"), _mol("OCCO")
    for mol, counts in ((left, (0, 0, 0, 1)), (right, (1, 0, 0, 0))):
        for atom, count in zip(mol.GetAtoms(), counts):
            atom.SetIntProp("MOLGR_LONE_PAIR_COUNT", count)
    if reverse:
        left, right = right, left
    result = eq.evaluate_equivalence(left, right, use_chirality=False, max_mappings=1)
    assert result.equivalent, result
    assert result.native_stats["mappings"] == 1


@pytest.mark.parametrize("smiles1,smiles2", PAIRS[:8])
def test_topology_candidates_allow_permuted_resonance_states(smiles1, smiles2):
    left = rdkit_to_mol_data(_mol(smiles1))
    mol = Chem.AddHs(_mol(smiles2))
    # Interleave rather than just reversing, including the explicit H indices.
    order = list(range(mol.GetNumAtoms()))
    right = rdkit_to_mol_data(Chem.RenumberAtoms(mol, order[1::2] + order[::2]))
    result = _core.equivalence.evaluate(left, right, use_chirality=False, max_resonance=200)
    assert result["decision"] == "equivalent", result
    _replay(left, right, result)


def test_radical_inventory_gate_stays_inconclusive():
    left, right = [rdkit_to_mol_data(_mol("[CH2]")) for _ in range(2)]
    atoms = right.atoms
    atoms[0].radical_num = 0
    right.atoms = atoms
    right.total_radical_num = 0
    result = _core.equivalence.evaluate(left, right, use_chirality=False)
    assert result["decision"] == "inconclusive", result
    assert result["mapping_nodes"] == 0
    assert result["stop_reason"] == "rules_exhausted"


def test_resonance_ordering_does_not_prefer_impossible_localized_identity():
    # BIZJEC: local charge agreement favors the longer of two ring alignments.
    # Its different electronic inventories already exclude direct identity.
    left = _mol(
        "Cc1n2-c3nc4cccc[c-]4n3->[Ru+2]34567(<-[I-])"
        "(<-[c-]2c[n+]1C)<-c1c->3c->4(C(C)C)c->5c->6c->71C"
    )
    right = _mol(
        "Cc1n(C)c[c-]2->[Ru+2]34567(<-[I-])"
        "(<-[cH]8[cH]->3[c]->4(C(C)C)[cH]->5[cH]->6[c]->78C)"
        "<-[n-]3c(-[n+]12)nc1ccccc13"
    )
    result = eq.evaluate_equivalence(left, right, use_chirality=False, max_resonance=100)
    assert result.equivalent, result
    assert result.resonance_witness


def test_joint_charge_failure_is_not_reported_as_topology_failure():
    # The same two nonisomorphic regular components occur on both sides, with
    # swapped component charges. Their color inventories collide, so only the
    # joint search reveals that a topology witness cannot also preserve charge.
    prism = [(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3), (0, 3), (1, 4), (2, 5)]
    bipartite = [(i, j) for i in range(3) for j in range(3, 6)]
    mol = Chem.RWMol()
    for _ in range(12):
        mol.AddAtom(Chem.Atom(6))
    for offset, edges in ((0, prism), (6, bipartite)):
        for u, v in edges:
            mol.AddBond(u + offset, v + offset, Chem.BondType.SINGLE)
    Chem.SanitizeMol(mol)
    left, right = [rdkit_to_mol_data(mol) for _ in range(2)]
    for data, charge in ((left, 1), (right, -1)):
        atoms = data.atoms
        atoms[0].formal_charge = charge
        atoms[6].formal_charge = -charge
        data.atoms = atoms
    result = _core.equivalence.evaluate(left, right, use_chirality=False)
    assert result["decision"] == "not_equivalent", result
    assert result["invariants"]["nonmetal_connectivity"] == 0
    assert "charge conservation" in result["reason"]
    assert not result["resonance_attempted"]


def test_all_region_conservation_gates_precede_any_bfs():
    left = rdkit_to_mol_data(_mol("CC(=O)NCC(=O)NCC(=O)N"))
    right = rdkit_to_mol_data(_mol("CC([O-])=[NH+]CC(=O)NCC(=O)N"))
    atoms = right.atoms
    oxygens = [i for i, a in enumerate(atoms) if a.atomic_num == 8]
    # The first amide needs a valid BFS step, but the two later regions would
    # need to exchange charge across a saturated barrier. Whole charge is fixed.
    atoms[oxygens[1]].formal_charge -= 1
    atoms[oxygens[2]].formal_charge += 1
    right.atoms = atoms
    result = _core.equivalence.evaluate(left, right, use_chirality=False)
    assert result["decision"] == "inconclusive", result
    assert result["region_constraint_rejections"] > 0
    assert result["states_left"] + result["states_right"] == 0


def test_symmetric_failed_targets_reuse_inconclusive_results():
    # ASEKEQ: independent ligand/water symmetries produce repeated aligned states.
    left = _mol(
        "O=C(O)c1cc(C(=O)O)cc(-c2nn[n-](->[Cd+2](<-O)(<-O)(<-O)(<-O)"
        "<-[n-]3nnc(-c4cc(C(=O)O)cc(C(=O)O)c4)n3)n2)c1"
    )
    right = _mol(
        "O=C(O)c1cc(C2=[N+]=[N](->[Cd+2](<-[OH2])(<-[OH2])(<-[OH2])(<-[OH2])"
        "<-[n]3nn[c-](-c4cc(C(=O)O)cc(C(=O)O)c4)n3)N=N2)cc([C-]([O-])O)c1"
    )
    result = eq.evaluate_equivalence(left, right, use_chirality=False, max_resonance=50)
    assert result.decision == eq.EquivalenceDecision.INCONCLUSIVE
    assert result.native_stats["resonance_cache_hits"] > 0
    assert result.native_stats["region_cache_hits"] > 0
    assert result.bounded_search.regional_state_count <= 100
    assert not result.resonance_witness


def test_cached_regional_success_replays_and_does_not_leak_between_calls():
    # AJAYIU: a successful region precedes a failed one under the first mapping.
    # Reuse its moves under a different mapping without spending its states twice.
    from molgr.fallback.utils.consts import NON_METAL_DICT

    left = _mol(
        "COc1cc2[N-]3C(=O)c4ccc5ccccc5n4->[Ru+4]<-34(<-[N-]=O)"
        "(<-[O-]c3ccc5nc6ccc(=O)cc-6[se]c5c3)<-n3c(C(=O)[N-]->4c2cc1OC)ccc1ccccc13"
    )
    right = _mol(
        "COc1cc2[N]3=C([O-])c4ccc5ccccc5[n]4->[Ru@OH20+4]<-34(<-[N-]=O)"
        "(<-[O]=c3ccc5nc6ccc([O-])cc6[se]c-5c3)<-[n]3c(C(=O)[N-]->4c2cc1OC)ccc1ccccc13"
    )
    a, b = [eq._prepare_input(m, False)[0] for m in (left, right)]
    result = _core.equivalence.evaluate(
        a, b, use_chirality=False, max_resonance=50, max_resonance_steps=12
    )
    assert result["decision"] == "equivalent", result
    assert result["region_cache_hits"] > 0
    assert result["states_left"] + result["states_right"] <= 100
    limited = _core.equivalence.evaluate(a, b, use_chirality=False, max_resonance=0)
    assert limited["decision"] == "inconclusive"
    assert not limited["witness"]
    # All metal edges here are dative; projection changes no charge/radical
    # bookkeeping. Keep atom indices, and replay the concrete cached moves.
    for data in (a, b):
        atoms = data.atoms
        data.bonds = [
            bond
            for bond in data.bonds
            if atoms[bond.begin_atom_idx - 1].atomic_num in NON_METAL_DICT
            and atoms[bond.end_atom_idx - 1].atomic_num in NON_METAL_DICT
        ]
    _replay(a, b, result)


def test_expanded_defaults_agree_across_public_wrapper_and_native_entry():
    # AGICAU needs more states than the former default permits.
    left = _mol("COc1ccc(N2=C3C=CC=CC3=N3N=c4nccc[n-]4->[Pd+2]<-3<-2<-[Cl-])cc1")
    right = _mol("COc1ccc([N-]2c3ccccc3/[N]3=N/c4nccc[n]4->[Pd@SP1+2]<-3<-2<-[Cl-])cc1")
    limited = eq.evaluate_equivalence(left, right, use_chirality=False, max_resonance=50)
    assert limited.decision == eq.EquivalenceDecision.INCONCLUSIVE
    result = eq.evaluate_equivalence(left, right, use_chirality=False)
    assert result.equivalent, result
    assert result.bounded_search.limit == 5000
    assert result.bounded_search.regional_state_count <= 10000
    a, b = [eq._prepare_input(m, False)[0] for m in (left, right)]
    native = _core.equivalence.evaluate(a, b, use_chirality=False)
    assert native["decision"] == "equivalent", native
    assert eq.check_equivalence(left, right, use_chirality=False)[0]


def _shared_metal_double_bond_frame(configuration):
    data = rdkit_to_mol_data(_mol("F[C]=[C]Cl.[Cu]"))
    bonds = data.bonds
    for center in (2, 3):
        bond = _core.utils.BondData()
        bond.begin_atom_idx, bond.end_atom_idx = center, 5
        bond.kind, bond.order = _core.utils.BondKind.Dative, 0
        bonds.append(bond)
    data.bonds = bonds
    record = _core.utils.StereoData()
    record.kind = _core.utils.StereoKind.DoubleBond
    record.center, record.end = 2, 3
    record.refs, record.configuration = [1, 5, 4, 5], configuration
    data.stereo = [record]
    return data


def test_double_bond_stereo_can_share_a_metal_reference_between_ends():
    a = _shared_metal_double_bond_frame(1)
    b = _shared_metal_double_bond_frame(2)
    assert _core.equivalence.evaluate(a, b, use_chirality=True)["decision"] == "equivalent"
    assert (
        _core.equivalence.evaluate(a, a, use_chirality=True, ignore_metal_coordination_bonds=False)[
            "decision"
        ]
        == "equivalent"
    )
    assert (
        _core.equivalence.evaluate(a, b, use_chirality=True, ignore_metal_coordination_bonds=False)[
            "decision"
        ]
        == "not_equivalent"
    )
    records = b.stereo
    records[0].refs = [1, 1, 4, 5]
    b.stereo = records
    invalid = _core.equivalence.evaluate(a, b, use_chirality=True)
    assert invalid["decision"] == "inconclusive" and invalid["stop_reason"] == "error"


@pytest.mark.parametrize("reverse", [False, True])
def test_missing_stereo_stops_after_one_topology_witness(reverse):
    # ASUTIR formerly exhausted 100,000 mapping nodes just to rediscover that
    # the reference has no stereo records. One topology witness is sufficient.
    left = _mol(
        "CC1(C)C2=N(->[Fe+2](<-[CH2-][Si](C)(C)C)(<-[CH2-][Si](C)(C)C)<-N3=C1OCC3C(C)(C)C)C(C(C)(C)C)CO2"
    )
    right = _mol(
        "CC1(C)C2=[N](->[Fe+2](<-[CH2-][Si](C)(C)C)(<-[CH2-][Si](C)(C)C)<-[N]3=C1OC[C@@H]3C(C)(C)C)[C@@H](C(C)(C)C)CO2"
    )
    if reverse:
        left, right = right, left
    result = eq.evaluate_equivalence(left, right, use_chirality=True, max_mappings=1)
    assert result.decision == eq.EquivalenceDecision.INCONCLUSIVE
    assert result.bounded_search.stop_reason == "stereo_incomplete"
    assert result.native_stats["mappings"] == 1
    assert not result.native_stats["mapping_limit"]
    assert not result.resonance_witness
    assert result.invariants["nonmetal_connectivity"].status == eq.InvariantStatus.PASSED
    limited = eq.evaluate_equivalence(left, right, use_chirality=True, max_mappings=0)
    assert limited.decision == eq.EquivalenceDecision.INCONCLUSIVE
    assert limited.native_stats["mapping_limit"]


def test_ignored_metal_stereo_does_not_hide_a_hard_contradiction():
    # ANABAU has a shared metal reference at both ends of a double bond.
    left = _mol(
        "CCOc1cccc2C=N3N=C([O-]->[V+5]<-34(<-[O-2])(<-O=c3ccoc(C)c3[O-]->4)<-[O-]c21)c1ccccc1Cl"
    )
    right = _mol(
        "CCOC1=CC=C/C2=C/[N]3=[N]4/C(=[O]->[V+]<-3<-45(<-[O-2])(<-[O]=C3C=C[O+]=C(C)C3=[O]->5)<-[O]=C12)c1ccccc1Cl"
    )
    result = eq.evaluate_equivalence(left, right, use_chirality=True)
    assert result.decision == eq.EquivalenceDecision.NOT_EQUIVALENT
    assert result.invariants["metal_state"].status == eq.InvariantStatus.FAILED
    # Full conversion retains the frame; native validation must also accept it.
    a, _ = eq._prepare_input(left, True, False)
    b, _ = eq._prepare_input(right, True, False)
    direct = _core.equivalence.evaluate(a, b, use_chirality=True)
    assert direct["decision"] == "not_equivalent"
    assert direct["invariants"]["metal_state"] == -1


def test_ignored_coordination_frame_is_filtered_before_double_bond_extraction():
    # BCOCFE includes an alkene endpoint with two covalent substituents plus
    # coordination. Its four-reference frame is not used on the projection.
    left = _mol("[O+]#[C-]->[Fe]12345(<-[C-]#[O+])<-C6=CC=C->1C->2(C->31=CC=CC=CC=C->41)=CC=C->56")
    right = _mol(
        r"[O+]#[C-]->[Fe]12345(<-[C-]#[O+])<-[CH]6=[CH]->1/C=C\[C]->2([C]->31=[CH]->4\C=C/C=C\C=C/1)=[CH]->5C=C6"
    )
    before = right.ToBinary(Chem.PropertyPickleOptions.AllProps)
    result = eq.evaluate_equivalence(left, right, use_chirality=True)
    assert result.decision == eq.EquivalenceDecision.INCONCLUSIVE
    assert result.bounded_search.stop_reason == "stereo_incomplete"
    assert right.ToBinary(Chem.PropertyPickleOptions.AllProps) == before
    # Strict coordination remains conservative about unsupported full frames.
    with pytest.raises(ValueError, match="Invalid double-bond stereo references"):
        rdkit_to_mol_data(right)


def test_ignored_ligand_stereo_does_not_leave_relative_group_constraints():
    left = rdkit_to_mol_data(_mol("F[C@](Cl)(Br)[Cu] |&1:1|"))
    right = rdkit_to_mol_data(_mol("FC(Cl)(Br)[Cu]"))
    assert left.stereo and left.stereo_groups
    result = _core.equivalence.evaluate(left, right, use_chirality=True)
    assert result["decision"] == "equivalent", result
    strict = _core.equivalence.evaluate(
        left, right, use_chirality=True, ignore_metal_coordination_bonds=False
    )
    assert strict["decision"] == "inconclusive", strict


@pytest.mark.parametrize("entry", [eq.evaluate_equivalence, eq.check_equivalence])
def test_removed_supplier_flags_cannot_change_coordination_policy(entry):
    left, right = _mol("N->[Cu]"), _mol("N.[Cu]")
    with pytest.raises(TypeError, match="resonance_flags"):
        entry(left, right, resonance_flags=0)
    with pytest.raises(TypeError):
        entry(left, right, False, 1000, 0)
    ignored = entry(left, right)
    strict = entry(left, right, ignore_metal_coordination_bonds=False)
    if isinstance(ignored, tuple):
        ignored, strict = ignored[1], strict[1]
    assert ignored.equivalent
    assert strict.decision == eq.EquivalenceDecision.NOT_EQUIVALENT


def test_native_resonance_diagnostics_do_not_claim_supplier_flags():
    from dataclasses import asdict

    result = eq.evaluate_equivalence(_mol("CC(=O)N"), _mol("CC([O-])=[NH2+]"))
    assert result.equivalent and result.resonance is not None
    detail = asdict(result.resonance)
    assert "resonance_flags" not in detail
    assert detail["max_resonance"] == 5000
    assert detail["mol1_resonance_count"] + detail["mol2_resonance_count"] > 0
