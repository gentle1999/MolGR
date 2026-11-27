"""Join independently optimized BorylXAT species to the published reaction graphs."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import struct

from rdkit import Chem

from scripts.freeze_borylxat_fixtures import SOURCE


# Optimized structures are linked by chemical role, independent of SMILES fragment order.
ROLES = (
    ("reactant_complex_key", "complex_r", 2),
    ("reactant_cl_key", "Cl", 1),
    ("product_complex_key", "complex_p", 1),
    ("product_c_radical_key", "c_radical", 2),
)
MULTIPLICITY = {category: mult for _, category, mult in ROLES}


def provenance(source):
    if not source.is_file() or source.stat().st_size != SOURCE["size_bytes"]:
        raise ValueError(f"Expected the published BorylXAT-DB.db from {SOURCE['download_url']}")
    digest = hashlib.md5()
    with source.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != SOURCE["published_md5"]:
        raise ValueError("BorylXAT-DB.db does not match the published Zenodo MD5")
    return {
        "source_url": SOURCE["record_url"],
        "md5": digest.hexdigest(),
        "geometry_basis": "GS_OPT",
        "reference_basis": "reaction AAM fragments",
    }


def reaction_fragments(aam):
    """Preserve each mapped H node and require the same isotope and element on both sides."""
    params = Chem.SmilesParserParams()
    params.removeHs = False
    inventories = []
    sides = []
    reactant, product = aam.split(">>")
    for side in (reactant, product):
        inventory = {}
        fragments = []
        first, second = side.split(".")
        for smiles in (first, second):
            mol = Chem.MolFromSmiles(smiles, params)
            if mol is None:
                raise ValueError("Invalid reaction fragment")
            for atom in mol.GetAtoms():
                idx = atom.GetAtomMapNum()
                if (
                    idx <= 0
                    or idx in inventory
                    or atom.GetNumExplicitHs()
                    or atom.GetNumImplicitHs()
                ):
                    raise ValueError("Each atom including hydrogen must be mapped exactly once")
                inventory[idx] = atom
            fragments.append((smiles, mol))
        inventories.append(inventory)
        sides.append(fragments)
    left, right = inventories

    def elements(atoms):
        return {i: (a.GetAtomicNum(), a.GetIsotope()) for i, a in atoms.items()}

    if elements(left) != elements(right):
        raise ValueError("Mapped reaction does not conserve atoms")
    # Find the mapped chlorine that changes its sole neighbour from C to B.
    # Other chlorines and the serialized order of disconnected fragments can vary.
    boron_maps = []
    for i, atom in left.items():
        if atom.GetAtomicNum() != 17 or atom.GetDegree() != 1 or right[i].GetDegree() != 1:
            continue
        before, after = atom.GetNeighbors()[0], right[i].GetNeighbors()[0]
        if before.GetAtomicNum() == 6 and after.GetAtomicNum() == 5:
            boron_maps.append(after.GetAtomMapNum())
    if len(boron_maps) != 1:
        raise ValueError("Expected one chlorine transfer from C to B")
    result = []
    for fragments in sides:
        complex_index = next(
            i
            for i, (_, mol) in enumerate(fragments)
            if any(a.GetAtomMapNum() == boron_maps[0] for a in mol.GetAtoms())
        )
        for smiles, mol in (fragments[complex_index], fragments[1 - complex_index]):
            # TS atom-map indices do not index an independently optimized component.
            for atom in mol.GetAtoms():
                atom.SetAtomMapNum(0)
            result.append((smiles, mol))
    return result


def optimized_rows(path):
    """Use TS rows only for labels and joins; emit each released GS_OPT geometry once."""
    structures = {}
    targets = {}
    conflicts = set()
    incident = {}
    expected_categories = {}
    with sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True) as db:
        for row_id, metadata in db.execute("SELECT id,key_value_pairs FROM systems ORDER BY id"):
            meta = json.loads(metadata)
            category, key = meta["category"], meta["key"]
            if category in MULTIPLICITY:
                if key in structures:
                    raise ValueError(f"Duplicate optimized structure key: {key}")
                structures[key] = (row_id, category)
            if category != "ts":
                continue
            for (role, expected, _), (smiles, mol) in zip(
                ROLES, reaction_fragments(meta["smiles"])
            ):
                component = meta[role]
                if expected_categories.setdefault(component, expected) != expected:
                    raise ValueError(f"Inconsistent component role for {component}")
                incident.setdefault(component, []).append(key)
                # Compare isomeric explicit-node graphs across reactions; never rebuild a
                # target from this canonical signature or from a single-species SMILES.
                signature = Chem.MolToSmiles(mol)
                if component not in targets:
                    targets[component] = (signature, mol, smiles, key)
                elif targets[component][0] != signature:
                    conflicts.add(component)
        if not targets:
            raise ValueError("Missing reaction AAM records")
        missing = targets.keys() - structures.keys()
        if missing:
            raise ValueError(f"Missing linked optimized geometry: {min(missing)}")
        for key, (row_id, category) in structures.items():
            n, charge, numbers, positions = db.execute(
                "SELECT natoms,charge,numbers,positions FROM systems WHERE id=?", (row_id,)
            ).fetchone()
            if expected_categories.get(key, category) != category:
                raise ValueError(f"Incorrect optimized category for {key}")
            geometry = Chem.RWMol()
            conf = Chem.Conformer(n)
            conf.Set3D(True)
            if n < 1 or len(positions) != 24 * n:
                raise ValueError(f"Invalid optimized geometry for {key}")
            for i, (z, point) in enumerate(
                zip(struct.unpack(f"<{n}i", numbers), struct.iter_unpack("<3d", positions))
            ):
                atom = Chem.Atom(z)
                atom.SetNoImplicit(True)
                geometry.AddAtom(atom)
                conf.SetAtomPosition(i, point)
            geometry.AddConformer(conf)
            evidence = {
                "ase_row_id": row_id,
                "category": category,
                "component_key": key,
                "reaction_keys": incident.get(key, []),
            }
            reference = None
            issue = None
            if key not in targets:
                issue = "No mapped reaction links this optimized component"
            elif key in conflicts:
                issue = "Linked reactions disagree on the component graph"
            else:
                _, reference, smiles, label_key = targets[key]
                evidence.update(mapped_fragment=smiles, label_reaction_key=label_key)
            if int(charge) != charge:
                raise ValueError(f"Nonintegral source charge for {key}")
            yield key, geometry, reference, int(charge), MULTIPLICITY[category], evidence, issue
