#!/usr/bin/env python3
# pyright: reportCallIssue=false
"""Measure native, previous, and patched stereo processing on fresh molecules.

Example: .tmp/venv311/bin/python scripts/stereo_workaround_speed.py --out /tmp/stereo.json
Preparation/copying is excluded. Each round shuffles the cases and methods.
The patch-only measurement starts with freshly assigned native stereo.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import platform
import random
import sys
from pathlib import Path
from statistics import mean, median
from time import perf_counter_ns


if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
from rdkit import Chem, RDLogger, rdBase
from rdkit.Chem import rdDistGeom, rdForceFieldHelpers

from molgr.utils.post_process import make_stereochemistry
from molgr.utils.rdkit_stereo import apply_3d_stereo_workarounds


ROOT = Path(__file__).resolve().parents[1]


def native(mol):
    mol.UpdatePropertyCache(strict=False)
    Chem.SetAromaticity(mol)
    Chem.AssignStereochemistryFrom3D(mol)
    return mol


def previous(mol):
    """MolGR's stereo processing before the workaround integration."""
    mol.UpdatePropertyCache(strict=False)
    Chem.SetAromaticity(mol)
    Chem.DetectBondStereochemistry(mol)
    Chem.SetBondStereoFromDirections(mol)
    Chem.AssignAtomChiralTagsFromStructure(mol)
    Chem.AssignStereochemistryFrom3D(mol)
    for idx in range(mol.GetNumBonds()):
        bond = mol.GetBondWithIdx(idx)
        if bond.GetStereo() == Chem.BondStereo.STEREONONE:
            bond.SetBondDir(Chem.BondDir.NONE)
    return mol


def embedded(smiles):
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    if rdDistGeom.EmbedMolecule(mol, randomSeed=0xC0FFEE) != 0:
        raise RuntimeError(f"Cannot embed {smiles}")
    rdForceFieldHelpers.UFFOptimizeMolecule(mol)
    return mol


def coordination(points, symbol="Rh", center=(0, 0, 0)):
    rw = Chem.RWMol()
    for element in (symbol, "F", "Cl", "Br", "I", "N", "O")[: len(points) + 1]:
        atom = Chem.Atom(element)
        atom.SetNoImplicit(True)
        rw.AddAtom(atom)
    for idx in range(1, len(points) + 1):
        rw.AddBond(0, idx, Chem.BondType.SINGLE)
    mol = rw.GetMol()
    mol.UpdatePropertyCache(strict=False)
    Chem.GetSymmSSSR(mol)
    conf = Chem.Conformer(mol.GetNumAtoms())
    conf.Set3D(True)
    for idx, point in enumerate((center, *points)):
        conf.SetAtomPosition(idx, point)
    mol.AddConformer(conf)
    return mol


def load_cases():
    cases = []
    common = {
        "ethanol": "CCO",
        "aspirin": "CC(=O)Oc1ccccc1C(=O)O",
        "caffeine": "Cn1c(=O)c2c(ncn2C)n(C)c1=O",
        "ibuprofen": "CC(C)Cc1ccc(cc1)[C@@H](C)C(=O)O",
        "cholesterol": "C[C@H](CCCC(C)C)[C@H]1CC[C@@H]2[C@@H]3CC=C4C[C@@H](O)CC[C@]4(C)[C@H]3CC[C@]12C",
        "trialanine": "N[C@@H](C)C(=O)N[C@@H](C)C(=O)N[C@@H](C)C(=O)O",
    }
    for name, smiles in common.items():
        cases.append(("common_organic", name, embedded(smiles)))
    for idx, smiles in enumerate((ROOT / "tests/test_cases.csv").read_text().splitlines()[1:], 1):
        if smiles.strip():
            cases.append(("regression_organic", f"case-{idx:02}", embedded(smiles.strip())))
    from scripts.tmqmg_fixture_data import reference_graph

    manifest = json.loads((ROOT / "tests/data/reviewed/tmqmg/manifest.json").read_text())
    for record in manifest["fixtures"]:
        if record["kind"] == "published_reference":
            mol = reference_graph(record["case_id"])
            mol.UpdatePropertyCache(strict=False)
            Chem.GetSymmSSSR(mol)
            cases.append(("published_coordination", record["case_id"], mol))
    examples = {
        "TH_outside": coordination(((1, 0, 0), (0, 1, 0), (0, 0, 1), (-1, -1, -1)), "C", (3, 0, 0)),
        "OH4": coordination(((1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1))),
        "TB4": coordination(((1, 0, 0), (-0.5, 0.866, 0), (0, 0, 1), (0, 0, -1))),
        "TB_distorted": coordination(
            ((0, 0, 1), (1, 0, 0), (0.8, 0.6, 0), (0.8, -0.6, 0), (0, 0, -1))
        ),
    }
    twisted = Chem.MolFromSmiles("FC(Cl)=C(Br)I")
    conf = Chem.Conformer(6)
    conf.Set3D(True)
    for idx, point in enumerate(
        ((-1, 1, 0), (-0.7, 0, 0), (-1, -1, 0), (0.7, 0, 0), (1, 0, 1), (1, 0, -1))
    ):
        conf.SetAtomPosition(idx, point)
    twisted.AddConformer(conf)
    examples["double_twisted"] = twisted
    cases.extend(("bug_reproducers", name, mol) for name, mol in examples.items())
    for _, _, mol in cases:
        for idx in range(mol.GetNumAtoms()):
            mol.GetAtomWithIdx(idx).SetNoImplicit(True)
        Chem.RemoveStereochemistry(mol)
    return cases


METHODS = {
    "native": native,
    "previous": previous,
    "patched": make_stereochemistry,
    "patch_only": apply_3d_stereo_workarounds,
}


def end_to_end(cases, rounds):
    import molgr.interface as interface
    from molgr.config import MolGRConfig

    inputs = []
    for group, name, mol in cases:
        if group == "common_organic":
            charge = sum(mol.GetAtomWithIdx(i).GetFormalCharge() for i in range(mol.GetNumAtoms()))
            inputs.append((group, name, Chem.MolToXYZBlock(mol), charge))
    directory = ROOT / "tests/data/tmqmg/reconstruction"
    with (directory / "manifest.csv").open() as fh:
        for row in csv.DictReader(fh):
            inputs.append(
                (
                    "reconstruction_coordination",
                    row["case_id"],
                    (directory / row["xyz_file"]).read_text(),
                    int(row["charge"]),
                )
            )
    config = MolGRConfig()
    config.cpp_backend.max_threads = 1
    records = []
    original = interface.restore_stereochemistry
    rng = random.Random(42)
    try:
        for group, name, xyz, charge in inputs:
            record = {"group": group, "name": name, "samples_ms": {}}
            counts = {}
            for method in ("native", "previous", "patched"):
                interface.restore_stereochemistry = METHODS[method]
                start = perf_counter_ns()
                mol = interface.xyz_to_rdmol(xyz, charge, 1, config=config)
                elapsed_ms = (perf_counter_ns() - start) / 1e6
                record["atoms"] = mol.GetNumAtoms()
                counts[method] = min(32, max(2, int(100 / elapsed_ms)))
                record["samples_ms"][method] = []
            record["repetitions_per_round"] = counts
            for _ in range(rounds):
                methods = list(record["samples_ms"])
                rng.shuffle(methods)
                for method in methods:
                    interface.restore_stereochemistry = METHODS[method]
                    start = perf_counter_ns()
                    for _ in range(counts[method]):
                        interface.xyz_to_rdmol(xyz, charge, 1, config=config)
                    record["samples_ms"][method].append(
                        (perf_counter_ns() - start) / counts[method] / 1e6
                    )
            records.append(record)
            print(
                name, {k: round(median(v), 3) for k, v in record["samples_ms"].items()}, flush=True
            )
    finally:
        interface.restore_stereochemistry = original
    return records


def measure(template, method, repetitions):
    # Fresh objects prevent a previous patch/cleanup from changing the workload.
    batch = [Chem.Mol(template) for _ in range(repetitions)]
    if method == "patch_only":
        for mol in batch:
            native(mol)
    fn = METHODS[method]
    gc_enabled = gc.isenabled()
    gc.disable()
    try:
        start = perf_counter_ns()
        for mol in batch:
            fn(mol)
        elapsed = perf_counter_ns() - start
    finally:
        if gc_enabled:
            gc.enable()
    return elapsed / repetitions / 1000  # microseconds per molecule


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=9)
    parser.add_argument(
        "--end-to-end",
        action="store_true",
        help="Measure the C++ XYZ-to-Mol pipeline, with one native worker, instead of stereo alone",
    )
    args = parser.parse_args()
    RDLogger.DisableLog("rdApp.*")
    Chem.SetAllowNontetrahedralChirality(True)
    Chem.SetUseLegacyStereoPerception(True)
    cases = load_cases()
    if args.end_to_end:
        records = end_to_end(cases, args.rounds)
        result = {
            "python": sys.version,
            "platform": platform.platform(),
            "rdkit": rdBase.rdkitVersion,
            "rounds": args.rounds,
            "methodology": "C++ XYZ-to-Mol, one native worker; warmup excluded; randomized method order per round; input preparation excluded",
            "cases": records,
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2) + "\n")
        return
    records = []
    for group, name, mol in cases:
        record = {"group": group, "name": name, "atoms": mol.GetNumAtoms(), "samples_us": {}}
        counts = {}
        for method in METHODS:
            estimate = measure(mol, method, 8)
            counts[method] = min(512, max(8, int(3000 / estimate)))
            record["samples_us"][method] = []
        record["repetitions_per_round"] = counts
        records.append(record)
    rng = random.Random(42)
    for round_idx in range(args.rounds):
        jobs = [(idx, method) for idx in range(len(cases)) for method in METHODS]
        rng.shuffle(jobs)
        for idx, method in jobs:
            records[idx]["samples_us"][method].append(
                measure(cases[idx][2], method, records[idx]["repetitions_per_round"][method])
            )
        print(f"round {round_idx + 1}/{args.rounds}", flush=True)
    groups = {}
    for group in dict.fromkeys(record["group"] for record in records):
        selected = [record for record in records if record["group"] == group]
        rounds = {
            method: [
                mean(record["samples_us"][method][idx] for record in selected)
                for idx in range(args.rounds)
            ]
            for method in METHODS
        }
        values = {method: median(samples) for method, samples in rounds.items()}
        groups[group] = {
            "cases": len(selected),
            "atom_range": [min(r["atoms"] for r in selected), max(r["atoms"] for r in selected)],
            "median_us_per_molecule": values,
            "round_min_max_us": {method: [min(v), max(v)] for method, v in rounds.items()},
            "net_extra_vs_previous_us": values["patched"] - values["previous"],
            "net_extra_vs_previous_pct": 100 * (values["patched"] / values["previous"] - 1),
        }
    result = {
        "python": sys.version,
        "platform": platform.platform(),
        "rdkit": rdBase.rdkitVersion,
        "numpy": np.__version__,
        "rounds": args.rounds,
        "methodology": "Median of equal-case-weighted round means; fresh copies and preparation excluded; legacy stereo and non-tetrahedral stereo enabled",
        "groups": groups,
        "cases": records,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(groups, indent=2))


if __name__ == "__main__":
    main()
