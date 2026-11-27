"""Use xyz2mol_tm’s general molecule entry point for organic XYZ geometries."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator

from rdkit import Chem

from benchmarks.smiles_xyz_benchmark.methods.base import (
    BenchmarkMethod,
    MethodRunOutput,
    allow_charged_fragments,
)


@contextmanager
def _call_valences(upstream: Any, charged_fragments: bool) -> Iterator[None]:
    """Use the upstream CLI valences for one call in a persistent benchmark worker."""
    baseline = upstream.atomic_valence
    table = baseline.copy()
    if not charged_fragments:
        # xyz2mol_local's CLI sets these before AC2BO searches for bond orders.
        table.update({6: [4, 2], 7: [3, 2], 8: [2, 1]})
    upstream.atomic_valence = table
    try:
        yield
    finally:
        upstream.atomic_valence = baseline


@dataclass(frozen=True)
class OrganicXYZ2Mol(BenchmarkMethod):
    method_id: str = "xyz2mol_tm_local"

    def run(self, case: dict[str, Any]) -> MethodRunOutput:
        try:
            from xyz2mol_tm.huckel_to_smiles import xyz2mol_local
        except ImportError as exc:
            return MethodRunOutput(status="skipped", error=str(exc))

        rows = [line.split() for line in case["xyz_block"].splitlines()[2:] if line.strip()]
        numbers = [xyz2mol_local.int_atom(row[0]) for row in rows]
        coordinates = [[float(x) for x in row[1:4]] for row in rows]
        try:
            charge = case["total_charge"]
            charged = allow_charged_fragments(charge, int(case.get("total_radical_electrons", 0)))
            with _call_valences(xyz2mol_local, charged):
                mol = xyz2mol_local.xyz2mol(
                    numbers,
                    coordinates,
                    charge=charge,
                    use_atom_maps=False,
                    allow_charged_fragments=charged,
                )
        except SystemExit as exc:
            return MethodRunOutput(
                status="error", error=f"xyz2mol_local exited during reconstruction: {exc.code}"
            )
        if mol is None:
            return MethodRunOutput(status="error", error="xyz2mol_local returned no molecule")
        # Upstream builds the first atom from the single-element SMARTS [#Z].
        # Keep its element, chemical properties, coordinates and atom order.
        first = mol.GetAtomWithIdx(0)
        if first.HasQuery():
            if (
                first.GetAtomicNum() <= 0
                or first.DescribeQuery().strip() != f"AtomAtomicNum {first.GetAtomicNum()} = val"
            ):
                raise ValueError("Unexpected constraint in xyz2mol_local's first atom")
            editable = Chem.RWMol(mol)
            editable.ReplaceAtom(0, Chem.Atom(first), preserveProps=True)
            mol = editable.GetMol()
        return MethodRunOutput(status="ok", predicted_smiles=Chem.MolToSmiles(mol), rdkit_mol=mol)
