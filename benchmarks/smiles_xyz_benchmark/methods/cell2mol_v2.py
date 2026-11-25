from __future__ import annotations

import contextlib
import tempfile
import time
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any

from rdkit import Chem

from benchmarks.smiles_xyz_benchmark.methods.base import BenchmarkMethod, MethodRunOutput
from benchmarks.smiles_xyz_benchmark.methods.postprocess import finalize_rdmol_with_dative_bonds


def _extract_rdmol(molecule: Any) -> Chem.Mol:
    """Convert cell2mol's complex result without inferring any new chemistry.

    v2 stores RDKit objects on ligands, not on the complete metal complex.
    Its atom bond lists omit auxiliary protons used during charge assignment.
    """
    if molecule is None:
        raise ValueError("cell2mol returned no molecule")
    for flag in (
        "error_get_poscharges",
        "error_multiple_distrib",
        "error_empty_distrib",
        "error_create_bonds",
    ):
        if getattr(molecule, flag, False):
            raise ValueError(f"cell2mol reported {flag}")
    direct = getattr(molecule, "rdkit_obj", None)
    if direct is not None:
        # Organic outputs can retain an exact-element QueryAtom from v2's
        # charge correction. It describes a concrete atom, but MolGR's graph
        # evaluator correctly rejects query molecules. Copy its ordinary atom
        # fields without changing chemistry, atom order, bonds or coordinates.
        concrete = Chem.RWMol(direct)
        for atom in direct.GetAtoms():
            if atom.HasQuery():
                expected = f"AtomAtomicNum {atom.GetAtomicNum()} = val"
                if atom.GetAtomicNum() <= 0 or atom.DescribeQuery().strip() != expected:
                    raise ValueError("cell2mol returned an unresolved atom query")
                concrete.ReplaceAtom(atom.GetIdx(), Chem.Atom(atom), preserveProps=True)
        if any(bond.HasQuery() for bond in direct.GetBonds()):
            raise ValueError("cell2mol returned an unresolved bond query")
        return concrete.GetMol()
    if not all(hasattr(atom, "charge") for atom in molecule.atoms):
        raise ValueError("cell2mol did not assign atomic charges")
    atoms = molecule.atoms
    indices = {id(atom): idx for idx, atom in enumerate(atoms)}
    metals = {id(atom) for atom in molecule.metals}
    rw = Chem.RWMol()
    conformer = Chem.Conformer(len(atoms))
    for idx, atom in enumerate(atoms):
        rd_atom = Chem.Atom(atom.label)
        # The molecule-level vector predates ligand zwitterion correction in
        # v2. Read the final per-atom assignment, as its native exporter does.
        rd_atom.SetFormalCharge(int(atom.charge))
        rd_atom.SetNoImplicit(True)
        rw.AddAtom(rd_atom)
        conformer.SetAtomPosition(idx, tuple(float(x) for x in atom.coord))
    # Preserve radicals assigned by cell2mol's ligand RDKit objects.
    for ligand in molecule.ligands:
        for idx, atom in enumerate(ligand.atoms):
            rw.GetAtomWithIdx(indices[id(atom)]).SetNumRadicalElectrons(
                ligand.rdkit_obj.GetAtomWithIdx(idx).GetNumRadicalElectrons()
            )
    orders = {
        0.0: Chem.BondType.ZERO,
        1.0: Chem.BondType.SINGLE,
        1.5: Chem.BondType.AROMATIC,
        2.0: Chem.BondType.DOUBLE,
        3.0: Chem.BondType.TRIPLE,
        4.0: Chem.BondType.QUADRUPLE,
    }
    for atom in atoms:
        for bond in getattr(atom, "bonds", []):
            i, j = indices[id(bond.atom1)], indices[id(bond.atom2)]
            if rw.GetBondBetweenAtoms(i, j) is not None:
                continue
            bond_type = orders[float(bond.order)]
            if float(bond.order) == 0 and (id(bond.atom1) in metals) != (id(bond.atom2) in metals):
                if id(bond.atom1) in metals:
                    i, j = j, i
                bond_type = Chem.BondType.DATIVE
            rw.AddBond(i, j, bond_type)
            if bond_type == Chem.BondType.AROMATIC:
                rw.GetAtomWithIdx(i).SetIsAromatic(True)
                rw.GetAtomWithIdx(j).SetIsAromatic(True)
    mol = rw.GetMol()
    mol.AddConformer(conformer)
    mol.UpdatePropertyCache(strict=False)
    return mol


@dataclass(frozen=True)
class Cell2MolV2Method(BenchmarkMethod):
    method_id: str = "cell2mol_v2"

    def run(self, case: dict[str, Any]) -> MethodRunOutput:
        timing_ms_breakdown: dict[str, float] = {
            "write_xyz_temp_ms": 0.0,
            "cell2mol_run_ms": 0.0,
            "extract_rdkit_ms": 0.0,
            "postprocess_ms": 0.0,
        }

        xyz_block = case.get("xyz_block")
        if not isinstance(xyz_block, str) or not xyz_block.strip():
            return MethodRunOutput(
                status="error",
                error="missing or empty xyz_block",
                timing_ms_breakdown=timing_ms_breakdown,
            )

        total_charge_value = case.get("total_charge", 0)
        try:
            total_charge = int(total_charge_value)
        except (TypeError, ValueError):
            return MethodRunOutput(
                status="error",
                error=f"invalid total_charge: {total_charge_value!r}",
                timing_ms_breakdown=timing_ms_breakdown,
            )

        try:
            get_molecule = import_module("cell2mol.xyz_molecule").get_molecule
        except Exception:  # noqa: BLE001
            return MethodRunOutput(
                status="skipped",
                error="cell2mol not installed",
                timing_ms_breakdown=timing_ms_breakdown,
            )

        classes_module: Any | None = None
        original_balance_charge: Any | None = None
        original_atom_init: Any | None = None
        try:
            try:
                classes_module = import_module("cell2mol.classes")
                original_atom_init = classes_module.atom.__init__

                def _compat_atom_init(self, *args: Any, **kwargs: Any) -> None:
                    original_atom_init(self, *args, **kwargs)
                    # Monatomic ligands have no covalent bonds; v2's later
                    # SMILES correction nevertheless reads this collection.
                    if not hasattr(self, "bonds"):
                        self.bonds = []

                classes_module.atom.__init__ = _compat_atom_init
                new_charge_assignment = import_module("cell2mol.new_charge_assignment")
                balance_charge_impl = getattr(new_charge_assignment, "balance_charge", None)
                original = getattr(classes_module, "balance_charge", None)
                if callable(balance_charge_impl) and callable(original):
                    original_balance_charge = original

                    def _compat_balance_charge(*args: Any, **kwargs: Any) -> Any:
                        if "charges_sum" in kwargs and "input_charge" not in kwargs:
                            kwargs["input_charge"] = kwargs.pop("charges_sum")
                        return balance_charge_impl(*args, **kwargs)

                    classes_module.balance_charge = _compat_balance_charge
            except Exception:  # noqa: BLE001
                pass

            write_started = time.perf_counter()
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir) / "input.xyz"
                temp_path.write_text(xyz_block, encoding="utf-8")
                timing_ms_breakdown["write_xyz_temp_ms"] = (
                    time.perf_counter() - write_started
                ) * 1000.0

                cell2mol_started = time.perf_counter()
                mol = get_molecule(str(temp_path), "mol", total_charge, temp_dir, 0)
                timing_ms_breakdown["cell2mol_run_ms"] = (
                    time.perf_counter() - cell2mol_started
                ) * 1000.0

                extract_started = time.perf_counter()
                rdkit_mol = _extract_rdmol(mol)
                timing_ms_breakdown["extract_rdkit_ms"] = (
                    time.perf_counter() - extract_started
                ) * 1000.0

                if rdkit_mol is None:
                    return MethodRunOutput(
                        status="error",
                        error="cell2mol molecule missing rdkit_obj",
                        timing_ms_breakdown=timing_ms_breakdown,
                    )
                postprocess_started = time.perf_counter()
                warnings: list[str] = []
                predicted_smiles: str | None = None
                try:
                    rdkit_mol, predicted_smiles = finalize_rdmol_with_dative_bonds(rdkit_mol)
                except Exception as exc:  # noqa: BLE001
                    warnings.append(f"MolToSmiles failed: {exc}")
                timing_ms_breakdown["postprocess_ms"] = (
                    time.perf_counter() - postprocess_started
                ) * 1000.0

                return MethodRunOutput(
                    status="ok",
                    error="; ".join(warnings) if warnings else None,
                    predicted_smiles=predicted_smiles,
                    rdkit_mol=rdkit_mol,
                    timing_ms_breakdown=timing_ms_breakdown,
                )
        except Exception as exc:  # noqa: BLE001
            return MethodRunOutput(
                status="error",
                error=f"cell2mol_v2 failed: {exc}",
                timing_ms_breakdown=timing_ms_breakdown,
            )
        finally:
            if classes_module is not None and original_atom_init is not None:
                classes_module.atom.__init__ = original_atom_init
            if classes_module is not None and callable(original_balance_charge):
                with contextlib.suppress(Exception):
                    classes_module.balance_charge = original_balance_charge
