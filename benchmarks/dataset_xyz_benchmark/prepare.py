"""Prepare a compressed, disk-backed case store from released geometries.

No embedding, hydrogen addition, optimization, or MolGR reconstruction is used.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import math
import pickle
import sqlite3
import struct
import tarfile
import zlib
from collections import Counter
from contextlib import suppress
from pathlib import Path

from rdkit import Chem

from benchmarks.dataset_xyz_benchmark import tmqmg
from benchmarks.dataset_xyz_benchmark.comparison import DEFAULT_POLICY, comparison_policy


ROOT = Path(__file__).resolve().parents[2]
DATASETS = ("geom-qm9", "geom-drugs", "borylxat", "borylxat-fixtures", "bde", "tmqmg-corrected")
SCHEMA_VERSION = 2
GEOM_URL = "https://doi.org/10.7910/DVN/JNGTDF"


def file_hash(path, algorithm="sha256"):
    digest = hashlib.new(algorithm)
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class GeomUnpickler(pickle.Unpickler):
    """GEOM needs only the RDKit molecule constructor, not arbitrary globals."""

    def find_class(self, module, name):
        if (module, name) == ("rdkit.Chem.rdchem", "Mol"):
            return Chem.Mol
        raise pickle.UnpicklingError(f"Unsupported GEOM pickle global: {module}.{name}")


def xyz_block(mol):
    if mol.GetNumConformers() != 1:
        raise ValueError("Expected exactly one source conformer")
    conf = mol.GetConformer()
    lines = [str(mol.GetNumAtoms()), "source coordinates; angstrom"]
    for atom in mol.GetAtoms():
        position = conf.GetAtomPosition(atom.GetIdx())
        if not all(math.isfinite(x) for x in position):
            raise ValueError("Nonfinite coordinates")
        lines.append(atom.GetSymbol() + " " + " ".join(format(x, ".17g") for x in position))
    return "\n".join(lines) + "\n"


def validate_reference(geometry, reference, charge, multiplicity):
    if reference is None:
        return "unresolved", "Missing source reference graph"
    if multiplicity is None:
        return "unresolved", "Source does not establish spin multiplicity"
    try:
        graph = Chem.Mol(reference)
        Chem.SanitizeMol(graph)
        expected = Counter(a.GetAtomicNum() for a in geometry.GetAtoms())
        observed = Counter(a.GetAtomicNum() for a in Chem.AddHs(graph).GetAtoms())
        if expected != observed:
            return "excluded", "Reference composition differs from coordinate atoms"
        if Chem.GetFormalCharge(graph) != charge:
            return "excluded", "Reference charge differs from source total charge"
        if (sum(a.GetAtomicNum() for a in geometry.GetAtoms()) - charge) % 2 != (
            multiplicity - 1
        ) % 2:
            return "excluded", "Electron parity contradicts spin multiplicity"
        if sum(a.GetNumRadicalElectrons() for a in graph.GetAtoms()) != multiplicity - 1:
            return "unresolved", "Reference formal radical count differs from multiplicity - 1"
        return "valid", "Source graph passes composition, charge and formal-radical checks"
    except Exception as exc:
        return "excluded", f"Reference validation failed: {exc}"


def make_case(dataset, molecule, conformer, geometry, reference, charge, multiplicity, evidence):
    block = xyz_block(geometry)
    status, reason = validate_reference(geometry, reference, charge, multiplicity)
    native = reference.ToBinary() if reference is not None else None
    smiles = None
    if reference is not None:
        # SMILES is display-only; preserve the native graph regardless.
        with suppress(Exception):
            smiles = Chem.MolToSmiles(reference, allHsExplicit=True)
    return (
        dataset,
        molecule,
        str(conformer),
        zlib.compress(block.encode()),
        smiles,
        charge,
        multiplicity,
        status,
        reason,
        json.dumps(evidence, allow_nan=False),
        native,
        hashlib.sha256(native).hexdigest() if native is not None else None,
    )


def geom_cases(path, dataset, limit_molecules=None):
    subset = "qm9" if dataset == "geom-qm9" else "drugs"
    seen = set()
    count = 0
    # The official .tar.gz is actually an uncompressed tar in this release.
    with tarfile.open(path, "r|*") as archive:
        for member in archive:
            parts = Path(member.name).parts
            if (
                not member.isfile()
                or subset not in parts[:-1]
                or not member.name.endswith(".pickle")
            ):
                continue
            raw = archive.extractfile(member).read()
            source = GeomUnpickler(io.BytesIO(raw)).load()
            molecule = source["smiles"]
            if molecule in seen:
                raise ValueError(f"Duplicate parent molecule: {molecule}")
            seen.add(molecule)
            conformers = source["conformers"]
            if not conformers:
                raise ValueError(f"Empty conformer ensemble: {member.name}")
            charge = int(source["charge"])
            if charge != source["charge"]:
                raise ValueError("Nonintegral source charge")
            for index, conf in enumerate(conformers):
                mol = conf["rd_mol"]
                # A single unpaired electron establishes a doublet; multiple
                # formal radicals do not establish their spin coupling.
                radicals = sum(a.GetNumRadicalElectrons() for a in mol.GetAtoms())
                multiplicity = radicals + 1 if radicals in (0, 1) else None
                evidence = {
                    "archive_member": member.name,
                    "member_sha256": hashlib.sha256(raw).hexdigest(),
                    "source_url": GEOM_URL,
                    "source_geom_id": conf.get("geom_id"),
                    "relativeenergy": conf.get("relativeenergy"),
                    "totalenergy": conf.get("totalenergy"),
                    "degeneracy": conf.get("degeneracy"),
                    "boltzmannweight": conf.get("boltzmannweight"),
                    "released_conformers": len(conformers),
                    "source_totalconfs": source.get("totalconfs"),
                    "state_basis": "Source charge; source graph closed-shell singlet or one-radical doublet; multiradical spin coupling unresolved",
                    "reference_basis": "Per-conformer released rd_mol; source uses xyz2mol; not independent chemical ground truth",
                }
                yield make_case(dataset, molecule, index, mol, mol, charge, multiplicity, evidence)
            count += 1
            if limit_molecules and count >= limit_molecules:
                return


def fixture_cases(path):
    manifest = json.loads((path / "manifest.json").read_text())
    for row in manifest["fixtures"]:
        mol = next(
            Chem.SDMolSupplier(
                str(path / "reference_graph" / f"{row['case_id']}.sdf"), removeHs=False
            )
        )
        if mol is None:
            raise ValueError(row["case_id"])
        yield make_case(
            "borylxat-fixtures",
            row["case_id"],
            0,
            mol,
            mol,
            row["charge"],
            row["spin_multiplicity"],
            row,
        )


def borylxat_cases(path):
    from scripts.freeze_borylxat_fixtures import SOURCE, corrected_reference, electronic_state_basis

    if (
        path.stat().st_size != SOURCE["size_bytes"]
        or file_hash(path, "md5") != SOURCE["published_md5"]
    ):
        raise ValueError("BorylXAT-DB does not match pinned Zenodo size/MD5")
    with sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True) as db:
        for row_id, natoms, charge, metadata, numbers, positions, data in db.execute(
            "SELECT id,natoms,charge,key_value_pairs,numbers,positions,data FROM systems ORDER BY id"
        ):
            meta = json.loads(metadata)
            category = meta["category"]
            elements = struct.unpack(f"<{natoms}i", numbers)
            coordinates = struct.unpack(f"<{3 * natoms}d", positions)
            geometry = Chem.RWMol()
            conf = Chem.Conformer(natoms)
            conf.Set3D(True)
            for i, z in enumerate(elements):
                atom = Chem.Atom(z)
                atom.SetNoImplicit(True)
                geometry.AddAtom(atom)
                conf.SetAtomPosition(i, coordinates[i * 3 : i * 3 + 3])
            geometry.AddConformer(conf)
            offset = int.from_bytes(data[:8], "little")
            payload = json.loads(data[offset:])
            spins = payload.get("spin_densities")
            is_radical = category in {"B", "complex_r", "c_radical"}
            is_closed = category in {"LB", "Cl", "complex_p"}
            multiplicity = 2 if is_radical else 1 if is_closed else None
            if is_radical and (
                not isinstance(spins, list)
                or len(spins) != natoms
                or not all(math.isfinite(x) for x in spins)
                or not math.isclose(sum(spins), 1, abs_tol=1e-4)
            ):
                multiplicity = None
            evidence = {
                "source_url": SOURCE["record_url"],
                "ase_row_id": row_id,
                "category": category,
                "source_smiles": meta.get("smiles"),
                "state_basis": electronic_state_basis(category)
                if category in {"B", "complex_r", "complex_p"}
                else "Author category and spin-density sum; electron parity checked separately",
                "spin_density_sum": sum(spins) if isinstance(spins, list) else None,
            }
            reference = None
            if category in {"B", "complex_r", "complex_p"} and multiplicity is not None:
                case = {
                    "case_id": meta["key"],
                    "natoms": natoms,
                    "category": category,
                    "source_smiles": meta["smiles"],
                    "charge": int(charge),
                    "spin_multiplicity": multiplicity,
                }
                try:
                    reference, mapping, removed = corrected_reference(geometry, case)
                    evidence.update(source_to_geometry=mapping, removed_extra_boron_h=removed)
                except Exception as exc:
                    evidence["reference_error"] = str(exc)
            elif is_radical or is_closed:
                reference = Chem.MolFromSmiles(meta.get("smiles", ""))
            result = make_case(
                "borylxat",
                meta["key"],
                row_id,
                geometry,
                reference,
                int(charge),
                multiplicity,
                evidence,
            )
            if category == "ts" or category not in {
                "B",
                "complex_r",
                "complex_p",
                "LB",
                "Cl",
                "c_radical",
            }:
                result = (
                    *result[:7],
                    "not_applicable",
                    "Transition-state/other category kept separately from stable-graph accuracy",
                    *result[9:],
                )
            yield result


def sdf_cases(path, dataset, source_url, multiplicity_property):
    for index, mol in enumerate(Chem.ForwardSDMolSupplier(str(path), removeHs=False)):
        if mol is None:
            raise ValueError(f"Unparseable SDF record {index}; do not silently drop it")
        multiplicity = (
            mol.GetIntProp(multiplicity_property) if mol.HasProp(multiplicity_property) else None
        )
        yield make_case(
            dataset,
            str(index),
            0,
            mol,
            mol,
            Chem.GetFormalCharge(mol),
            multiplicity,
            {
                "source_url": source_url,
                "sdf_record": index,
                "source_name": mol.GetProp("_Name") if mol.HasProp("_Name") else None,
                "state_basis": f"SDF property {multiplicity_property}; missing values stay unresolved",
            },
        )


def tmqmg_cases(path):
    # Eligibility and input states belong to the release, not local heuristics.
    # In particular, a singlet may have coupled formal radicals in its target.
    for row in tmqmg.rows(path, include_excluded=True):
        native = (
            zlib.decompress(base64.b64decode(row["native_graph"]))
            if row["native_graph"] is not None
            else None
        )
        yield (
            "tmqmg-corrected",
            row["id"],
            "0",
            zlib.compress(row["xyz"].encode()),
            None,
            int(row["charge"]),
            int(row["multiplicity"]),
            "valid" if native is not None else "excluded",
            f"Published {tmqmg.VERSION} native target"
            if native is not None
            else "No reference released; excluded by the published standard",
            json.dumps({"source_id": row["id"], "revision": tmqmg.REVISION}),
            native,
            hashlib.sha256(native).hexdigest() if native is not None else None,
        )


def select_molecules(cases, limit):
    """Import complete parent groups; all source adapters yield grouped records."""
    seen = set()
    for case in cases:
        if limit is not None and case[1] not in seen and len(seen) >= limit:
            return
        seen.add(case[1])
        yield case


def prepare(
    source,
    output,
    dataset,
    *,
    limit_molecules=None,
    source_url=None,
    multiplicity_property="MULTIPLICITY",
):
    if dataset not in DATASETS:
        raise ValueError(f"Unknown dataset: {dataset}")
    if limit_molecules is not None and limit_molecules < 1:
        raise ValueError("--limit-molecules must be positive")
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    if dataset in {"geom-qm9", "geom-drugs"}:
        if limit_molecules is None and (
            source.stat().st_size != 50137579520
            or file_hash(source, "md5") != "e8f2168b7050652db22c976be25c450e"
        ):
            raise ValueError(
                "GEOM archive does not match the pinned official size/MD5; use an explicitly limited pilot for a partial archive"
            )
        cases = geom_cases(source, dataset, limit_molecules)
    elif dataset == "borylxat-fixtures":
        cases = fixture_cases(source)
    elif dataset == "borylxat":
        cases = borylxat_cases(source)
    elif dataset == "tmqmg-corrected":
        release = tmqmg.provenance(source)  # Validate with actionable checkout instructions first.
        cases = tmqmg_cases(source)
    else:
        if not source_url:
            raise ValueError("SDF sources require --source-url")
        cases = sdf_cases(source, dataset, source_url, multiplicity_property)
    provenance = {
        "schema": SCHEMA_VERSION,
        "dataset": dataset,
        "source": str(source.resolve()),
        "source_sha256": file_hash(source)
        if source.is_file()
        else file_hash(source / "manifest.json"),
        "source_url": source_url,
        "limit_molecules": limit_molecules,
        "conformer_policy": "all released conformers, no degeneracy replication",
        "complete": False,
        "comparison_policy": dict(DEFAULT_POLICY),
        "comparison_policy_locked": False,
    }
    if dataset == "tmqmg-corrected":
        provenance.update(
            source_url=tmqmg.REPOSITORY,
            release=release,
            comparison_policy=release["benchmark"],
            comparison_policy_locked=True,
        )
    comparison_policy(provenance)  # Validate before creating a case store.
    with sqlite3.connect(output) as db:
        db.executescript("""
        CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE cases (id INTEGER PRIMARY KEY, dataset TEXT, molecule TEXT, conformer TEXT,
          xyz BLOB, reference_smiles TEXT, charge INTEGER, multiplicity INTEGER,
          reference_status TEXT, reference_reason TEXT, evidence TEXT,
          reference_native BLOB, reference_sha256 TEXT,
          UNIQUE(dataset,molecule,conformer));
        CREATE INDEX molecule_cases ON cases(molecule);
        """)
        db.execute("INSERT INTO metadata VALUES ('provenance',?)", (json.dumps(provenance),))
        db.commit()
        for count, case in enumerate(select_molecules(cases, limit_molecules), 1):
            db.execute(
                "INSERT INTO cases(dataset,molecule,conformer,xyz,reference_smiles,charge,multiplicity,reference_status,reference_reason,evidence,reference_native,reference_sha256) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                case,
            )
            if count % 1000 == 0:
                db.commit()
                print(f"Prepared {count} conformers", flush=True)
        provenance["complete"] = True  # Complete selected input, not necessarily the full dataset.
        provenance["pilot"] = bool(limit_molecules) or dataset.endswith("-fixtures")
        provenance["cases"] = db.execute("SELECT count(*) FROM cases").fetchone()[0]
        provenance["molecules"] = db.execute(
            "SELECT count(DISTINCT molecule) FROM cases"
        ).fetchone()[0]
        provenance["reference_statuses"] = dict(
            db.execute("SELECT reference_status,count(*) FROM cases GROUP BY reference_status")
        )
        db.execute("UPDATE metadata SET value=? WHERE key='provenance'", (json.dumps(provenance),))
    return provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--dataset",
        choices=DATASETS,
        required=True,
    )
    parser.add_argument("--source-url")
    parser.add_argument("--multiplicity-property", default="MULTIPLICITY")
    parser.add_argument("--limit-molecules", type=int)
    args = parser.parse_args()
    print(
        json.dumps(
            prepare(
                args.source,
                args.out,
                args.dataset,
                limit_molecules=args.limit_molecules,
                source_url=args.source_url,
                multiplicity_property=args.multiplicity_property,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
