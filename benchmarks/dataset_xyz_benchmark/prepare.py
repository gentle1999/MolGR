"""Prepare a compressed, disk-backed case store from released geometries.

No embedding, hydrogen addition, optimization, or MolGR reconstruction is used.
"""

from __future__ import annotations

import argparse
import base64
import gzip
import hashlib
import io
import json
import math
import pickle
import sqlite3
import tarfile
import zlib
from collections import Counter
from contextlib import suppress
from decimal import Decimal
from pathlib import Path

from rdkit import Chem

from benchmarks.dataset_xyz_benchmark import borylxat, tmqmg
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
        # RDKit's XYZ reader can reject scientific notation for small GS_OPT coordinates.
        values = [format(x, ".17g") for x in position]
        lines.append(
            " ".join(
                [atom.GetSymbol(), *(format(Decimal(v), "f") if "e" in v else v for v in values)]
            )
        )
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


def make_case(
    dataset,
    molecule,
    conformer,
    geometry,
    reference,
    charge,
    multiplicity,
    evidence,
    *,
    normalize_geom_queries=False,
):
    block = xyz_block(geometry)
    status, reason = validate_reference(geometry, reference, charge, multiplicity)
    native = reference.ToBinary() if reference is not None else None
    smiles = None
    if reference is not None:
        # SMILES is display-only; preserve the native graph regardless.
        with suppress(Exception):
            smiles = Chem.MolToSmiles(reference, allHsExplicit=True)
    if normalize_geom_queries and status == "valid":
        source_hash = hashlib.sha256(native).hexdigest()
        native, query_count = _geom_scoring_reference(native)
        evidence = dict(
            evidence, source_reference_sha256=source_hash, exact_element_query_atoms=query_count
        )
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


GEOM_SEED = 20261125
GEOM_DRUGS_PARENTS = 100_000
_GEOM_NAMES = {"geom-drugs": "GEOM-Drugs", "geom-qm9": "GEOM-QM9"}


def _geom_scoring_reference(native):
    """Replace exact-element query atoms in eligible GEOM targets with ordinary atoms.

    GEOM uses these queries for some concrete elements. Keep ambiguous queries
    and query bonds intact so the comparator can report an unsupported reference.
    """
    mol = Chem.Mol(native)
    if not mol.HasQuery():
        return native, 0
    atoms = [atom for atom in mol.GetAtoms() if atom.HasQuery()]
    if any(bond.HasQuery() for bond in mol.GetBonds()) or any(
        atom.GetAtomicNum() <= 0
        or atom.DescribeQuery().strip() != f"AtomAtomicNum {atom.GetAtomicNum()} = val"
        for atom in atoms
    ):
        return native, -1
    graph = Chem.RWMol(mol)
    for atom in atoms:
        graph.ReplaceAtom(atom.GetIdx(), Chem.Atom(atom), preserveProps=True)
    return graph.GetMol().ToBinary(), len(atoms)


def _geom_digest(dataset, domain, member):
    key = f"{_GEOM_NAMES[dataset]}/{GEOM_SEED}/{domain}\0{member}"
    return hashlib.sha256(key.encode("utf-8")).digest()


def _geom_conformer_index(dataset, member, size):
    """Select a uniform zero-based ensemble index without modulo bias."""
    ceiling = (1 << 256) // size * size
    counter = 0
    while True:
        digest = _geom_digest(dataset, f"conformer/{counter}", member)
        value = int.from_bytes(digest, "big")
        if value < ceiling:
            return value % size
        counter += 1


def _geom_members(archive, dataset, pilot):
    """Yield archive members in case-ID order, with their Drugs sampling ranks."""
    subset = dataset.split("-", 1)[1]
    if dataset == "geom-qm9" or pilot:
        for member in archive:
            if (
                member.isfile()
                and subset in Path(member.name).parts[:-1]
                and member.name.endswith(".pickle")
            ):
                yield None, member
        return

    # The pinned release is an uncompressed tar despite its .tar.gz suffix.
    # Index headers once and seek directly to each ranked member's payload.
    prefix = f"rdkit_folder/{subset}/"
    members = {m.name: m for m in archive.getmembers() if m.isfile() and m.name.startswith(prefix)}
    with archive.extractfile("rdkit_folder/summary_drugs.json") as stream:
        inventory = json.load(stream)
    paths = {
        "rdkit_folder/" + item["pickle_path"]
        for item in inventory.values()
        if item.get("pickle_path")
    }
    del inventory
    ranked = sorted(paths, key=lambda name: (_geom_digest(dataset, "parent", name), name))
    for rank, name in enumerate(ranked, 1):
        if name in members:
            yield rank, members[name]


def geom_cases(path, dataset, limit_molecules=None):
    """Prepare the fixed one-conformer cohort, or an archive-order all-conformer pilot."""
    pilot = limit_molecules is not None
    sampled_drugs = dataset == "geom-drugs" and not pilot
    seen = set()
    with tarfile.open(path, "r|*" if pilot else "r:*") as archive:
        for rank, member in _geom_members(archive, dataset, pilot):
            with archive.extractfile(member) as stream:
                raw = stream.read()
            source = GeomUnpickler(io.BytesIO(raw)).load()
            molecule = source["smiles"]
            if molecule in seen:
                if sampled_drugs:
                    continue
                raise ValueError(f"Duplicate parent molecule: {molecule}")
            conformers = source["conformers"]
            if not conformers:
                if sampled_drugs:
                    continue
                raise ValueError(f"Empty conformer ensemble: {member.name}")
            charge = int(source["charge"])
            if charge != source["charge"]:
                raise ValueError("Nonintegral source charge")
            indices = (
                range(len(conformers))
                if pilot
                else (_geom_conformer_index(dataset, member.name, len(conformers)),)
            )
            member_hash = hashlib.sha256(raw).hexdigest()
            for index in indices:
                conf = conformers[index]
                mol = conf["rd_mol"]
                radicals = sum(a.GetNumRadicalElectrons() for a in mol.GetAtoms())
                multiplicity = radicals + 1 if radicals in (0, 1) else None
                evidence = {
                    "archive_member": member.name,
                    "member_sha256": member_hash,
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
                if not pilot:
                    evidence["seed"] = GEOM_SEED
                if rank is not None:
                    evidence["sample_rank"] = rank
                case = make_case(
                    dataset,
                    molecule,
                    index,
                    mol,
                    mol,
                    charge,
                    multiplicity,
                    evidence,
                    normalize_geom_queries=not pilot,
                )
                if sampled_drugs and case[7] != "valid":
                    continue
                seen.add(molecule)
                yield case
            if len(seen) == (GEOM_DRUGS_PARENTS if sampled_drugs else limit_molecules):
                return
    if sampled_drugs:
        raise ValueError(f"GEOM-Drugs inventory exhausted after {len(seen)} eligible parents")


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
    for key, geometry, target, charge, multiplicity, evidence, issue in borylxat.optimized_rows(
        path
    ):
        case = make_case("borylxat", key, 0, geometry, target, charge, multiplicity, evidence)
        if issue:
            case = (*case[:7], "unresolved", issue, *case[9:])
        yield case


BDE_URL = "https://doi.org/10.6084/m9.figshare.12158646.v1"
BDE_SDF_SHA256 = "18f40da3c4a36954234d37ad37c72655e5ed0544913da8ca231f93d570b50348"
BDE_GZ_MD5 = "296709b4c0b0d2c9e750565e40e6e237"


def bde_cases(path):
    """Read optimized, explicit-H organic doublets from the published BDE SDF.

    The embedded bond table supplies the reference. AtomSpins independently
    distinguishes doublets from parent molecules with cancelling spin densities.
    """
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rb") as stream:
        for index, mol in enumerate(
            Chem.ForwardSDMolSupplier(stream, removeHs=False, sanitize=False)
        ):
            if mol is None:
                raise ValueError(f"Unparseable BDE SDF record {index}")
            if not any(a.GetAtomicNum() == 6 for a in mol.GetAtoms()):
                continue
            radicals = sum(a.GetNumRadicalElectrons() for a in mol.GetAtoms())
            spins = json.loads(mol.GetProp("AtomSpins")) if mol.HasProp("AtomSpins") else []
            if spins and (len(spins) != mol.GetNumAtoms() or not all(map(math.isfinite, spins))):
                raise ValueError(f"Invalid spin densities in BDE record {index}")
            total_spin = math.fsum(spins)
            if not radicals and abs(total_spin) < 1e-4:
                continue
            if radicals != 1 or abs(total_spin - 1) > 1e-4:
                raise ValueError(f"Inconsistent doublet annotation in BDE record {index}")
            Chem.SanitizeMol(mol)
            yield make_case(
                "bde",
                mol.GetProp("_Name"),
                0,
                mol,
                mol,
                Chem.GetFormalCharge(mol),
                2,
                {
                    "source_url": BDE_URL,
                    "sdf_record": index,
                    "source_smiles": mol.GetProp("SMILES"),
                    "spin_density_sum": total_spin,
                    "state_basis": "SDF radical and AtomSpins",
                },
            )


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
        release = borylxat.provenance(source)
        cases = borylxat_cases(source)
    elif dataset == "tmqmg-corrected":
        release = tmqmg.provenance(source)  # Validate with actionable checkout instructions first.
        cases = tmqmg_cases(source)
    else:  # BDE also supports explicitly attributed, custom SDF inputs.
        compressed = source.suffix == ".gz"
        if file_hash(source, "md5" if compressed else "sha256") == (
            BDE_GZ_MD5 if compressed else BDE_SDF_SHA256
        ):
            source_url = BDE_URL
            cases = bde_cases(source)
        elif (
            not source_url
            or source_url == BDE_URL
            or source_url.rstrip("/").endswith(("/4944855", "/22357962"))
        ):
            raise ValueError("BDE SDF checksum differs from Figshare release 12158646 v1")
        else:
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
    if dataset in _GEOM_NAMES:
        provenance["source_url"] = GEOM_URL
        if limit_molecules is None:
            provenance.update(
                seed=GEOM_SEED,
                conformer_policy="one released conformer per parent, SHA-256 rejection sampling",
                parent_sampling=(
                    f"first {GEOM_DRUGS_PARENTS} eligible distinct parents in seeded SHA-256 order"
                    if dataset == "geom-drugs"
                    else "all released parents in archive order"
                ),
            )
    if dataset == "bde" and source_url == BDE_URL:
        provenance.update(
            conformer_policy="one optimized geometry per carbon-containing doublet fragment",
            release={"figshare_file_id": 22357962, "sdf_sha256": BDE_SDF_SHA256},
        )
    if dataset == "borylxat":
        provenance.update(
            source_url=release["source_url"],
            release=release,
            conformer_policy="one released optimized geometry per component key",
        )
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
    parser.add_argument(
        "--limit-molecules",
        type=int,
        help="Prepare a pilot. For GEOM, import all conformers of the first N parents in archive order.",
    )
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
