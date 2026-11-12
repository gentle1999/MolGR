# RDKit SMILES round trips with explicit hydrogens

`xyz_to_rdmol()` and `iter_xyz_to_rdmol_batch()` validate successful outputs
against native RDKit `MolToSmiles()` / `MolFromSmiles(..., removeHs=False)`.
The exported `mol_from_smiles()` is a convenience wrapper around that same
strict reader; it does not bypass any sanitization checks.
Both reconstruction backends use the same check, including calls with
`make_stereochemistry=False`.

The check preserves element, isotope, formal charge,
hydrogen count, connectivity, bond type, aromaticity, dative-bond direction,
and atom maps. It compares the writer's actual atom correspondence;
a different symmetry mapping or resonance form cannot conceal a graph mismatch.
Atom ordering may change. All explicit hydrogen nodes must survive; MolGR
does not remove H or validate a separate hydrogen-folded graph.
Every output atom has `NoImplicit=True`. All H are atom nodes: neither
implicit H nor atom-attached H counts (such as the H in `[NH]`) are allowed.
The validator rejects source molecules that do not meet this explicit-H model.
Radical counts inferred by the SMILES reader are not used to judge graph
identity or to reconstruct MolGR's electronic state.

The public entry point is `molgr.validate_smiles_roundtrip(mol)`. It accepts an
RDKit Mol, uses the currently imported RDKit for both writing and reading, and
returns the validated SMILES. Failure raises `molgr.SmilesRoundtripError`;
`str(error)` describes the failure and `error.details` contains available
mismatch details. It does not modify the input or require a second RDKit
installation. The guarantee is within this version, not across versions.

The validator compares the native sanitized readback directly, without setting
output-only bracket properties on that temporary molecule. It fetches only the
writer's atom order and checks the same atom/bond fields under that mapping.
The public `mol_from_smiles()` still prepares explicit-H graphs for subsequent
writes. These implementation optimizations do not disable default validation,
skip sanitization, cache a previous verdict, or sample only some molecules.

With `make_stereochemistry=True`, the returned Mol's stereo is assigned from
the input 3D coordinates using MolGR's existing RDKit perception workarounds.
SMILES readback is not the authority for those assignments. Toolkit stereo
serialization differences neither reject a reconstruction nor overwrite its
coordinate-derived tags. Stereo preservation through SMILES can be checked
separately with `check_stereochemistry=True`; that optional diagnostic compares
tetrahedral, E/Z, SP, TB and OH reference frames under the writer's atom mapping.

```python
from molgr import mol_from_smiles, validate_smiles_roundtrip
from molgr.interface import xyz_to_rdmol
from rdkit import Chem

mol = xyz_to_rdmol(xyz, total_charge=0, spin_multiplicity=1)
smiles = Chem.MolToSmiles(mol)
params = Chem.SmilesParserParams()
params.removeHs = False
restored = Chem.MolFromSmiles(smiles, params)
# Equivalent convenience reader, retaining the explicit-H writing convention:
restored = mol_from_smiles(smiles)

# Also available for an existing RDKit molecule. Returns validated SMILES
# without modifying the input, or raises SmilesRoundtripError.
smiles = validate_smiles_roundtrip(mol)

# Optional toolkit diagnostic, beyond the default reconstruction contract:
smiles = validate_smiles_roundtrip(mol, check_stereochemistry=True)
```

Validation applies to the default canonical, isomeric SMILES writer and a
native sanitized RDKit reader with `removeHs=False` in the installed RDKit version. Using
`sanitize=False`, `isomericSmiles=False`, a different aromaticity model,
`removeHs=True`, another toolkit/version, or modifying the molecule afterwards is outside this
contract. Ordinary SMILES does not preserve coordinates, arbitrary properties,
MolGR's lone-pair/unresolved-center/metal-spin annotations, or enhanced relative
stereo groups. The optional stereo diagnostic rejects relative groups or
explicitly unknown stereo that would lose their meaning.

## Metals with finite RDKit valence tables

`make_dative_bond()` converts coordination only for metal elements whose
**current RDKit valence table contains no `-1` entry**. The `-1` entry permits
unrestricted valence; membership in the main group alone is not a reason to
change a metal's bonds or charges. Contact selection and reconstruction search
algorithms are unchanged. Conversion occurs before aromaticity/stereo perception
and native SMILES validation, retaining connectivity, explicit H nodes and the
molecular total charge.

An independent scan covers all 93 elements classified as metals by MolGR within
Z=1..118, formal charges −3 through +6, and 1 through 8 incoming neutral water
donors. An isolated ion must first pass the strict property-cache check. These
are representation probes, not claims that every charge/coordination combination
is chemically realistic. The elements exhibiting acceptor-valence failures match
those with finite valence tables:

| RDKit version | Elements selected for covalent coordination normalization |
| --- | --- |
| 2024.03.5 | Be, Al, Ga, Rb, In, Sn, Cs, Pb, Bi, Fr |
| 2026.03.1 | Be, Al, Ga, In, Sn, Cs, Pb, Bi, Fr |

Rb's table gains an unrestricted entry in the newer version, so it automatically
keeps dative bonds there. K, Li, Na, Mg, Ca and Tl are excluded in both versions,
as are the transition and f-block metals, including Zn, Cd and Hg. Selection is
by element's native table; an individual ion of a selected element may still
have accepted a particular dative coordination count before normalization.

For each bond-order unit, the donor formal charge increases by one and the
acceptor formal charge decreases by one:

| Donor before coordination | Ordinary bond | Donor charge change | Metal charge change |
| --- | --- | --- | --- |
| Neutral lone-pair donor | Single | +1 | −1 |
| Isolated −1 donor | Single | +1, to zero | −1 |
| Isolated −2 donor | Double | +2, to zero | −2 |
| Isolated −3 donor | Triple | +3, to zero | −3 |

A donor bridging multiple selected metals first receives one single bond to
each acceptor; any remaining negative-charge budget adds bond order to the
shortest contacts, capped at triple bonds. For example, an oxide donor bridging
two Al centers forms two single bonds and becomes neutral. A resolved singlet
lone-pair center also uses a charge-separated single bond to a selected metal;
transition-metal double/triple-bond rules are retained. Existing incoming dative
contacts are normalized as well, and repeated postprocessing does not transfer
the charge again. Ligand-internal bonds are not rewritten to force acceptance.

K contacts are left in their original dative representation with their original
charges, including the shared aromatic C→Ni/C→K donor in `tests/data/xyz/test.xyz`.
No K-specific rollback or organometallic-cleanup exception is needed.

In the reported four-coordinate Al³⁺ fixture, four incoming dative contacts
become four single bonds, Al's formal charge becomes −1, and donor charges
increase correspondingly. The total molecular charge stays +1. This is a
change of formal-charge representation, **not an assertion that the aluminum
oxidation state is −1**. Both tested RDKit versions sanitize the final graph
with their original rules.

No reader exception for metal valence remains. If the final representation
fails native sanitization or changes during readback, reconstruction reports
that failure. The native round-trip check validates serialization of the final
representation; it does not establish chemical plausibility for every possible
coordination geometry or resolve arbitrary multicenter bonding. The convenience
reader does not rewrite an externally supplied Al³⁺ dative SMILES to hide a
sanitization failure.

## Failure behavior

MolGR normalizes every atom before serialization. RDKit can write a
singlet zero-H carbon as bare `C` even with `SetNoImplicit(True)`, causing the
reader to add hydrogens. MolGR sets `NoImplicit=True` and RDKit's `smilesSymbol`
property to the unchanged element symbol on **every atom**. This requests
bracketed output such as `[C]` and aromatic `[c]` from a direct
`Chem.MolToSmiles(mol)` call, so readback also disables implicit H everywhere.
It preserves the H count without
changing bonds, charges, atom maps or actual electronic assignments. The
reader may assign radicals to the bracketed atom; those inferred counts are
not a reconstruction of the source electronic state.

Any remaining graph loss is rejected with a
`ReconstructionError` whose diagnostic code is `SMILES_ROUNDTRIP_FAILED` and
stage is `rdkit.smiles_roundtrip`. Atom mismatches include the one-based source
atom number, changed field and before/after values in `diagnostics.details`.

Batch results carry this diagnostic in `status` and continue with other items,
unless `raise_on_error=True`. The existing `return_suspicious` failure policy
can return an untrusted initial-perception molecule with
`_MolGRReconstructionStatus="suspicious_fallback"`; **that fallback is not a
successful reconstruction and has no round-trip guarantee**.

For applications that deliberately use MolGR-specific electronic states and
serialize through another format, the previous permissive behavior remains
available explicitly:

```python
from molgr import MolGRConfig

config = MolGRConfig()
config.interface.require_smiles_roundtrip = False
mol = xyz_to_rdmol(xyz, config=config)
```

Disabling validation removes the guarantee. It does not undo the conversion
fix: stale Open Babel aromatic flags are discarded before perception, and a
failed Kekulization is never allowed to corrupt the returned integer bonds.
Low-level conversion helpers also preserve independent electron annotations;
they do not enforce this high-level SMILES contract automatically.

## Regression inputs

For integer-bond reconstruction outputs, stale aromatic flags are cleared
before fresh perception. This refresh runs in place without copying the molecule:
only radical counts and graph fields that perception can change are saved.
RDKit temporarily infers radicals from valence for aromaticity perception; MolGR's
original radical counts are restored even on failure. A failed perception also
restores bond types, aromaticity and conjugation, and invalidates intermediate
computed caches. Inputs with actual `BondType.AROMATIC` bonds still use a
protective copy because they require potentially destructive Kekulization.
This preserves the authoritative multiple bonds. In the reported triple
bond cases, RDKit can assign `IsAromatic=True` while retaining `BondType.TRIPLE`;
the SMILES still contains `#`, and normal sanitized readback preserves that
combination. Reading with `sanitize=False` omits aromaticity perception and is
not a valid comparison of the resulting aromatic flags.

SMILES text can change when corrupted bonds are repaired or when every atom
gets explicit brackets. The guarantee concerns the mapped molecular graph,
not byte-identical strings or coordinates serialized through SMILES.

The public regression suite uses the repository's existing XYZ fixtures and
synthetic molecules. It covers repeated native explicit-H graph round trips
with both reconstruction backends, failed Kekulization, perception rollback,
zero-H electronic states, aromatic triple bonds, finite-valence coordination,
and graph-only versus optional stereo validation. Coordinate-based stereo
perception is covered by the existing ideal-geometry regression tests.

The existing four-coordinate aluminum fixture separately checks that native
reading accepts the covalent representation with all H retained and total
molecular charge unchanged. No external input archive is required to run
these tests.
