# Native equivalence evaluator contract

The default `evaluate_equivalence` entry now uses the C++ kernel in
`src/cpp/src/utils/equivalence.cpp`. Python captures input diagnostics, decodes
MolGR properties, prepares explicit hydrogens and stereo, then packages native
results. It does not run graph matching, resonance search, SMILES generation,
InChI generation, or an RDKit resonance supplier.

This is molecular representation equivalence under the rules below, not a proof
of identical quantum electronic states. The implementation and rule coverage
changed: historical Python verdicts are comparison evidence, not an oracle.
The archived `_legacy_equivalence.py` is used only for comparison/tests; the
production entry never falls back to it. Its former contract is preserved in
[evaluator-legacy-contract.md](evaluator-legacy-contract.md).

## MoleculeData boundary for the native evaluator

The existing flat `MoleculeData` layout remains in use. `AtomData.isotope`
defaults to zero (no isotope label). `BondData.kind` defaults to `Covalent`;
`Dative` uses the existing one-based endpoints as donor -> acceptor, and
`CoordinationUnspecified` retains connectivity without inventing a direction.
For coordination edges `order` is unused (the RDKit adapter writes zero).
No conformer collection or separate public electronic-state model is introduced.

`rdkit_to_mol_data` takes a prepared RDKit molecule, adds all hydrogens as atom
nodes before coordination is projected out, and stores explicit Kekule orders
alongside aromatic flags. It does not modify the source. Hydrogen nodes and
their connections are fixed during resonance search; implicit hydrogen counts
are zero on output. Existing `x/y/z` fields retain coordinates when supplied.
The electron-property mask continues to describe explicit MolGR annotations;
native nonmetal radical values do not fabricate those annotations. Metal
unpaired counts come from the existing MolGR metal annotation.

The adapter batches primitive atom/bond rows through one native constructor;
it does not allocate Python `AtomData`/`BondData` wrappers or assign each field
through a separate binding call. Charge/radical totals are accumulated natively.
Indexed RDKit traversal is shared with stereo extraction. Public conversion
preserves stereo and coordinates by default (`include_stereo=True`,
`include_coordinates=True`). Equivalence omits coordinates and skips stereo
perception/extraction when `use_chirality=False`; stereo tags are still cleared
on the private graph before Kekulization to preserve its bond-order selection.
The evaluator combines atom validation, raw diagnostics and annotation decoding
in one source scan. It reuses element/isotope/charge fields and decoded MolGR
values in transfer rows; invalid metadata stays unknown and known zero/false
retains its presence bit. Only a changed explicit nonmetal radical assignment
requires a copy before `AddHs`, which creates the adapter's private graph.
Unannotated radicals and bond orders are read after chemical processing. Newly
added H nodes retain their actual isotopes, including isotopic H restored from
RDKit's `RemoveHs` metadata. Original input properties and coordinates remain
unchanged. All RDKit access stays in Python; the C++ build has no RDKit dependency.

Both native and Python Open Babel converters omit coordination edges and
preserve the atoms, explicit hydrogens and isotopes. These OBMol objects are
covalent projections: retain the original MoleculeData for coordination checks.
Extracting a projection back into MoleculeData does not restore discarded edges.
Passing Open Babel checks on the projection is not a proof of full coordination
or electronic equivalence.

Sparse `stereo` records retain tetrahedral winding, double-bond cis/trans,
square-planar, trigonal-bipyramidal and octahedral configurations. References
are one-based atom indices in the explicit-H graph, independent of bond-array
order. Zero denotes a virtual non-H reference (such as a lone pair); a zero
configuration records explicitly unspecified stereo, distinct from no record.
`stereo_groups` retains ABS/AND/OR atom groups. Tetrahedral winding is adjusted
for neighbor permutations; SP uses opposite reference pairs; TB/OH preserve
their reference order when reconstructing adjacency. No CIP labels, SMILES
roundtrip or coordinate perception are needed for this transfer.

Open Babel receives supported tetrahedral, cis/trans and square-planar records
using actual OB atom IDs. Stereo frames depending on omitted coordination edges,
TB/OH records and enhanced relative group centers stay in the original data.
They are not converted into absolute configurations on the covalent projection.
OB stereo extraction reads existing annotations without coordinate perception;
hydrogens at stereo centers must already be explicit.

The input adapter rejects queries, dummy atoms, unsupported bond types and
unsupported stereo classes (including allene and atropisomeric bond tags),
rather than silently dropping them. Bond members of enhanced stereo groups are
not yet supported. Directed dative output is supported; direction-unspecified
coordination cannot yet be exported to RDKit. This boundary is used by the default native evaluator.

## Execution order

1. **Python input boundary.** Capture raw diagnostics and decode annotations;
   copy the input only if annotation repair is needed. Public atom maps are not
   exported. Skip stereo processing when disabled;
   add all H atoms and obtain integer Kekule bond orders, then perceive radical
   bookkeeping once. Explicit MolGR unpaired assignments, including zero, are
   restored. Unknown/malformed electronic properties remain unknown. Unsupported
   queries, bonds, stereo or failed Kekulization return `INCONCLUSIVE`.
2. **Native validation and preparation.** Check atom/bond/stereo indices and
   declared totals. Build immutable topology plus charge/radical/bond arrays.
   Apply established metal Lewis-form and supported hypervalent normalization.
   Coordinates are not used to infer stereo or equivalence.
3. **Hard invariants.** Check original total charge, element and isotope
   inventories, explicit-H count, atom count and normalized metal formal states.
   Contradictions establish `NOT_EQUIVALENT`. Component radical counts alone
   are diagnostic and do not establish inequivalence.
4. **Shared topology candidates.** Joint color refinement ignores covalent
   order and electronic labels while retaining element/isotope labels, explicit
   H connections and requested coordination constraints. Unequal color counts
   reject immediately; equal colors alone never prove isomorphism. Covalent
   component color inventories provide a further necessary topology check.
5. **Joint mapping search.** Component charge inventories must match, and each
   atom's candidates must belong to compatible components. One bounded search
   serves both identity and resonance: assignments immediately remove used or
   adjacency-incompatible targets from remaining domains, with reversible
   updates on backtracking. Local identity is preferred only when electronic
   inventories allow it. After a topology/stereo witness is established, known
   annotation conflicts also prune partial assignments; missing fields remain
   compatible. Alternative symmetric mappings remain searchable. Equivalent
   terminal-H permutations are pruned unless used by stereo. Each complete
   mapping is checked for direct identity (aromatic edges may ignore arbitrary
   Kekule order), or passed to the regional proof below. Both require stereo
   and electronic annotations to agree under that same mapping.
   When neither input admits even a first carbene-normalization step, unequal
   covalent-component radical inventories skip mapping and return `INCONCLUSIVE`.
   A raw radical-count mismatch is never used to reject carbene equivalence.
   With active stereo records on exactly one side, the first topology mapping
   satisfying component charges establishes missing information and returns
   `INCONCLUSIVE` (`stereo_incomplete`). No symmetric mapping can supply the
   missing records. This search can prune equivalent terminal H permutations
   and never starts resonance; hard topology/charge checks and mapping budgets
   still apply. If both sides have active records, the normal joint search
   remains necessary, including when the record counts differ.
6. **Mapped normalization and regional proof.** Normalize supported
   carbene/zwitterion representations; identify potentially conjugated atoms
   from both aligned states. Saturated barriers and fixed stereo centers split
   the search. Frozen atoms and boundary bonds must agree. Regions are proved
   separately; their state spaces are not multiplied.
7. **Alternating cycles, then BFS.** Equal charge/radical arrays and balanced
   single/double changes have a direct alternating-cycle witness, including
   unmarked Kekule versus aromatic inputs. Other differing regions use a
   bidirectional BFS. The smaller nonempty frontier is expanded; exhaustion of
   one side does not discard the other side's possible witness.
8. **Result.** A joint graph/stereo/electronic witness establishes `EQUIVALENT`.
   Failed bounded search, unsupported transformations and validation errors
   yield `INCONCLUSIVE`. Exhausted topology mappings prove a missing sigma
   isomorphism; exhausted mappings with contradictory specified stereo prove
   stereochemical inequivalence. Restricted resonance exhaustion is never a
   proof of inequivalence. Component charge contradictions establish
   `NOT_EQUIVALENT` because no allowed move transfers charge between covalent
   components. Exhaustion after component-charge filtering is reported as a
   joint constraint failure, without falsely declaring topology alone different.

## Resonance rules and limits

The sigma skeleton and attached explicit hydrogen atoms are fixed. No hydrogen
transfer, tautomer enumeration or sigma-bond rearrangement occurs. Each BFS
state is a compact regional vector of formal charges, radical counts and integer
bond orders. Hash-table equality compares complete vectors, so hash collisions
cannot establish equivalence. No molecule objects are copied per state.

Supported local moves are:

| Move | Bond changes | Electron labels |
| --- | --- | --- |
| Pair polarization | `A=B → A–B` | `q(A)+1, q(B)−1` |
| Pair donation | `A–B → A=B` | `q(A)+1, q(B)−1` with donor pair/acceptor vacancy |
| Pair shift | `A–B=C → A=B–C` | `q(A)+1, q(C)−1` |
| Cation shift | `A–B=C → A=B–C` | `q(A)−1, q(C)+1` |
| Radical shift | `A–B=C → A=B–C` | `r(A)−1, r(C)+1` |

Candidates must satisfy local valence-electron parity, nonnegative nonbonding
occupancy, shell-size bounds, bond-order bounds and a bounded charge separation.
These conservative Lewis rules do not cover every chemically plausible form.
Carbene normalization is a separate representation convention: supported
`C−=X+` converts to neutral single-bonded low-valence carbon with two bookkeeping
radicals. It is recorded separately and is not advertised as spin conservation.

The admissible remaining-step bound is the maximum of half the L1 charge,
radical and bond-order distances, rounded upward. Candidates are ordered by
this bound within a BFS layer. A `2k`-bond carrier corridor is **not** used when
local charge-pair creation is enabled, because its assumptions would not hold.

Defaults:

- `use_chirality=False`: compare without stereochemical constraints. Set it to
  `True` explicitly to enable stereo comparison. This default is shared by the
  Python evaluator, boolean wrapper, native binding and C++ options; public
  MoleculeData conversion still preserves stereo by default.
- `max_resonance=1000`: at most **2000 visited BFS states in total**, including
  both roots, both directions, all regions and attempted mappings.
- `max_resonance_steps=24`: maximum joined elementary path length per region.
- `max_mappings=10000`: completed mappings in the joint topology/electronic search.
- `max_mapping_nodes=100000`: mapping DFS node budget.
- Input size limit: 4096 atoms including explicit H.

The defaults expand the original `max_resonance=50` / 12-step settings and the
intermediate `250` and `500` / 24-step settings. Explicit caller limits, including zero,
are still honored. The tmQMg replay comparison
found most of the additional witnesses at `max_resonance=250`; 500 added none
in that sample, and 1000 added two. The larger step bound permits longer paths,
although 12 versus 24 steps did not change that sample's decisions.

Direct identity, carbene normalization and alternating-cycle proofs do not
consume the BFS state budget. Exhausted/truncated searches stay inconclusive.
A found witness is sufficient even if some alternative branches were pruned.
The native Python API and its `ResonanceDetail` output no longer expose
`resonance_flags`: supplier flags have no meaning for this kernel. Remove that
argument from callers; passing it raises `TypeError`, not a chemical
`INCONCLUSIVE` result. Options after `max_resonance`, including
`ignore_metal_coordination_bonds`, are keyword-only so a positional legacy flag
cannot silently change coordination policy. The archived Python comparison
engine retains its own supplier flags.

### Reusing resonance work across mappings

Within one evaluation, normalize the left representation once and skip carbene
normalization scans when no first normalization step can apply. Check charge
and radical conservation for **all** candidate regions before any regional BFS.

Failed normalized targets are memoized by complete charge/radical/bond-order
arrays in the left input's atom frame. Regional keys additionally include atom
and edge membership, fixed boundary valences and both endpoint states. Stereo
and annotation compatibility are checked under each mapping before memo lookup.
A successful regional entry retains its concrete move sequence for witness
replay. Failed searches remain `INCONCLUSIVE`; their reuse is scoped to the
same fixed rules/step limit and monotonically decreasing shared state budget.
No cache persists across inputs, calls or budget changes.

Cache insertion is lazy: first-mapping successes need no entries. A later
regional failure preserves any earlier regional proofs before discarding the
partial whole-molecule witness. Entries are bounded to 1024 and 4 MiB of tracked
key/proof payload per evaluation (plus container/allocator overhead); reaching
that cap disables new insertions, not chemical search. Cache hits consume no
additional BFS state budget. Native diagnostics expose `resonance_cache_hits`,
`region_cache_hits` and `region_constraint_rejections`.

## Coordination and stereochemistry

`ignore_metal_coordination_bonds=True` remains the default: `N->[Cu]` and
`N.[Cu]` may match. All metal-incident connectivity is omitted after established
metal Lewis normalization, but metal identity/formal state and jointly known
metal unpaired counts remain constraints. Stereo frames involving removed
metal edges are excluded from comparison.

The evaluator omits these frames during Python extraction, before enforcing
their reference counts. In particular, an alkene endpoint with extra metal
coordination need not fit a four-reference frame on the covalent projection.
Public MoleculeData conversion continues to retain full stereo and rejects
unsupported full frames. Native comparison applies the same active-record
policy to relative groups, fixed resonance centers and mapping symmetry guards.
A valid double-bond frame may share a reference atom between its two ends;
references must be unique within each end and connected to that end. Malformed
records are still rejected, including in direct MoleculeData input.

With `False`, the same atom mapping must preserve which atoms attach to each
metal, including metal–metal edges, and pass all other checks. Original metal
bond orders/directions remain normalized by the established Lewis convention.
Nonmetal dative edges retain direction and exclude incident atoms from resonance
search. Converters to Open Babel still omit coordination edges; the kernel
compares the original MoleculeData graph directly.

Stereo comparison is disabled by default. When explicitly enabled with
`use_chirality=True`, native comparison supports tetrahedral, double-bond,
square-planar, trigonal-bipyramidal and octahedral records, adjusting reference
permutations under atom mapping. AND/OR tetrahedral groups admit consistent
whole-group inversion, not independent center flips. Group IDs are not chemical
labels. Specified versus missing/unknown stereo is inconclusive. More general
relative non-tetrahedral group differences are conservatively inconclusive.
When disabled, isotopes remain mandatory labels.

The non-tetrahedral permutation tables are redistributed under RDKit's BSD
license; see `src/molgr/data/RDKIT_STEREO_LICENSE.txt`. The kernel does not link
RDKit C++. Open Babel supplies the existing element information/conversion
infrastructure; no OBMol conversion, cloning or perception runs inside BFS.

## Electronic annotations and evidence

Presence bits distinguish known zero/false from unknown. Compare jointly known
metal unpaired counts under the accepted mapping. Active lone-pair counts are
reconstruction annotations, not a complete Lewis-pair inventory: equal known
counts are compatible; unequal counts require local balance of
`2 × active_pairs + formal_charge + bond_order_sum + radical_count` alongside
an actual structural witness. Known unresolved centers defer the pair check.
Missing fields never manufacture zero or veto an otherwise valid proof.

`atom_mapping` is zero-based, left explicit-H input → right explicit-H input.
`resonance_witness` records paths from both roots to a meeting state; `side=1`
indices are also expressed in the **left** frame. Moves from each side are in
execution order. `carbene_normalization` is explicitly distinguished from BFS
moves. `kekule_cycle` carries a closed alternating atom trail and the first
bond delta in `atoms[2]`; apply the cycle simultaneously. Witnesses start after
native metal/hypervalent preparation. Failed mappings never leak partial paths.

`native_stats` reports mapping counts/nodes and native preparation, mapping and
resonance milliseconds. Python input/binding/result overhead is outside these
three stage timers. Old canonical/resonance SMILES and InChI diagnostics remain
`None`; no positive decision relies on an identifier. `component_electrons`
currently remains `not_evaluated`, and `bounded_search.exhaustive` is never true
for the restricted resonance rules.

The C++ API accepts `const MoleculeData&` and releases the GIL through its Python
binding. Callers must keep those inputs unchanged until evaluation returns.
The compatibility `check_equivalence` boolean is false for both NEQ and INC;
scoring must inspect `decision` to preserve all three outcomes.

## Verification and performance reproduction

Native tests independently replay charge/radical/bond moves, exercise cation,
anion and radical cases in both directions, verify region splitting/budgets,
check stereo permutations, and prohibit calls to Python search or serializers.
Historical implementation-specific tests explicitly import the archived engine.

Run the sequential saved-pair benchmark with the rebuilt extension and matching
Python/RDKit environment:

```bash
python scripts/benchmark_native_equivalence.py path/to/results.csv \
  --limit 0 --repeat 2 --max-resonance 100 --out /tmp/native-comparison.json
```

The report includes input and binary/source hashes, latency percentiles, native
stage timings and every changed verdict. It measures evaluation including
preprocessing, excluding CSV/SMILES parsing and XYZ reconstruction. Saved SMILES
cannot reproduce electronic annotations absent from the CSV; this is not a
full tmQMg reconstruction rerun or an accuracy claim.

Replay checkpoints now fingerprint the compiled MolGR binary and Python package
sources as well as the evaluator wrapper. A wrapper snapshot alone is no longer
self-contained; replay requires the corresponding package/build. Never mix this
implementation with frozen historical accuracy results or checkpoint identities.
