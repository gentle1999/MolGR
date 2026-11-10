> Historical Python evaluator contract. The production implementation is described in [evaluator-contract.md](evaluator-contract.md).

# MolGR operational evaluator contract

The evaluator measures graph/cheminformatics-level equivalence for MolGR reconstruction. It does not prove identical quantum electronic states or exact localization of radical electrons in a transition-metal Lewis representation.

Policy designation: `molgr-evaluator-v2` (an evaluator semantic-version change, not a reconstruction or accuracy-improvement release).

Transition-metal coordination and Lewis-form representations can preserve structural molecular-graph equivalence, formula, connectivity, and total formal charge while differing in RDKit-local radical bookkeeping. Therefore component-electron mismatch remains diagnostic evidence but no longer independently establishes graph non-equivalence.

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
coordination cannot yet be exported to RDKit. This data boundary is not yet wired
into the evaluator's comparison or resonance algorithms.

## Hard invariants

The following independently establish `NOT_EQUIVALENT` when they fail:

- heavy-atom formula;
- explicit-hydrogen formula;
- atom count;
- metal identity/formal-valence signature;
- non-metal connectivity;
- raw whole-molecule total formal charge.

These checks are not weakened by the component-electron policy.

The raw charge gate compares the sum of atom formal charges on each input before
metal-bond standardization, metal removal, or resonance normalization. Its values
and status are recorded in `invariants["raw_total_formal_charge"]`; the existing
`formal_charge` check remains a prepared-organic diagnostic. Charge mismatch
returns `NOT_EQUIVALENT` before any positive comparison path: resonance cannot
equate species that differ in total electron count. The local regression uses
`[S]C(=NN=Cc1ccccc1)N` versus `[S-]C(=NN=Cc1ccccc1)N` (charges 0 and -1),
which previously matched after thiosemicarbazone normalization.

## Metal coordination option

`evaluate_equivalence` and `check_equivalence` accept
`ignore_metal_coordination_bonds=True` by default. This preserves the historical
graph-level result: `N->[Cu]` and `N.[Cu]` are equivalent after the metal is
removed. Set the option to `False` to require the same metal-incident edge
connectivity under atom symmetry. The same mapping must support the organic
structural proof and jointly known electronic annotations, even when no
annotations are present. Separate mappings for coordination topology and organic
equivalence do not constitute a joint witness. `invariants["metal_coordination"]` reports
`not_evaluated` in the default mode, and `passed` or `failed` in strict mode.

The strict check runs after the existing metal-bond standardization. It compares
which atoms are connected to each metal, including metal-metal edges, while
allowing the existing Lewis-form bond-order normalization. It does not require
identical original metal-ligand bond orders or prove a physical coordination
geometry. Benchmark callers retain the default until a benchmark explicitly
opts into strict coordination connectivity.

The coordination invariant records the preliminary topology check. Passing it
alone does not prove the existence of a compatible organic/electronic mapping;
the final result can still be `INCONCLUSIVE`. For example, attaching Cu to the
opposite oxygen in `[O]CC[O-]` cannot use an oxygen swap to prove coordination
while keeping the oxygens fixed to prove localized charge/radical identity.

## Positive evidence

After the hard structural checks pass, the evaluator may return `EQUIVALENT` for:

- normalized graph identity;
- established carbene/zwitterion normalization;
- successful resonance equivalence.

For open-shell forms, a mapped elementary three-center radical/bond-order
shift is explicit positive evidence. The mapping search is bounded and its
attempted count and limit status remain diagnostic; reaching the bound returns
`INCONCLUSIVE` unless an accepted structural path has already matched.

The witness requires equal, positive total RDKit radical labels on its two
normalized inputs. If this necessary condition fails, only this witness is
skipped: its search metadata remains unattempted, and all existing comparison
paths continue. Radical-label mismatch is not a top-level `NOT_EQUIVALENT` gate.

InChIKey is diagnostic only. Agreement of either the prepared input key or the
normalized organic key, even after all hard checks pass, remains `INCONCLUSIVE`
without an accepted structural path. Disagreement is not a hard veto. The
normalized organic key is computed after metal removal; it is not an identifier
for the full original molecule. Standard InChI can normalize distinct proton
and Lewis representations, so matching keys alone do not establish the
evaluator's graph/electron equivalence.
An `INCONCLUSIVE` result may retain `relation=identifier_equivalence` and
`method=inchi_key` to identify this diagnostic signal; neither field implies a
positive equivalence proof. See the [InChI Trust technical FAQ](https://www.inchi-trust.org/technical-faq/)
for standard InChI normalization and InChIKey layer semantics.

This policy changes at least the directed `WEXMIW_PATTERN` benchmark control
from `EQUIVALENT` to `INCONCLUSIVE`: its keys agree, but bounded resonance search
finds no structural witness. The previously reported 74,547-slot replay and
source hash describe an earlier evaluator snapshot; a new full replay is needed
before claiming aggregate decision compatibility for this revision.

With `use_chirality=False`, InChIKey generation removes stereochemistry and
conformers only from its private molecule copy, preventing InChI from inferring
stereochemistry from 3D coordinates. Input molecules remain unchanged.
With `use_chirality=True`, coordinate-derived stereochemistry remains enabled.

## Diagnostic signals

Identical normalized canonical graphs already prove nonmetal connectivity,
component electron compatibility, and stereochemistry under the selected policy.
That path records these invariants as `passed`, with an explicit implication
reason and `null` per-side values; redundant topology and component keys are not
computed. Electronic annotations and strict coordination still require a
compatible mapped proof before EQ can be returned.

For nonidentical normalized graphs, the per-component electron signature is
computed and persisted in `invariants["component_electrons"]`, together with
aggregate radical counts, raw diagnostics, normalization effects, and
contradiction metadata. A mismatch is diagnostic by default. RDKit radical
bookkeeping, metal–ligand Lewis representation, and component-local electron
accounting do not independently establish `NOT_EQUIVALENT`.

RDKit atom radical flags alone still do not establish a global electronic-state
contradiction. Jointly known MolGR annotations additionally constrain positive
structural evidence as described below.

An accepted structural result can intentionally contain both:

```text
decision = equivalent
relation = resonance_equivalence
component_electrons.status = failed
```

## Known electronic annotations (v2 integration)

The evaluator reads `MOLGR_METAL_UNPAIRED_ELECTRONS`,
`MOLGR_LONE_PAIR_COUNT`, and `MOLGR_UNRESOLVED_TWO_ELECTRON_CENTER` before
normalization. Known zero/false is retained through Open Babel/RDKit conversion,
coordination consumption, and explicit property-preserving serialization.
Absent or malformed fields stay unknown; generic reference/baseline molecules
do not acquire synthetic zero annotations.

Every positive comparison path, including identical canonical SMILES, must
admit a compatible atom mapping for jointly known occupancy information.
Alternative symmetric mappings are considered. An atom-mapped instance of the
existing graph, carbene/zwitterion, resonance, or elementary three-center proof
must support that same mapping; an unrelated global match is insufficient.
Metal annotations follow the existing metal-state multiset contract when metal
coordination bonds are ignored. Strict mode also aligns them to the matched
metal coordination connectivity.

For a fixed atom mapping, exact graph identity is checked directly against atom
elements, isotopes, charges, radical counts, attached hydrogens, aromaticity and
mapped bonds, including dative direction. This avoids copying, atom-map labeling,
and canonical SMILES generation for an already identical mapped graph. Defined
stereochemistry retains the existing stereo-aware canonical proof because raw
chiral tags depend on neighbor order.

Lone-pair counts describe **reconstruction-active pairs**, not all Lewis lone
pairs. Equal known counts are compatible. A changed count may be transported
by a supported mapped representation change when the change in twice the
active-pair count balances formal charge, bond-order sum, and RDKit radical
bookkeeping locally. This balance is never independent positive EQ evidence.
An unresolved-center marker means occupancy is deferred, not one/two radical
electrons; its zero occupancy placeholders do not conflict with a resolved
annotation. Different resolution flags alone are not different physical states.

Unknown fields never veto otherwise sufficient structural evidence. The same
known-information rule applies symmetrically to all methods and references.
`invariants["electronic_properties"]` records known/unknown/comparable field
counts, deferred centers, and mapping progress alongside the existing raw
property values. It is not a confidence score or an additional scoring system.

An incompatible mapping is not a negative verdict. Only exhaustive mapping
failure with an irreconcilable localized conflict and no eligible electronic
rearrangement establishes NEQ in this minimal integration. Other unresolved
annotation conflicts, mapping/resonance limits, or search errors yield INC,
not NEQ. A compatible mapped positive witness is sufficient even if further
alternative mappings were not enumerated. Timeouts retain the existing caller
contract. The mapping bound is 10,000 and the resonance bound is unchanged.

Mapping enumeration first requests one mapping. Only if that mapping fails to
provide a compatible structural witness are the remaining bounded alternatives
materialized. A successful first mapping therefore does not generate thousands
of explicit-hydrogen permutations. `mappings_examined` counts mappings actually
checked, and `mapping_limit_reached` records consumption of the configured budget.

This remains evaluator v2. It requires a new source hash and output snapshot;
it must not overwrite or be mixed into the frozen `cace4ccc8` results.


## Inconclusive results

The evaluator returns `INCONCLUSIVE` when evidence is insufficient, including:

- prepared-organic identifier-only agreement;
- a bounded resonance search that reaches `max_resonance` without a match;
- a bounded open-shell topology-mapping search that reaches its limit without a witness;
- other comparison failures where no safe negative conclusion is available.

The resonance bound remains configuration-controlled; this contract does not increase it.

Ordinary resonance-search exceptions are not negative evidence. Failed or partial
enumerations are not cached as completed results. The evaluator continues to the
existing independent positive comparison paths; if none succeeds, it returns
`INCONCLUSIVE`. Search failure sets `bounded_search.exhaustive = False`, and
`reason` includes the exception type and message even when an independent path
returns `EQUIVALENT`. A later evaluation after recovery needs no manual cache
clear. `TimeoutError` still propagates to the caller; no retry or larger search
bound is introduced.

Local charge migration uses a two-ended, multi-source breadth-first search.
Each side maintains its visited states and current depth layer. The smaller
nonempty layer is expanded first; equal sizes favor the side with fewer expanded
states. Every new state is immediately checked against the opposite visited
set. A meeting ends the search, without completing either closure. Each state
is queued once per side, and parent links retain the actual local paths from
the two seeds to the common witness. Canonical identity and a direct local-edge
probe can still terminate before preparing the second side. If the direct probe
misses, BFS reuses its validated neighbors and keys. Stereo removal is performed
once per search source when chirality is ignored.

Both ends generate validated **outgoing** migrations. RDKit sanitization and
valence guards can make this generator asymmetric, so this is a search for a
common reachable witness, not a shortest directed path from one input to the
other. An empty or capped frontier does not stop the opposite frontier: its
visited states remain potential meeting targets. A negative local result needs
both forward closures exhausted without limits or validation failures. Special
normalizers remain outside this graph and are not treated as invertible edges.

After the direct local-edge probe misses, an optional regional proof can avoid
enumerating combinations of independent migrations. It runs only when that
first layer exposes at least two distinct successors; unbranched first layers
continue through the general BFS without alignment overhead. This is a
performance heuristic, not a claim that later layers cannot branch.
Alignment first verifies
equal atom indices or existing unique atom-map labels, including attached added
hydrogens. Verification checks immutable atom labels and every topology edge;
equal indices alone are not evidence. Otherwise it requests one topology
isomorphism ignoring charges and bond orders. It does not enumerate symmetry
alternatives to improve this optional proof. Topology copies are made only for
that fallback alignment, after the cheaper witnesses have failed.

Under this mapping, connected components of changed bonds define independent
candidate regions, including regions joined by unchanged bonds in one molecule.
Unchanged bonds are frozen for this restricted attempt. Each region uses compact
charge/bond-order states and BFS, without molecular copies, sanitization, or
SMILES generation during expansion. The attempt first uses the bond-edit lower
bound as its step budget. For a budget of `k` two-bond migrations, vertices must
satisfy `distance_from_sources + distance_to_targets <= 2*k`; remaining charge
distance and bond-edit lower bounds further prune states. This is a bound on
paths within the chosen budget, not a claim that molecular shortest distance
upper-bounds the number of required migrations. Novel-state budget is shared
across regions rather than multiplied by their count.

Only a successful abstract path is replayed on the full molecule, sanitizing
each actual step and checking that no other atom/bond state changed. Final
mapped graph identity is required. `regional_region_count`,
`regional_state_count`, and `regional_pruned_moves` describe a successful
regional proof; ordinary per-side form counts describe the validated path, not
the projected search states. Stereo-defined inputs retain the general search.
Failure of this first mapping, frozen-region search, or replay falls back to
the general search and is never negative evidence. In particular, equal
endpoint states outside a region do not prove those atoms cannot participate
in another valid intermediate path.

Expansion starts at eligible negatively charged atoms using a fixed adjacency
table. Only states with an applicable local move need an electronic snapshot.
Within fixed atom numbering, a proposal key records charges, radicals, aromatic
flags, hydrogen/implicit-H settings, bond types, and retained stereo information.
Duplicate proposals reuse an already validated molecule before copying or
sanitizing it again. Canonical keys remain necessary for comparison between
different molecular graphs; a raw index-based state vector is not such a proof.
Sanitization can change more than the edited atoms, so a resulting state's
snapshot is read from the actual molecule when it is expanded.

Complete neighbor expansions and comparison keys are reused across local and
supplier stages within one evaluation. A partially consumed neighbor generator
is never memoized as complete. Caches have entry and transition-reference bounds
proportional to the search budget; eviction only causes recomputation, never a
negative conclusion or a change to the search limit.

After the initial two-ended local search, the supplier iterators advance alternately
and compare their prefixes immediately. A common state ends the search. Otherwise
the accumulated seeds, keys, and local transition context initialize the same
two-ended BFS with multiple roots per side. A supplier seed is a root, not a
fabricated one-step local migration. The initial local stage retains at most
`max_resonance` distinct states per side including its source. After enumeration,
each side retains its source and up to `max_resonance` supplier forms, plus up to
`max_resonance` newly discovered local states. These are separate supplier/local
budgets, consistent with the previous queue bound; reported distinct-state
counts include seeds and can exceed either individual budget. Supplier duplicates
still count toward the supplier budget. Local truncation is recorded when a
new state cannot be retained; already discovered states remain usable as targets.

The live evaluator passes molecule objects throughout this process and
orders atoms by canonical ranks, with explicit H atoms after heavy atoms, before
enumeration. It does not reconstruct resonance sources by parsing SMILES, so
source atom annotations and hydrogen identities survive preparation. This does
not assume an inverse-step API or complete coverage of RDKit resonance forms.

`bounded_search.stop_reason` distinguishes `not_attempted`, `target_found`,
`exhausted`, `limit_reached`, `validation_failed`, and `error`. An early positive witness is sufficient
for EQ, but does not mean the search was exhaustive. Supplier truncation is tracked
from the number of supplier forms consumed, before canonical deduplication;
`supplier_limit_reached` and `local_limit_reached` distinguish the two budgets.
Consuming the full supplier budget is conservatively treated as possibly
truncated; local truncation records an omitted new state. Limits from
earlier search stages are retained if later stages also fail to find a witness.
These flags describe exhaustion only within the configured transformation model,
not all chemically possible resonance states.

Failed candidate sanitization and missing supplier forms set `validation_failed`.
An invalid local candidate is not yielded as a witness. Independent validated
positive evidence is still sufficient for EQ; without such evidence, validation
failure prevents an exhaustive negative conclusion. Timeouts still propagate.

The optional SMILES-input compatibility helper retains its bounded cache;
the live evaluator uses the per-evaluation contexts above. Its target-directed
cache entries include the target set in their key and retain
their stopping metadata. A cached prefix cannot be reused as a complete closure
for a different target. Cache entries contain structural search results, never
complete equivalence decisions that depend on possibly unknown atom annotations.

## Reproducibility

Benchmark releases should pin the RDKit major/minor version and persist the evaluator source hash, runtime inventory, input manifest hash, `use_chirality`, and `max_resonance`. RDKit-dependent resonance and sanitization behavior must be reported as version-sensitive rather than treated as proof of physical electronic-state identity.

Current authoritative validation uses RDKit `2026.03.6`. Resonance classification is not fully cross-version stable, so results involving resonance enumeration or normalization must be interpreted against the pinned runtime and revalidated when the RDKit version changes.

When two generated resonance sets intersect, search stops at the first witness
in the deterministic traversal order. A direct target takes precedence; otherwise
the recorded hit is the lexicographically smallest intersection of the prefixes
actually explored. It need not be the smallest hit in the full closure. Hit
selection remains independent of Python's hash seed, while bounded searches and
their witnesses can change with RDKit versions or evaluator source revisions.

## Benchmark scoring and reference eligibility

The three benchmark callers preserve `evaluator_decision`: EQ maps to
`equivalent=True`, NEQ to `False`, and INC to `None`. CSV output keeps the
decision even when the boolean cell is empty. tmQMg scores the original RDKit
candidate graph; SMILES roundtrip results remain separate diagnostics.

Reference eligibility is independent of evaluator and method outcomes:
`reference_status` is `valid`, `excluded`, or `unresolved`, with the supporting
`reference_reason`. Explicit confirmed-error metadata with a reason, or the
existing independent reference/XYZ formula contradiction, permits exclusion.
Missing/unparseable references and unresolved review records are not automatically
confirmed errors. The historical manual whitelist is read-only reference-quality
evidence, never an equivalence rule. `accepted_reference_issue` requires explicit
`reference_answer_wrong=true` and a reason to establish exclusion;
`accepted_ambiguous` and unconfirmed entries remain unresolved.

`total_count` includes every input. `count` aliases `reference_valid_count`;
confirmed exclusions and unresolved references have separate counts and do not
enter that denominator. Within valid references, `success_count`, `fail_count`,
`inconclusive_count`, and `runtime_failure_count` partition the denominator.
INC and failed/skipped methods retain their valid-reference denominator slots.
Runtime failures do not become NEQ or reference exclusions.

`tmqmg_regression.py` preserves its raw `equivalence_decision` and no longer lets
manual review or formula diagnostics overwrite `equivalent`, `strict_equivalent`,
or `effective_equivalent`. The last field is compatibility data, not the formal
score. Official `equivalent_fraction` is EQ divided by the valid-reference count.
Freeze the same reference-status membership and graph inputs before comparing
the original evaluator, merged v1, and v2; do not derive eligibility from their
decisions. Review records and existing artifacts are not rewritten.

An explicit frozen reference-status membership is reused rather than reopened
by an unconfirmed legacy whitelist entry. Independent reference/XYZ composition
contradictions are still recorded; evaluator outcomes never select membership.

## Frozen three-version replay

`scripts/evaluator_replay.py` consumes a fixed reference-validity manifest,
candidate byte inventory, and three standalone evaluator snapshots. Frozen
membership uses `valid`, `confirmed_excluded`, and `unresolved`; existing curated
adjudications remain evidence without requiring newly introduced review fields.
Original published references are not replaced with accepted candidate graphs.

Every selected case gets raw per-version records before validity aggregation.
INC, evaluator errors, timeouts, input errors, and unavailable candidates remain
distinct. Empty reference SMILES are input errors, not empty-molecule NEQ.
Lossless MolBlocks take precedence over V2000, with no SMILES fallback.
Calls follow benchmark order, reference then candidate, retaining conformers.

SQLite commits each `(case_id, version)` independently. Identical invocations
resume missing rows; changed inputs, source, runtime binaries, or parameters
require a new output directory. Use `--limit` for smoke only; the default exact
RDKit requirement is `2026.03.6`. The full run requires separate authorization.

The replay runner supports Linux and macOS environments with the multiprocessing
`fork` start method. Windows is not supported; replay integration tests are skipped
there. Summary/statistics imports do not require matplotlib; plotting loads the
optional dependency only when requested.
