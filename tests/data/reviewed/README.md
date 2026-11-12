# Reviewed molecule graph fixtures

This directory contains authoritative molecule graphs selected through manual
review and used as regression-test references.

Each dataset has an independent `manifest.json` and may contain three fixture
types:

- `approved_graph`: an accepted MolGR reconstruction stored as SDF;
- `reference_graph`: an accepted dataset SMILES with its XYZ and electronic state;
- `manual_reference`: a corrected SMILES with the original XYZ and electronic state.

Pending and skipped decisions remain in the local review database and are not
test references. Reviewer identity, notes, timestamps, queues, and database
exports are also excluded from this directory.

The tmQMg corpus is stored under `tmqmg/`. Its manifest records the pinned
dataset revision and source-file checksums required to reproduce the inputs.

The `reconstruction/` corpus contains ten user-approved graphs from the
repository XYZ review. SDF `APPROVED_STEREO` records freeze the accepted reference
frames (atom IDs are one-based), independently of subsequent coordinate perception.
`APPROVED_SMILES` is included only when a strict stereo round trip was verified.
The `test` and `ABEGOD` cases use their SDF graphs and frozen stereo frames because
the reviewed RDKit version does not preserve their stereo through SMILES export.
