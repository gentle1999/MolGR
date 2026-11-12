# Reconstruction fixtures

`xyz_stress_manifest.csv` lists every XYZ input used by concurrency and explicit-H
SMILES round-trip tests. Passing those tests establishes execution and serialization
stability; it does not establish chemical correctness.

The reference columns index the existing correctness fixtures:

- `reference_file`: a reference graph path relative to this directory, when available.
- `reference_source`: a repository-relative source path, followed by the case ID,
  Python constant, or SDF property after `#`.
- `reference_status`: `reviewed` for an approved replacement, `dataset` for a
  published dataset answer, `test_reference` for an existing regression answer,
  `representation_issue` for a documented representation disagreement,
  `source_issue` for an invalid original answer without a complete replacement,
  `missing` when no complete answer has been recorded, or `rejected` when manual
  review rejected the candidate and a corrected replacement still awaits review.

`source_issue`, `missing`, and `rejected` inputs remain useful stress cases, but must not be
counted as reference-validated reconstructions. Review new reconstruction results
before promoting them to reference answers.

338-S uses `xyz/338-S.xyz` and `sdf/338-S.sdf`. The latter contains the corrected
Fe(II)–Pd(II)–Fe(II) graph and its verified `SOURCE_SMILES`: three Cp⁻ rings,
one Cp²⁻ ring, and one O⁻. Its SDF regression checks ligand charges,
coordination, and stereochemistry for both backends. Documentation reads this
same fixture; there is no separate documentation copy of the answer.
The SMILES starts at Pd to avoid version-dependent interpretation of a
square-planar center reached through a ring closure. Independently checked XYZ
trans pairs N28/O30 and C19/O21 are asserted alongside rejection of the other
square-planar isomers; their angles are approximately 169.74 and 174.61 degrees.

The ten accepted cases from the reconstruction review are frozen under
`reviewed/reconstruction/`, with their original XYZ checksums and electron-state
annotations. Their regression tests compare coordination and stereochemistry on
both backends. The aluminum case uses the corrected, subsequently approved graph with no
radical oxygen–Al contact.
