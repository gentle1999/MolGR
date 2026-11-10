# Native evaluator performance (2026-11-10)

## Measurement scope

Measurements replay 4,929 parseable pairs from the saved first 5,000-case tmQMg
chunk, sequentially on an Intel Core i7-14700K under Linux/WSL, Python 3.10 and
RDKit 2026.03.6, with a Release extension. They include the public evaluator's
input preparation and result packaging, excluding SMILES parsing and XYZ
reconstruction. This is not a new full-dataset reconstruction run. Missing source
electronic or stereo annotations cannot be recovered from the saved SMILES.

Input CSV SHA256:
`8ee708954f07cffd8f746ba61a8b933e41ef4d479d1441c5a4d2e3952c478de9`.

## Early PR comparison

The baseline is PR commit `0f7ff6b9275ba05f2383245335fda7f191017be6`.
The native measurement includes the stereo fixes, with the intermediate
`max_resonance=250`, `max_resonance_steps=24` settings; the baseline uses its
50-form limit. Runs were ordered baseline/native/native/baseline in independent
processes, each after 100 warmup pairs (legacy caches cleared afterward).
Numbers are means of two runs; maxima are the largest per-case mean.

| Metric | Early PR | Native, budget 250 |
| --- | ---: | ---: |
| Total, stereo off | 83.428 s | 3.859 s |
| P50 | 10.800 ms | 0.762 ms |
| P99 | 151.252 ms | 1.597 ms |
| Maximum per-case mean | 584.929 ms | 3.027 ms |
| Cases averaging >10 ms | 2,821 | 0 |
| Total, stereo on | 81.921 s | 4.994 s |
| P99, stereo on | 125.030 ms | 1.932 ms |

This is 21.6x faster with stereo off and 16.4x with stereo on. For the 4,346
cases both versions accept with stereo off, the speedup is 20.6x. Native kernel
stages account for about 0.597 s of the 3.859 s; Python preprocessing, transfer
and result packaging account for the remainder. No external parallelism was
used to obtain these improvements.

Verdicts are intentionally not identical: early PR EQ/NEQ/INC =
4,356/465/108; native budget 250 = 4,448/460/21. InChIKey-only agreement is no
longer structural proof, and the native engine supports additional witnesses.
All reference SMILES in this sample lack stereo tags; stereo-on timings test
missing-label handling and do not establish complete stereochemical accuracy.

## Final budget: 1,000

A separate interleaved 500/1000/1000/500 comparison uses the same native binary,
24-step limit, stereo off and ignored metal coordination:

| Metric | Budget 500 | Budget 1,000 |
| --- | ---: | ---: |
| Mean total | 3.946 s | 3.851 s |
| Mean P99 | 1.587 ms | 1.498 ms |
| EQ / NEQ / INC | 4,448 / 460 / 21 | 4,450 / 460 / 19 |
| Total visited states | 20,563 | 22,042 |

The small timing difference is consistent with run-to-run variation; increasing
the budget showed no appreciable cost on this sample. Only ASEKEQ and BADGOG
change INC to EQ. Their witnesses were independently replayed: 8 moves / 1,402
visited states and 9 moves / 1,077 states, respectively. No EQ or NEQ regresses.
The shared visited-state cap is `2 * max_resonance`, including both roots and
all mappings/regions; the final default therefore allows 2,000 states.

## Repeating comparisons

`scripts/benchmark_native_equivalence.py` compares the current native engine
against the archived Python engine on a supplied saved-pair CSV and emits
timings, verdict transitions and input/source/binary hashes:

```sh
python scripts/benchmark_native_equivalence.py results.csv \
  --limit 0 --repeat 2 --max-resonance 1000 --out comparison.json
```

That archived engine is a later optimized Python snapshot, not the early
`0f7ff6b` baseline; this command does not reproduce the first table verbatim.
Keep runtime, inputs and per-engine limits explicit when comparing revisions.
The tables above are local observations, not CI thresholds or universal speedup
guarantees. See [the evaluator contract](evaluator-contract.md) for supported
rules, conservative outcomes, defaults and API migration.
