# Published tmQMg regression selection

The historical 706-case regression selection is now stored as IDs only in
[`../../tmqmg/published/regression_case_ids.json`](../../tmqmg/published/regression_case_ids.json).
Inputs and native reference graphs are loaded exclusively through
`scripts.tmqmg_fixture_data` from the shared 941-case published fixture store.
There is no database snapshot or duplicate graph payload in this directory.

The selection retains 670 ordinary reference cases and 36 eligible coupled-spin
cases. Coupled-spin evidence and native properties remain checked; only the
closed-shell reconstruction repair regression skips those 36 cases.

See [the tmQMg fixture README](../../tmqmg/README.md) for the pinned release,
comparison policy, regeneration command and CC BY 4.0 attribution.
