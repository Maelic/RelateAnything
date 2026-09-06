# Training configurations of the released models

One JSON per released model: the exact argument set (`args.json`) of the run
that produced it, with machine-local paths removed and the backbone written
as its Hugging Face id. `train.py` takes flags, not a config file; `train.sh`
is the same recipe as a command, and these files are the record of what each
published checkpoint was trained with.

| config | mixture (per-image fractions) | epochs |
|---|---|---|
| `relsgg-vits16.json`, `relsgg-vits16plus.json`, `relsgg-vitb16.json` | `megasg_clean` 0.727 + `vg_raw` 0.063 + `hicodet` 0.210, source-aware negatives on HICO-DET | 12 |
| `relsgg-*-zeroshot.json` | `megasg_clean` 0.919 + `vg_raw` 0.081 (no HICO-DET; the paper's zero-shot rows) | 12 |

Keys starting with `_` are provenance added at export time (`_model_id`,
`_run_name`, `_hf_repo`). To reproduce a run, start from `train.sh` and
change `BACKBONE`; every other flag there matches these files.
