# Training configurations

One JSON per training run of the model family: the arguments that produced it,
with machine-local paths removed and the backbone written as its Hugging Face
id. `train.py` takes flags rather than a config file, so these are the record of
what each checkpoint was trained with; `train.sh` is the same recipe as a
command.

The three `relsgg-vit*.json` runs are the published models. The `-zeroshot`
runs are the evaluation control the report's zero-shot rows come from — the same
recipe without HICO-DET, so a HICO-DET number measured on that tower is
cross-dataset. No weights are published for them; the config is here so the rows
can be reproduced.

| config | mixture (per-image fractions) | epochs |
|---|---|---|
| `relsgg-vits16.json`, `relsgg-vits16plus.json`, `relsgg-vitb16.json` | `megasg_clean` 0.727 + `vg_raw` 0.063 + `hicodet` 0.210, source-aware negatives on HICO-DET | 12 |
| `relsgg-*-zeroshot.json` (not published) | `megasg_clean` 0.919 + `vg_raw` 0.081 (no HICO-DET; the report's zero-shot rows) | 12 |

Keys starting with `_` are provenance (`_model_id`, `_hf_repo` on a published
model, `_released: false` on a control). Everything else is a `train.py` flag,
and the values that are not shown here are the defaults in [`relsgg/config.py`](../../relsgg/config.py) — which are the
released recipe. To reproduce a run, start from `train.sh` and change
`BACKBONE`.
