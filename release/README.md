# Release tooling

Everything here produces or publishes a release artifact. Numbers in cards
are generated from measured evaluation files, never typed.

| script | does |
|---|---|
| `strip_checkpoint.py` | training checkpoint -> `model.pth` (EMA weights, scrubbed args, embedded backbone config) + `text_student.pt` + `predicate_embeddings.npz` (the training vocabulary encoded with that student); verifies a strict offline load |
| `make_model_cards.py` | writes `deploy/dist/<model_id>/README.md` from the eval JSONs under `$RA_RUNS` |
| `hf_upload.py` | one Hugging Face model repository per `deploy/release_manifest.json` entry |
| `upload_datasets.py` | the `RA-4M` and `OV-SGG-Bench` dataset repositories |
| `ra4m/` | the RA-4M image manifest and the extractor that rebuilds the image set from `JosephZ/mega_1m` |
| `cards/` | dataset cards; model cards live next to each bundle under `deploy/dist/` |

## Publishing a release, in order

Run from the training checkout, where `$RA_RUNS/train/<run>/` holds the
checkpoints and evaluation outputs (`RA_RUNS` defaults to `runs/`).

```bash
# 0. export the ONNX / OpenVINO bundles (released models only)
python deploy/build_release.py                       # reads deploy/release_manifest.json

# 1. strip the three released checkpoints (needs the backbone config: HF login,
#    or --backbone_config). The -zeroshot arms in the manifest are status
#    "unreleased": no weights are published for them and every step here skips them.
for m in relsgg-vits16 relsgg-vits16plus relsgg-vitb16; do
  run=$(python -c "import json;print([e['run_dir'] for e in json.load(open('deploy/release_manifest.json'))['models'] if e['model_id']=='$m'][0])")
  python release/strip_checkpoint.py --checkpoint "$run/checkpoint_last.pth" --out "deploy/dist/$m/model.pth"
done

# 2. cards
python release/make_model_cards.py

# 3. models, then datasets (add --private to create private repositories first)
python release/hf_upload.py --org maelic --dry-run
python release/hf_upload.py --org maelic
python release/upload_datasets.py --org maelic --ra4m --bench --dry-run
python release/upload_datasets.py --org maelic --ra4m --bench
```

Detector weights (`*detector*`, any `.pt` other than the text student) are
refused by `hf_upload.py`: they derive from ultralytics (AGPL-3.0) and are
rebuilt locally by every user (`deploy/README.md`).
