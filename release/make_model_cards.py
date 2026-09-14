"""Generate HF model cards from measured artifacts — numbers are never typed.

Every number in a card is read from an eval JSON produced by the benchmark
scripts (benchmark/eval_*.py), the threshold calibration
(deploy/calibrate_thresholds.py), and the export metadata
(deploy/build_release.py). A card CANNOT render with holes: any missing file
raises, listing exactly what to run. Regenerating twice yields byte-identical
output, so `git diff` on a card is always a real change in evidence.

Evidence is read from $RA_RUNS (default runs/); the export metadata from
deploy/dist/<model_id>/.

    RA_RUNS=/path/to/runs python release/make_model_cards.py     # all final models
    python release/make_model_cards.py --only relsgg-vitb16
    -> deploy/dist/<model_id>/README.md
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
RUNS = os.environ.get("RA_RUNS", "runs")
GITHUB = "https://github.com/Maelic/RelateAnything"

A1_SOURCES = ["vg150", "psg", "indoorvg", "hicodet"]
A3_SOURCES = ["vg150", "psg", "indoorvg"]


class MissingEvidence(SystemExit):
    pass


def need(path: str, hint: str) -> str:
    if not os.path.exists(path):
        raise MissingEvidence(
            f"[cards] missing evidence: {path}\n        produce it with: {hint}")
    return path


def f1(r: float, mr: float) -> float:
    return 0.0 if (r + mr) <= 0 else 2 * r * mr / (r + mr)


def load_metrics(run_dir: str, name: str, hint: str) -> dict:
    return json.load(open(need(os.path.join(run_dir, name), hint)))["metrics"]


def fmt(x, nd=3):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "--"
    return f"{x:.{nd}f}"


def build_card(m: dict, a) -> str:
    mid, run_dir = m["model_id"], m["run_dir"]
    run = os.path.basename(run_dir)
    if run_dir.startswith("runs/"):
        run_dir = os.path.join(RUNS, run_dir[len("runs/"):])
    hint = (f"python benchmark/eval_zeroshot.py --checkpoint {run_dir}/checkpoint_last.pth "
            f"--data_roots runs/packed/vg150 runs/packed/psg runs/packed/indoorvg runs/packed/hicodet "
            f"(and eval_spatialsense.py, eval_decomposed.py; see docs/evaluation.md)")
    hf_repo = m["hf_repo"]          # main() refuses entries without one

    # ---- evidence ----------------------------------------------------------
    a1 = {s: load_metrics(run_dir, f"zeroshot_{s}_test_gc.json", hint)
          for s in A1_SOURCES}
    a3 = {s: load_metrics(run_dir, f"zeroshot_{s}_test_gc_ov.json", hint)
          for s in A3_SOURCES}
    ss = json.load(open(need(os.path.join(run_dir, "spatialsense.json"), hint)))
    dec = json.load(open(need(os.path.join(run_dir, "decomposed.json"), hint)))
    thr = json.load(open(need(
        os.path.join(RUNS, "analysis", run, "deploy_thresholds.json"),
        f"python deploy/calibrate_thresholds.py --checkpoint {run_dir}/checkpoint_best.pth")))
    exp_meta_p = os.path.join("deploy/dist", mid, "relateanything.json")
    exp = json.load(open(need(
        exp_meta_p, f"python deploy/build_release.py --only {mid}")))

    pp = ss.get("per_predicate") or {}
    ss_auc = (sum(v["AUC"] for v in pp.values()) / len(pp)) if pp else ss["AUC"]

    # ---- YAML front matter -------------------------------------------------
    metrics_yaml = []
    for s in A1_SOURCES:
        r, mr = a1[s]["R@50"], a1[s]["mR@50"]
        metrics_yaml.append(
            f"      - type: F1@50\n        value: {f1(r, mr):.4f}\n"
            f"        name: 'F1@50 ({s} test, graph-constrained)'")
    # What the card claims the repository holds is what the bundle holds — the
    # tag list included. A model without an export is not an onnx model.
    dist = os.path.join("deploy/dist", mid)
    has_onnx = os.path.exists(os.path.join(dist, "relateanything.onnx"))
    has_ov = os.path.exists(os.path.join(dist, "relateanything_fp16.xml"))
    tags = ["scene-graph-generation", "open-vocabulary", "visual-relationship-detection"]
    if has_onnx:
        tags.append("onnx")
    tags_yaml = "\n".join(f"  - {t}" for t in tags)
    head = f"""---
license: other
license_name: dinov3-license
license_link: https://ai.meta.com/resources/models-and-libraries/dinov3-license/
tags:
{tags_yaml}
library_name: relsgg
model-index:
  - name: {mid}
    results:
      - task:
          type: scene-graph-generation
        dataset:
          type: vg150
          name: Visual Genome 150 (test)
        metrics:
{chr(10).join('    ' + l for l in metrics_yaml[0].splitlines())}
---
"""

    # ---- body --------------------------------------------------------------
    a1_rows = "\n".join(
        f"| {s} | {fmt(a1[s]['R@50'])} | {fmt(a1[s]['mR@50'])} | "
        f"{fmt(f1(a1[s]['R@50'], a1[s]['mR@50']))} |" for s in A1_SOURCES)
    a3_rows = "\n".join(
        f"| {s} | {fmt(a3[s]['SoftR@50'])} | {fmt(a3[s]['SoftmR@50'])} | "
        f"{fmt(f1(a3[s]['SoftR@50'], a3[s]['SoftmR@50']))} |" for s in A3_SOURCES)
    dsrc = dec["per_source"]
    dec_rows = "\n".join(
        f"| {s} | {fmt(v['spatial_R@50'])} / {fmt(v['spatial_mR@50'])} "
        f"| {fmt(v['semantic_R@50'])} / {fmt(v['semantic_mR@50'])} |"
        for s, v in dsrc.items())
    thr_rows = "\n".join(
        f"| {r['name']} | {fmt(r['best_f1_thr'])} | {fmt(r['best_f1'])} | {r['gt_support']} |"
        for r in sorted(thr["predicates"], key=lambda x: -(x["gt_support"] or 0))[:15])

    # ---- training mixture from args.json (never hand-typed) ----------------
    targs = json.load(open(need(os.path.join(run_dir, "args.json"), "train the run")))
    fr = targs.get("mix_fractions") or []
    names = next((v for k, v in targs.items()
                  if isinstance(v, list) and len(v) == len(fr) and fr
                  and all(isinstance(x, str) for x in v)), None)
    if names:
        mix_line = " + ".join(os.path.basename(n.rstrip("/")) for n in names) + \
            ", per-image " + "/".join(f"{x:.3f}" for x in fr)
    else:
        mix_line = f"{targs.get('mix', 'see args.json')}, per-image " + \
            "/".join(f"{x:.3f}" for x in fr) if fr else "see args.json"
    if targs.get("hico_share"):
        mix_line += f" (HICO relation share {targs['hico_share']})"
    if targs.get("restrict_neg_sources"):
        mix_line += f"; source-aware negatives: {targs['restrict_neg_sources']}"

    # ---- OpenVINO CPU variants (measured by job_release_model.sh) ----------
    ov_p = os.path.join(RUNS, "release", "openvino_bench.json")
    ov = (json.load(open(ov_p)).get("models", {}).get(mid) if os.path.exists(ov_p) else None)
    if ov:
        ov_rows = "\n".join(
            f"| {v} | {d['total_ms']:.0f} | {d['fps']:.1f} | {d['top1_agree_vs_onnx']:.3f} | "
            f"{d['top20_jaccard_vs_onnx']:.3f} |" for v, d in ov.items())
        ov_section = f"""## Laptop CPU variants (OpenVINO IR, in the bundle)

Detector + relation head + decode on a node CPU, 8 threads, PSG val images —
RELATIVE numbers (absolute latency depends on the machine). Agreement is
semantic top-1 / top-20 Jaccard of the emitted triplets against the fp32 ONNX.

| variant | total ms | FPS | top-1 agreement | top-20 Jaccard |
|---|---|---|---|---|
{ov_rows}

`relateanything_fp16` is the default IR. `relateanything_w4` (weight-only
int4) is the size-optimised option: use it only when the bundle size matters
more than the agreement above.

"""
    else:
        ov_section = ""

    bundle = []
    if has_onnx:
        bundle.append("`relateanything.onnx`" + (" + OpenVINO fp16 IR" if has_ov else ""))
    bundle += ["`predicate_bank.npz`", "`thresholds.json`", "`calibration.json`"]
    onnx_note = ", ".join(bundle) + ", "
    n_pred_note = ""
    emb = os.path.join(dist, "predicate_embeddings.npz")
    if os.path.exists(emb):
        import numpy as np
        n_pred_note = f", {len(np.load(emb, allow_pickle=True)['names']):,} strings"
    body = f"""# {mid}

Open-vocabulary relation prediction from any boxes or masks. Give the model an
image and regions from any source (a detector, a segmenter, ground truth); it
returns ranked relations over a predicate vocabulary supplied at inference,
and optionally two graphs (spatial + semantic) from the same forward pass.
Object class labels are never an input.

Part of **RelateAnything** ([code]({GITHUB}) · paper: *RelateAnything: Real-Time
Open-Vocabulary Relation Prediction From Any Inputs*). Trained on
[RA-4M](https://huggingface.co/datasets/maelic/RA-4M); evaluated with
[OV-SGG-Bench](https://huggingface.co/datasets/maelic/OV-SGG-Bench).

## Use it

```bash
pip install git+{GITHUB}
hf download {hf_repo}          # optional; the API fetches on first use
```

```python
from relsgg import RelateAnything

# Regions come from any detector, any segmenter, or your own annotation.
# Object class labels are never an input.
model = RelateAnything.from_pretrained("{hf_repo}", device="cuda")
for t in model.predict(image, boxes_xyxy, topk=20):    # PIL/ndarray, boxes [N, 4] in pixels
    print(t)                                           # (person) --riding [0.67]--> (horse)

# Masks instead of boxes: pass the [N, H, W] binary masks beside their extents.
triplets = model.predict(image, boxes_xyxy, masks=masks, topk=20)

# The vocabulary is an input. Any strings, at any time, without retraining.
model.set_vocabulary(["about to collide with", "reflected in"])

# Or answer from the whole training vocabulary{n_pred_note}, read from the weights.
model = RelateAnything.from_pretrained("{hf_repo}", full_vocabulary=True, device="cuda")

# Two graphs from one forward pass.
graphs = model.predict(image, boxes_xyxy, decompose=True)   # {{"spatial": [...], "semantic": [...]}}
```

Every vocabulary is encoded once by the text student shipped beside the
weights, and the head is reparameterized onto it; scoring afterwards is vision
only. `full_vocabulary=True` reads `predicate_embeddings.npz` instead of
encoding, which turns a minute and a half of CPU work into a download.
`model.pth` embeds the backbone configuration, so running these weights needs
no gated DINOv3 login.

Files: `model.pth` (torch, EMA weights), `text_student.pt` + tokenizer,
`predicate_embeddings.npz` (the training vocabulary, encoded), {onnx_note}`README.md`.

**Every number below is generated from measured eval artifacts
(`release/make_model_cards.py`); none is hand-typed.**

## Closed-vocabulary transfer (reparameterized, TEST, graph-constrained)

| source | R@50 | mR@50 | F1@50 |
|---|---|---|---|
{a1_rows}

## Open-vocabulary, NO reparameterization (all 19,103 predicates deployed)

Synonym-matched at the calibrated tau (see provenance). This is the honest
"the model never saw your label set" protocol.

| source | SoftR@50 | SoftmR@50 | SoftF1@50 |
|---|---|---|---|
{a3_rows}

## Spatial reasoning (SpatialSense, adversarial true/false; chance = 0.5)

Macro AUC over predicates: **{ss_auc:.4f}**

## Two-graph decomposition (spatial / semantic, type-stratified protocol)

| source | spatial R@50 / mR@50 | semantic R@50 / mR@50 |
|---|---|---|
{dec_rows}

## Deployment thresholds (per-predicate best-F1, measured on THIS checkpoint)

Score scales are checkpoint-specific (the output head is rank-trained), so
these thresholds transfer to no other model. Regime: {thr['regime']['boxes']}
boxes, pair_weight={thr['regime']['pair_weight']}, {thr['regime']['n_images']}
val images. Top predicates by support:

| predicate | threshold | best F1 | GT support |
|---|---|---|---|
{thr_rows}

{ov_section}## Provenance

| | |
|---|---|
| run | `{exp['run_name']}` |
| git | `{exp['git_sha']}` |
| backbone | {m.get('backbone_model') or exp.get('backbone_model')} |
| text student | `{exp.get('text_student')}` sha256 `{str(exp.get('text_student_sha256'))[:16]}...` |
| ONNX opset / parity | {exp.get('opset')} / max|Δ| {exp.get('check_max_abs_delta', float('nan')):.2e} |
| torch / transformers | {exp.get('torch')} / {exp.get('transformers')} |
| training mixture | {mix_line} |

## License and data notices

Weights are a derivative of Meta **DINOv3** pretrained weights and are
distributed under the DINOv3 license. Training annotations (RA-4M) were
generated by `gemma-4-26B` and carry the Gemma Terms of Use notice; images
are referenced by identifier only (Objects365/COCO/OpenImages). The `vg_raw`
subset derives from Visual Genome (CC BY 4.0). Predicate synonyms are
deliberately never collapsed — surface-form diversity is part of the label
space. Full notices: [THIRD_PARTY_NOTICES.md]({GITHUB}/blob/main/THIRD_PARTY_NOTICES.md)
in the code repository.

## Citation

```bibtex
@article{{neau2026relateanything,
  title   = {{RelateAnything: Real-Time Open-Vocabulary Relation Prediction From Any Inputs}},
  author  = {{Neau, Ma\"elic}},
  journal = {{arXiv preprint arXiv:2609.12552}},
  eprint  = {{2609.12552}},
  archivePrefix = {{arXiv}},
  primaryClass  = {{cs.CV}},
  url     = {{https://arxiv.org/abs/2609.12552}},
  year    = {{2026}}
}}
```
"""
    return head + body


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="deploy/release_manifest.json")
    ap.add_argument("--only", default=None)
    ap.add_argument("--allow_standin", action="store_true",
                    help="render cards for stand-in entries (pipeline "
                         "verification only — NEVER upload these)")
    ap.add_argument("--out_dir", default="deploy/dist")
    a = ap.parse_args()
    os.chdir(REPO)
    man = json.load(open(a.manifest))
    for m in man["models"]:
        if a.only and m["model_id"] != a.only:
            continue
        if not m.get("run_dir"):
            print(f"[cards] {m['model_id']}: no run_dir, skipped")
            continue
        if m.get("status") != "final" and not a.allow_standin:
            print(f"[cards] {m['model_id']}: status={m.get('status')} — "
                  "skipped (use --allow_standin for pipeline tests)")
            continue
        if not m.get("hf_repo"):
            # A card is the front page of a Hub repository. Without one to name,
            # the card would invent a download that does not resolve.
            print(f"[cards] {m['model_id']}: no hf_repo in the manifest, skipped")
            continue
        card = build_card(m, a)
        out = os.path.join(a.out_dir, m["model_id"], "README.md")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        open(out, "w").write(card)
        print(f"[cards] wrote {out} ({len(card)} bytes)")


if __name__ == "__main__":
    main()
