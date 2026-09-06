"""ra_cost_table.py -- collapse the per-GPU RelateAnything cost JSONs into one report.

Three tables, matching the three questions the benchmark answers:
  1  relation head alone   -- what the model costs, independent of any detector
  2  params + FLOPs        -- GPU-independent, closed and open vocabulary
  3  detector + head       -- the deployed pipeline, per detector

Reads runs/benchmark/cost/ra_{latency,flops,pipeline}_<gpu>[_<vocab>].json.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

GPU_ORDER = ["a40", "a100", "h100"]
MODEL_ORDER = ["ViT-S/16", "ViT-S/16+", "ViT-B/16"]
RUN_TO_MODEL = {"_vits16_": "ViT-S/16", "_vits16plus_": "ViT-S/16+"}
STAGES = ["backbone", "spatial_pool", "sampler", "rel_transformer", "vocab_head"]


def model_of(run: str) -> str:
    for tok, name in RUN_TO_MODEL.items():
        if tok in run:
            return name
    return "ViT-B/16"          # the B run carries no size token -- it is the default


def gkey(g):
    return GPU_ORDER.index(g) if g in GPU_ORDER else 99


def mkey(m):
    return MODEL_ORDER.index(m) if m in MODEL_ORDER else 99


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", default="runs/benchmark/cost")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    d = Path(a.dir)
    L = []
    synthetic = False

    # ---- 1. relation head alone --------------------------------------------
    lat = {}
    for f in d.glob("ra_latency_*.json"):
        j = json.loads(f.read_text())
        synthetic |= bool(j.get("synthetic_vocab"))
        lat[f.stem.split("ra_latency_")[1]] = j
    if lat:
        L.append("## 1. Relation head alone — batch 1, VG150 test, bf16 autocast\n")
        L.append("| model | GPU | closed V=50 | open V=19,103 | open−closed | "
                 "img/s @bs32 | backbone | spatial_pool | sampler | rel_transformer | vocab_head |")
        L.append("|---|---|---|---|---|---|---|---|---|---|---|")
        rows = [(model_of(r), g, v) for g, j in lat.items() for r, v in j["runs"].items()]
        for m, g, v in sorted(rows, key=lambda t: (mkey(t[0]), gkey(t[1]))):
            c, o = v["closed_eval_bs1"]["mean"], v["open_eval_bs1"]["mean"]
            st = v.get("open_stages_ms", {})
            L.append(f"| {m} | {g.upper()} | {c:.1f} ms | **{o:.1f} ms** | {o-c:+.1f} | "
                     f"{v[f'open_batch32_img_s']:.0f} | "
                     + " | ".join(f"{st.get(s, 0.0):.1f}" for s in STAGES) + " |")

    # ---- 2. params + FLOPs --------------------------------------------------
    fl = {}
    for f in d.glob("ra_flops_*.json"):
        j = json.loads(f.read_text())
        synthetic |= bool(j.get("synthetic_vocab"))
        gpu, vocab = f.stem.split("ra_flops_")[1].rsplit("_", 1)
        fl[(gpu, vocab)] = j
    if fl:
        any_gpu = sorted({g for g, _ in fl})[0]
        rel_c = fl.get((any_gpu, "closed"), {}).get("relation", {})
        rel_o = fl.get((any_gpu, "open"), {}).get("relation", {})
        L.append("\n## 2. Parameters and FLOPs — GPU-independent\n")
        L.append("### Relation head\n")
        L.append("| model | params M | GFLOPs closed V=50 | GFLOPs open V=19,103 | vocab head cost |")
        L.append("|---|---|---|---|---|")
        for m in sorted(rel_c, key=mkey):
            gc = rel_c[m].get("GFLOPs") or 0.0
            go = (rel_o.get(m) or {}).get("GFLOPs") or 0.0
            L.append(f"| {m} | {rel_c[m]['params_M']:.1f} | {gc:.1f} | {go:.1f} | {go-gc:+.1f} |")
        det = fl.get((any_gpu, "closed"), {}).get("detector", {})
        if det:
            L.append("\n### Detectors (at imgsz 640)\n")
            L.append("| detector | params M | GFLOPs |")
            L.append("|---|---|---|")
            for n, v in det.items():
                g = v.get("GFLOPs")
                L.append(f"| {n} | {v['params_M']:.1f} | "
                         + ("n/a" if g is None else f"{g:.1f}") + " |")

    # ---- 3. pipeline --------------------------------------------------------
    pipes = []
    for f in d.glob("ra_pipeline_*.json"):
        j = json.loads(f.read_text())
        synthetic |= bool(j.get("synthetic_vocab"))
        g = f.stem.split("ra_pipeline_")[1]
        for r in j["rows"]:
            r["_gpu"] = g
            pipes.append(r)
    if pipes:
        L.append("\n## 3. Detector + relation head — end-to-end, batch 1\n")
        L.append("| model | detector | classes | GPU | boxes/img | det ms | rel ms | "
                 "total ms | p95 | FPS |")
        L.append("|---|---|---|---|---|---|---|---|---|---|")
        order = {n: i for i, n in enumerate(dict.fromkeys(r["detector"] for r in pipes))}
        for r in sorted(pipes, key=lambda r: (mkey(r["model"]),
                                              order.get(r["detector"], 99),
                                              gkey(r["_gpu"]))):
            L.append(f"| {r['model']} | {r['detector']} | {r['det_classes']} | "
                     f"{r['_gpu'].upper()} | {r['boxes_per_img']:.1f} | {r['det_ms']:.1f} | "
                     f"{r['rel_ms']:.1f} | **{r['total_ms']:.1f}** | "
                     f"{r['total_ms_p95']:.1f} | {r['fps']:.1f} |")
        p0 = pipes[0]
        L.append(f"\nDetector settings: imgsz {p0.get('imgsz', 640)}, conf 0.10, "
                 f"max_det 100, max_boxes 60 into the head (REACT's protocol). "
                 f"`@N` in a detector name is the prompted class-head width.")

    if synthetic:
        L.append("\n> **Vocabulary stand-in.** The 512-d text student "
                 "(`runs/packed/text_student_v2_512/student.pt`) did not survive the "
                 "cluster migration, so the vocabulary matrix is a shape-correct "
                 "stand-in. Cost is a function of that matrix's SHAPE only, so every "
                 "latency and FLOP number here is exact; nothing in this report is an "
                 "accuracy claim.")

    text = "\n".join(L)
    print(text)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(text + "\n")
        print("\nwrote", a.out)


if __name__ == "__main__":
    main()
