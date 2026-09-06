"""ra_deploy_table.py -- deployment-ready end-to-end results, as one report.

Reads runs/benchmark/cost/ra_deploy_<gpu>.json (the arm sweep) and
runs/benchmark/cost/checks/overlap_<gpu>_*.json (the isolation diagnostic that says
whether the concurrent backbone could have helped at all).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

GPU = ["a40", "a100", "h100"]
MOD = ["ViT-S/16", "ViT-S/16+", "ViT-B/16"]
ARM = ["torch-eager-seq", "torch-eager-concurrent", "torch-compile-seq",
       "torch-compile-concurrent", "torch-cudagraph-concurrent", "onnx-cuda-seq"]
PRETTY = {"torch-eager-seq": "eager, sequential",
          "torch-eager-concurrent": "eager, **concurrent backbone**",
          "torch-compile-seq": "torch.compile, sequential",
          "torch-compile-concurrent": "torch.compile, concurrent",
          "torch-cudagraph-concurrent": "CUDA graphs, concurrent",
          "onnx-cuda-seq": "ONNX Runtime CUDA EP"}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", default="runs/benchmark/cost")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    d = Path(a.dir)
    rows = []
    meta = {}
    for f in d.glob("ra_deploy_*.json"):
        j = json.loads(f.read_text())
        g = f.stem.split("ra_deploy_")[1]
        meta[g] = j
        for r in j["rows"]:
            if "error" in r:
                continue
            r["_gpu"] = g
            rows.append(r)
    L = []
    L.append("## Deployment end-to-end — batch 1, VG150 test, real 243-predicate "
             "release vocabulary\n")
    L.append("| model | configuration | GPU | p50 ms | p95 | max | FPS | det | rel | stalls |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    rows.sort(key=lambda r: (MOD.index(r["model"]), ARM.index(r["arm"]),
                             GPU.index(r["_gpu"])))
    for r in rows:
        t = r["total"]
        L.append(f"| {r['model']} | {PRETTY.get(r['arm'], r['arm'])} | "
                 f"{r['_gpu'].upper()} | **{t['p50']:.1f}** | {t['p95']:.1f} | "
                 f"{t['max']:.1f} | {t['fps_p50']:.1f} | {r.get('det_ms', 0):.1f} | "
                 f"{r.get('relation_ms', 0):.1f} | {t['stalls_gt_3x_median']} |")

    # best arm per (model, gpu)
    L.append("\n### Fastest configuration per model and GPU\n")
    L.append("| model | GPU | best configuration | p50 ms | FPS | vs eager sequential |")
    L.append("|---|---|---|---|---|---|")
    for m in MOD:
        for g in GPU:
            cand = [r for r in rows if r["model"] == m and r["_gpu"] == g]
            if not cand:
                continue
            base = next((r for r in cand if r["arm"] == "torch-eager-seq"), None)
            best = min(cand, key=lambda r: r["total"]["p50"])
            sp = (base["total"]["p50"] / best["total"]["p50"]) if base else float("nan")
            L.append(f"| {m} | {g.upper()} | {PRETTY.get(best['arm'], best['arm'])} | "
                     f"{best['total']['p50']:.1f} | {best['total']['fps_p50']:.1f} | "
                     f"{sp:.2f}x |")

    ov = []
    for f in glob.glob(os.path.join(a.dir, "checks", "overlap_*.json")):
        j = json.loads(Path(f).read_text())
        j["_tag"] = os.path.basename(f)[len("overlap_"):-len(".json")]
        ov.append(j)
    if ov:
        L.append("\n## Concurrent backbone — isolation diagnostic\n")
        L.append("The relation backbone reads only the image, so it is independent of the "
                 "detector and can in principle run beside it. Timing each branch alone "
                 "gives the ceiling; the realized column says how much of it a worker "
                 "thread actually collects.\n")
        L.append("| GPU / detector | det alone | backbone alone | sequential | "
                 "concurrent | perfect overlap | realized | % of theoretical |")
        L.append("|---|---|---|---|---|---|---|---|")
        for j in sorted(ov, key=lambda j: j["_tag"]):
            L.append(f"| {j['_tag']} | {j['det_only_ms']:.1f} | "
                     f"{j['backbone_only_ms']:.1f} | {j['sequential_ms']:.1f} | "
                     f"{j['concurrent_ms']:.1f} | {j['perfect_overlap_ms']:.1f} | "
                     f"**{j['realized_saving_ms']:+.1f} ms** | "
                     f"{100 * j['fraction_of_theoretical']:.0f}% |")

    if meta:
        m0 = list(meta.values())[0]
        L.append(f"\nProtocol: batch 1, {m0['n_warmup']} warm-up + {m0['n_images']} real "
                 f"frames, detector `{os.path.basename(m0['det'])}` at imgsz "
                 f"{m0['imgsz']} conf {m0['conf']}, max_objects {m0['max_objects']}, "
                 f"decode included, torch {m0['torch']} / CUDA {m0['cuda']}. "
                 f"Vocabulary is each bundle's `predicate_bank.npz` — 243 predicates, "
                 f"really encoded, no stand-in. torch.compile arms are DEPLOYMENT "
                 f"numbers only: inductor changes reduction order and moves eval "
                 f"metrics by ~40x the noise floor.")
    text = "\n".join(L)
    print(text)
    if a.out:
        Path(a.out).write_text(text + "\n")
        print("\nwrote", a.out)


if __name__ == "__main__":
    main()
