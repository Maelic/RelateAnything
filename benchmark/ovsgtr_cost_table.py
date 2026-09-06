"""ovsgtr_cost_table.py -- collapse the per-GPU OvSGTR cost JSONs into one table.

Reads the runs/benchmark/cost/ovsgtr_*.json files written by benchmark/ovsgtr/bench_ovsgtr_cost.py and prints markdown.
Latency is per-GPU; FLOPs and parameters are not (they are a property of the model and
the input), so they are printed once per checkpoint rather than repeated per row.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

GPU_ORDER = ["a40", "a100", "h100"]
PREC_ORDER = ["strict", "tf32"]


def load(d: Path):
    rows = []
    for f in sorted(d.glob("ovsgtr_*.json")):
        r = json.loads(f.read_text())
        stem = f.stem[len("ovsgtr_"):]
        for p in PREC_ORDER:
            if stem.endswith("_" + p):
                r["_prec"], stem = p, stem[: -len(p) - 1]
                break
        else:
            r["_prec"] = "?"
        r["_gpu"] = stem.rsplit("_", 1)[-1]
        r["_ck"] = stem.rsplit("_", 1)[0]
        rows.append(r)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", default="runs/benchmark/cost")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    rows = load(Path(a.dir))
    if not rows:
        raise SystemExit(f"no ovsgtr_*.json under {a.dir}")

    key = lambda r: (r["_ck"], GPU_ORDER.index(r["_gpu"]) if r["_gpu"] in GPU_ORDER
                     else 99, PREC_ORDER.index(r["_prec"]) if r["_prec"] in PREC_ORDER else 99)
    rows.sort(key=key)

    L = []
    L.append("## OvSGTR batch-1 latency, VG150 test, end-to-end (detection + relation)\n")
    L.append("| checkpoint | GPU | precision | transformer ms | graph_infer ms | "
             "total ms | p95 | FPS | boxes/img | pairs/img |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        L.append("| {ck} | {gpu} | {prec} | {t:.1f} | {g:.1f} | **{tot:.1f}** | {p95:.1f} "
                 "| {fps:.1f} | {bx:.1f} | {pr:.0f} |".format(
                     ck=r["_ck"].replace("vg-ovdr-", "").replace("-mega-best", ""),
                     gpu=r["_gpu"].upper(), prec=r["_prec"],
                     t=r["transformer_ms"]["mean"], g=r["postprocess_graph_infer_ms"]["mean"],
                     tot=r["total_ms"]["mean"], p95=r["total_ms"]["p95"], fps=r["fps"],
                     bx=r["boxes_per_img"], pr=r["pairs_per_img"]))

    L.append("\n## Parameters and FLOPs (GPU-independent)\n")
    L.append("| checkpoint | params M | transformer GFLOPs | graph_infer GFLOPs | total GFLOPs |")
    L.append("|---|---|---|---|---|")
    seen = set()
    for r in rows:
        if r["_ck"] in seen or "total_GFLOPs" not in r:
            continue
        seen.add(r["_ck"])
        L.append("| {ck} | {p:.1f} | {t:.1f} | {g:.1f} | **{tot:.1f}** |".format(
            ck=r["_ck"].replace("vg-ovdr-", "").replace("-mega-best", ""),
            p=r["params_M"], t=r["transformer_GFLOPs"]["mean"],
            g=r["postprocess_graph_infer_GFLOPs"]["mean"], tot=r["total_GFLOPs"]["mean"]))

    hosts = {(r["_gpu"], r["gpu"], r["host"]["cpu"]) for r in rows}
    L.append("\n## Hardware\n")
    for g, name, cpu in sorted(hosts):
        L.append(f"- **{g.upper()}** — {name}, host CPU {cpu}")
    r0 = rows[0]
    L.append(f"\nProtocol: batch 1, {r0['n_warmup']} warmup + {r0['n_images']} timed images, "
             f"CUDA-event timed, native OvSGTR test resize (short side 800, long side "
             f"<=1333), torch {r0['host']['torch']} / CUDA {r0['host']['cuda']}. "
             f"fp32 only: their ms_deform_attn kernel has no fp16/bf16 dispatch.")

    text = "\n".join(L)
    print(text)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(text + "\n")
        print("\nwrote", a.out)


if __name__ == "__main__":
    main()
