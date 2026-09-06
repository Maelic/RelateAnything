#!/usr/bin/env python
"""Full statistical analysis of a sgg_vllm_generate.py run (any shard count).

Computes and plots:
  01  top-20 predicate distribution (stacked semantic/spatial)
  02  long tail: Zipf rank-frequency + cumulative predicate mass
  03  per-image relation counts + scaling vs number of objects
  04  scene-graph degree: node degree histogram, isolated objects, components
  05  spatial layer: category shares + direction balance
  06  noise: gate/verifier drop counters + residual geometry audit

Writes PNGs + stats.json to <run>/analysis/.

Usage:
  python datagen/analyze_sgg_run.py --run runs/vllm_generate/megasg_26b_F2_10k
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from sgg_canon import canonicalize_spatial                    # light import
from audit_contact_geometry import (CONTAINMENT, CONTACT, ATTENTION,
                                    contain_frac, iou)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from relsgg.paths import DATASETS, DATAMIX, RUNS, REPO  # noqa: E402

MEGASG = DATASETS / "MEGASG"

# spatial surface form -> category, via canonicalize_spatial (collapse on)
SPA_CAT = {"to the left of": "horizontal", "to the right of": "horizontal",
           "above": "vertical", "below": "vertical",
           "in front of": "depth", "behind": "depth",
           "near": "proximity", "inside": "containment"}
CONTACT_CANON = {"on", "on top of", "atop", "sitting on", "standing on",
                 "lying on", "leaning against", "leaning on", "hanging from",
                 "hanging on", "attached to", "touching", "against",
                 "resting on", "mounted on"}

C_SEM, C_SPA, C_ACC = "#4363d8", "#f58231", "#3cb44b"


# surface forms that canonicalize to "above" but are contact-verified in the
# pipeline (V2_NEEDS_CONTACT) — count them as contact, not vertical
SURFACE_CONTACT = {"on top of", "atop", "on top"}


def spa_category(pred: str) -> str:
    if pred.lower().strip() in SURFACE_CONTACT:
        return "contact"
    c = canonicalize_spatial(pred, collapse_spatial=True)
    if c in SPA_CAT:
        return SPA_CAT[c]
    if c in CONTACT_CANON:
        return "contact"
    return "other"


def is_spatial(rel: dict) -> bool:
    return bool(rel.get("spatial")) or rel.get("source") == "geometric"


def entropy(counter: Counter) -> float:
    tot = sum(counter.values())
    return -sum(c / tot * math.log(c / tot) for c in counter.values()) if tot else 0.0


def load_boxes(split: str, max_objects: int):
    """img_id -> {local_id: bbox}. Relation subject/object ids are LOCAL 1-based
    positions in the image's object list (annotation-file order, trimmed like
    the generator), NOT global COCO annotation ids."""
    with open(MEGASG / split / "_annotations.coco.json") as f:
        data = json.load(f)
    by_img: dict = defaultdict(dict)
    for a in data["annotations"]:
        d = by_img[a["image_id"]]
        if len(d) < max_objects:
            x, y, w, h = a["bbox"]
            d[len(d) + 1] = (x, y, x + w, y + h)
    return by_img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True, nargs="+",
                    help="one or more run dirs (e.g. an initial batch + a later "
                         "disjoint --skip/--exclude_run extension of it). On "
                         "img_id collision across dirs, the LAST --run wins — "
                         "same policy as datagen/assemble_dataset.py.")
    ap.add_argument("--split", default="train")
    ap.add_argument("--max_objects", type=int, default=40)
    ap.add_argument("--outdir", type=Path, default=None,
                    help="default: analysis/ under the first --run dir")
    args = ap.parse_args()

    root = Path(__file__).parent.parent
    run_dirs = [r if r.is_absolute() else root / r for r in args.run]
    outdir = args.outdir or (run_dirs[0] / "analysis")
    outdir.mkdir(parents=True, exist_ok=True)

    # ── Load run(s): merge with last-run-wins per img_id (matches assemble_dataset.py) ──
    winner: dict = {}                              # img_id -> (path, line_no)
    for run in run_dirs:
        for shard in sorted(run.glob("shard_*.jsonl")):
            with open(shard) as f:
                for i, line in enumerate(f):
                    try:
                        iid = json.loads(line)["img_id"]
                    except Exception:                          # noqa: BLE001
                        continue
                    winner[iid] = (shard, i)

    records = []
    for run in run_dirs:
        for shard in sorted(run.glob("shard_*.jsonl")):
            with open(shard) as f:
                for i, line in enumerate(f):
                    r = json.loads(line)
                    if winner.get(r["img_id"]) == (shard, i):
                        records.append(r)
    drops: Counter = Counter()
    sec_per_img = []
    for run in run_dirs:
        for sp in sorted(run.glob("summary_shard_*.json")):
            s = json.loads(sp.read_text())
            drops.update(s.get("drops", {}))
            sec_per_img.append(s["timing"]["sec_per_img"])
    print(f"{len(run_dirs)} run dir(s) → {len(records)} images (deduplicated), "
          f"{len(sec_per_img)} shard summaries")

    print("Loading MegaSG boxes …")
    boxes = load_boxes(args.split, args.max_objects)

    # ── Pass over relations ─────────────────────────────────────────────────
    preds, sem_preds, spa_preds = Counter(), Counter(), Counter()
    spa_cats, dirs = Counter(), Counter()
    rels_per_img, sem_per_img, spa_per_img, n_obj_per_img = [], [], [], []
    degrees_all, iso_share, comp_share, density = [], [], [], []
    n_geo = n_dup = n_selfloop = n_zero = 0
    audit = {"containment": Counter(), "contact": Counter(), "attention": Counter()}

    for r in records:
        rels = r.get("relations", [])
        bx = boxes.get(r["img_id"], {})
        n_zero += not rels
        seen, deg = set(), Counter()
        n_sem = n_spa = 0
        for x in rels:
            p, s, o = x["predicate"], x["subject_id"], x["object_id"]
            preds[p] += 1
            key = (s, p, o)
            n_dup += key in seen
            seen.add(key)
            n_selfloop += s == o
            deg[s] += 1
            deg[o] += 1
            if is_spatial(x):
                n_spa += 1
                spa_preds[p] += 1
                n_geo += x.get("source") == "geometric"
                cat = spa_category(p)
                spa_cats[cat] += 1
                c = canonicalize_spatial(p, collapse_spatial=True)
                if (cat in ("horizontal", "vertical", "depth")
                        and c in SPA_CAT):
                    dirs[c] += 1
            else:
                n_sem += 1
                sem_preds[p] += 1
            # residual geometry audit (post-gate noise measurement)
            if s in bx and o in bx:
                sb, ob = bx[s], bx[o]
                if p in CONTAINMENT:
                    audit["containment"]["ok" if contain_frac(ob, sb) >= 0.3
                                         else "weak" if iou(sb, ob) > 0 else "disjoint"] += 1
                elif p in CONTACT:
                    audit["contact"]["overlap" if iou(sb, ob) > 0 else "disjoint"] += 1
                elif p in ATTENTION:
                    audit["attention"]["overlap" if iou(sb, ob) > 0 else "disjoint"] += 1

        rels_per_img.append(len(rels))
        sem_per_img.append(n_sem)
        spa_per_img.append(n_spa)
        n_objects = len(bx) or r.get("n_objects", 0)
        n_obj_per_img.append(n_objects)
        img_deg = [deg.get(oid, 0) for oid in bx]
        degrees_all.extend(img_deg)
        if img_deg:
            iso_share.append(sum(d == 0 for d in img_deg) / len(img_deg))
            density.append(len(rels) / len(img_deg))
        # connected components (union-find over objects with edges)
        parent = {oid: oid for oid in bx}

        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        for x in rels:
            s, o = x["subject_id"], x["object_id"]
            if s in parent and o in parent:
                parent[find(s)] = find(o)
        if parent:
            comps = Counter(find(a) for a in parent)
            comp_share.append(comps.most_common(1)[0][1] / len(parent))

    tot = sum(preds.values())
    n_img = len(records)

    # ── 01: top-20 predicates, stacked sem/spa ──────────────────────────────
    top20 = preds.most_common(20)
    labels = [p for p, _ in top20][::-1]
    sem_c = [sem_preds.get(p, 0) for p in labels]
    spa_c = [spa_preds.get(p, 0) for p in labels]
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(labels, sem_c, color=C_SEM, label="semantic layer")
    ax.barh(labels, spa_c, left=sem_c, color=C_SPA, label="spatial layer")
    for i, (a, b) in enumerate(zip(sem_c, spa_c)):
        ax.text(a + b + tot * 0.001, i, f"{(a + b) / tot * 100:.1f}%",
                va="center", fontsize=8)
    ax.set_xlabel("relation count")
    ax.set_title(f"Top-20 predicates ({tot:,} relations, {len(preds)} unique)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(outdir / "01_top20_predicates.png", dpi=130)

    # ── 01b: top-50 predicates, stacked sem/spa ─────────────────────────────
    top50 = preds.most_common(50)
    labels50 = [p for p, _ in top50][::-1]
    sem_c50 = [sem_preds.get(p, 0) for p in labels50]
    spa_c50 = [spa_preds.get(p, 0) for p in labels50]
    fig, ax = plt.subplots(figsize=(9, 15))
    ax.barh(labels50, sem_c50, color=C_SEM, label="semantic layer")
    ax.barh(labels50, spa_c50, left=sem_c50, color=C_SPA, label="spatial layer")
    for i, (a, b) in enumerate(zip(sem_c50, spa_c50)):
        ax.text(a + b + tot * 0.001, i, f"{(a + b) / tot * 100:.1f}%",
                va="center", fontsize=7)
    ax.set_xlabel("relation count")
    ax.set_title(f"Top-50 predicates ({tot:,} relations, {len(preds)} unique)")
    ax.tick_params(axis="y", labelsize=8)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(outdir / "07_top50_predicates.png", dpi=130)

    # ── 02: long tail ───────────────────────────────────────────────────────
    freqs = np.array(sorted(preds.values(), reverse=True))
    cum = np.cumsum(freqs) / tot
    hapax = int((freqs == 1).sum())
    le5 = int((freqs <= 5).sum())
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.6))
    a1.loglog(np.arange(1, len(freqs) + 1), freqs, ".", ms=3, color=C_SEM)
    a1.set_xlabel("predicate rank")
    a1.set_ylabel("frequency")
    a1.set_title(f"Zipf rank–frequency ({len(freqs)} predicates)")
    a1.grid(alpha=.3, which="both")
    a2.plot(np.arange(1, len(freqs) + 1), 100 * cum, color=C_ACC)
    for k in (10, 20, 50, 100):
        if k <= len(cum):
            a2.axvline(k, ls=":", c="gray", lw=.8)
            a2.annotate(f"top{k}: {100 * cum[k - 1]:.0f}%", (k, 100 * cum[k - 1]),
                        textcoords="offset points", xytext=(4, -12), fontsize=8)
    a2.set_xscale("log")
    a2.set_xlabel("predicate rank")
    a2.set_ylabel("% of relation mass")
    a2.set_title(f"cumulative coverage — hapax {hapax}, ≤5 uses {le5}")
    a2.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(outdir / "02_longtail_zipf.png", dpi=130)

    # ── 03: per-image counts + scaling with objects ─────────────────────────
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.6))
    bins = np.arange(0, max(rels_per_img) + 2) -.5
    a1.hist(rels_per_img, bins=bins, color=C_ACC, alpha=.75, label="total")
    a1.hist(sem_per_img, bins=bins, histtype="step", lw=1.8, color=C_SEM, label="semantic")
    a1.hist(spa_per_img, bins=bins, histtype="step", lw=1.8, color=C_SPA, label="spatial")
    a1.set_xlabel("relations per image")
    a1.set_ylabel("images")
    a1.set_xlim(-0.5, np.percentile(rels_per_img, 99.5))
    a1.set_title(f"relations/image (mean {np.mean(rels_per_img):.2f}, "
                 f"median {np.median(rels_per_img):.0f}, zero-rel {n_zero})")
    a1.legend()
    no, rp = np.array(n_obj_per_img), np.array(rels_per_img)
    xs = sorted(set(no[no <= np.percentile(no, 99)]))
    mean = [rp[no == x].mean() for x in xs]
    p25 = [np.percentile(rp[no == x], 25) for x in xs]
    p75 = [np.percentile(rp[no == x], 75) for x in xs]
    a2.fill_between(xs, p25, p75, alpha=.25, color=C_SEM, label="p25–p75")
    a2.plot(xs, mean, color=C_SEM, lw=2, label="mean")
    a2.set_xlabel("objects in image")
    a2.set_ylabel("relations")
    a2.set_title("relation count vs scene size")
    a2.legend()
    a2.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(outdir / "03_rels_per_image.png", dpi=130)

    # ── 04: graph degree ────────────────────────────────────────────────────
    deg = np.array(degrees_all)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.6))
    a1.hist(deg, bins=np.arange(0, np.percentile(deg, 99.5) + 2) -.5,
            color=C_SEM, alpha=.8)
    a1.set_yscale("log")
    a1.set_xlabel("relations per object (degree)")
    a1.set_ylabel("objects (log)")
    a1.set_title(f"node degree — mean {deg.mean():.2f}, median {np.median(deg):.0f}, "
                 f"isolated {100 * (deg == 0).mean():.1f}%")
    a2.hist(comp_share, bins=40, color=C_ACC, alpha=.8)
    a2.set_xlabel("largest connected component (fraction of objects)")
    a2.set_ylabel("images")
    a2.set_title(f"graph connectivity — mean {np.mean(comp_share):.2f}, "
                 f"fully connected {100 * np.mean(np.array(comp_share) == 1):.0f}% of images")
    fig.tight_layout()
    fig.savefig(outdir / "04_degree.png", dpi=130)

    # ── 05: spatial breakdown + direction balance ───────────────────────────
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.6))
    cats = spa_cats.most_common()
    a1.bar([c for c, _ in cats], [v for _, v in cats], color=C_SPA, alpha=.85)
    n_spa_tot = sum(spa_cats.values())
    for i, (_, v) in enumerate(cats):
        a1.text(i, v, f"{100 * v / n_spa_tot:.1f}%", ha="center", va="bottom", fontsize=9)
    a1.set_ylabel("relations")
    a1.set_title(f"spatial layer by category ({n_spa_tot:,} rels, "
                 f"{len(spa_preds)} surface forms, H={entropy(spa_preds):.2f})")
    pairs = [("in front of", "behind"), ("to the left of", "to the right of"),
             ("above", "below")]
    x = np.arange(len(pairs))
    a2.bar(x -.18, [dirs.get(a, 0) for a, _ in pairs],.36, color=C_SEM, label="A")
    a2.bar(x +.18, [dirs.get(b, 0) for _, b in pairs],.36, color=C_SPA, label="B")
    a2.set_xticks(x, [f"{a}\nvs {b}" for a, b in pairs], fontsize=8)
    a2.set_ylabel("relations")
    a2.set_title("direction balance (canonical forms)")
    fig.tight_layout()
    fig.savefig(outdir / "05_spatial_breakdown.png", dpi=130)

    # ── 06: noise — drops + residual audit ──────────────────────────────────
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.8))
    dd = drops.most_common(14)
    a1.barh([k for k, _ in dd][::-1], [v for _, v in dd][::-1], color="#911eb4", alpha=.8)
    a1.set_xlabel("relations dropped")
    a1.set_title(f"gate/verifier drops during generation ({sum(drops.values()):,} total)")
    names, ok_v, bad_v = [], [], []
    for cat, good_keys in [("containment", ("ok", "weak")), ("contact", ("overlap",)),
                           ("attention", ("overlap",))]:
        c = audit[cat]
        good = sum(c.get(k, 0) for k in good_keys)
        bad = c.get("disjoint", 0)
        names.append(f"{cat}\n(n={good + bad:,})")
        s = max(good + bad, 1)
        ok_v.append(100 * good / s)
        bad_v.append(100 * bad / s)
    a2.bar(names, ok_v, color=C_ACC, alpha=.85, label="boxes consistent")
    a2.bar(names, bad_v, bottom=ok_v, color="#e6194b", alpha=.85, label="boxes disjoint")
    for i, b in enumerate(bad_v):
        a2.text(i, 101, f"{b:.1f}% disjoint", ha="center", fontsize=9)
    a2.set_ylim(0, 112)
    a2.set_ylabel("% of audited relations")
    a2.set_title("residual geometry audit (post-gate)")
    a2.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(outdir / "06_noise_gates.png", dpi=130)

    # ── stats.json ──────────────────────────────────────────────────────────
    stats = {
        "images": n_img, "relations": tot,
        "rels_per_img": round(tot / n_img, 2),
        "sem_per_img": round(float(np.mean(sem_per_img)), 2),
        "spa_per_img": round(float(np.mean(spa_per_img)), 2),
        "pct_spatial": round(100 * sum(spa_per_img) / tot, 1),
        "geo_topup_rels": n_geo,
        "unique_predicates": len(preds),
        "unique_semantic": len(sem_preds), "unique_spatial": len(spa_preds),
        "entropy_all": round(entropy(preds), 3),
        "entropy_semantic": round(entropy(sem_preds), 3),
        "entropy_spatial": round(entropy(spa_preds), 3),
        "hapax_predicates": hapax, "predicates_le5": le5,
        "top10_mass_pct": round(100 * float(cum[9]), 1),
        "top50_mass_pct": round(100 * float(cum[49]), 1) if len(cum) >= 50 else None,
        "top50": top50,
        "zero_rel_images": n_zero,
        "duplicate_triples": n_dup, "self_loops": n_selfloop,
        "degree_mean": round(float(deg.mean()), 2),
        "degree_median": float(np.median(deg)),
        "isolated_objects_pct": round(100 * float((deg == 0).mean()), 1),
        "largest_component_share_mean": round(float(np.mean(comp_share)), 3),
        "fully_connected_images_pct": round(100 * float(np.mean(np.array(comp_share) == 1)), 1),
        "sec_per_img_by_shard": sec_per_img,
        "drops": dict(drops),
        "audit": {k: dict(v) for k, v in audit.items()},
        "spatial_categories": dict(spa_cats),
        "direction_balance": dict(dirs),
        "top20": top20,
    }
    (outdir / "stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps({k: v for k, v in stats.items()
                      if k not in ("drops", "audit", "top20", "sec_per_img_by_shard")},
                     indent=2))
    print(f"\n6 plots + stats.json → {outdir}/")


if __name__ == "__main__":
    main()
