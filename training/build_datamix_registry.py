"""Build the image-overlap registry for the multi-dataset training mix.

The mix adds SVG (jamepark3922/svg), ASv2 (OpenGVLab/AS-V2) and VG150-train to
the MEGASG base. All of them draw images from Visual Genome and/or COCO — and
so do our eval sets (PSG val/test are COCO images; VG150 val/test are VG
images; roughly half of VG's photos ARE COCO photos under different filenames).
Training on any photo that appears in an eval split leaks the benchmark, so
this script builds the canonical "protected" id sets and reports, per source,
exactly how many candidate images are dropped and why.

Identity bridges:
  * VG id  <-> COCO id: VG image_data.json (`coco_id` field, 51,498 mapped)
  * PSG id <-> COCO id: OpenPSG psg.json (`coco_image_id`, all 48,749 mapped)

Protected sets (dropped from ALL train sources):
  * PSG   val + test  (user's primary zero-shot benchmark)  as COCO ids
  * VG150 val + test  (secondary diagnostic)                 as VG ids
... each bridged into the OTHER id space too, so a VG-named photo that is
  physically a PSG-val COCO photo is caught (and vice versa).
MEGASG val is Objects365 — no id bridge to VG/COCO exists; treated as disjoint
(different crawl), flagged as residual risk in PLAN.md.

Outputs runs/datamix/registry.json with the protected sets + maps digest.

Usage:
    python training/build_datamix_registry.py
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from relsgg.paths import DATASETS, DATAMIX, RUNS, REPO  # noqa: E402

# Dataset and run roots come from relsgg.paths (RA_DATASETS / RA_RUNS).
META = DATASETS / "VG_metadata"
OUT = DATAMIX

COCO_FN = re.compile(r"(\d{12})\.jpg$")


def disk_ids(d: Path) -> set:
    return {f[:-4] for f in os.listdir(d) if f.endswith(".jpg")}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    # ---- identity bridges ----
    vg_meta = json.load(open(META / "image_data.json"))
    vg2coco = {str(x["image_id"]): int(x["coco_id"])
               for x in vg_meta if x.get("coco_id")}
    coco2vg = {}
    for v, c in vg2coco.items():
        coco2vg.setdefault(c, set()).add(v)

    psg = json.load(open(META / "psg.json"))["data"]
    psg2coco = {str(x["image_id"]): int(x["coco_image_id"]) for x in psg}

    # ---- on-disk split id sets ----
    vg_splits = {s: disk_ids(DATASETS / f"VG150_coco_format/{s}")
                 for s in ("train", "val", "test")}
    psg_splits = {s: disk_ids(DATASETS / f"PSG_coco_format/{s}")
                  for s in ("train", "val", "test")}

    # ---- protected sets ----
    psg_eval = psg_splits["val"] | psg_splits["test"]
    vg_eval = vg_splits["val"] | vg_splits["test"]

    prot_coco = {psg2coco[i] for i in psg_eval if i in psg2coco}
    prot_coco |= {vg2coco[i] for i in vg_eval if i in vg2coco}
    prot_vg = set(vg_eval)
    for c in list(prot_coco):
        prot_vg |= coco2vg.get(c, set())

    print(f"protected: {len(psg_eval)} PSG eval imgs + {len(vg_eval)} VG150 "
          f"eval imgs -> {len(prot_coco)} COCO ids, {len(prot_vg)} VG ids")

    # ---- per-source overlap report ----
    report = {}

    # 1. VG150 train (photo identity leaks via vg->coco)
    tr = vg_splits["train"]
    hit = {i for i in tr if i in vg2coco and vg2coco[i] in prot_coco}
    hit |= tr & prot_vg
    report["vg150_train"] = {"total": len(tr), "dropped": len(hit)}
    print(f"[vg150_train]  {len(tr)} imgs, drop {len(hit)} "
          f"(photo-identical to a protected eval image)")

    # 2. SVG-VG (image_id = "<vgid>.jpg")
    import pyarrow.parquet as pq
    svg_vg_ids = set()
    for shard in sorted((DATASETS / "SVG/sg_vg").glob("*.parquet")):
        t = pq.read_table(shard, columns=["image_id"])
        svg_vg_ids |= {v[:-4] for v in t.column("image_id").to_pylist()}
    hit = svg_vg_ids & prot_vg
    hit |= {i for i in svg_vg_ids if i in vg2coco and vg2coco[i] in prot_coco}
    on_disk = svg_vg_ids & (vg_splits["train"] | vg_eval)
    usable = (on_disk & vg_splits["train"]) - hit
    report["svg_vg"] = {"total": len(svg_vg_ids), "dropped_protected": len(hit),
                        "no_image_on_disk": len(svg_vg_ids - on_disk),
                        "usable": len(usable)}
    print(f"[svg_vg]       {len(svg_vg_ids)} imgs, drop {len(hit)} protected, "
          f"{len(svg_vg_ids - on_disk)} images not on disk -> usable {len(usable)}")

    # 3. SVG-PSG (id = PSG image id; image_id = COCO path)
    svg_psg_ids = set()
    for shard in sorted((DATASETS / "SVG/sg_psg").glob("*.parquet")):
        t = pq.read_table(shard, columns=["id"])
        svg_psg_ids |= {str(v) for v in t.column("id").to_pylist()}
    hit = svg_psg_ids & psg_eval
    hit |= {i for i in svg_psg_ids
            if i in psg2coco and psg2coco[i] in prot_coco}
    usable = (svg_psg_ids & psg_splits["train"]) - hit
    report["svg_psg"] = {"total": len(svg_psg_ids), "dropped_protected": len(hit),
                         "usable": len(usable)}
    print(f"[svg_psg]      {len(svg_psg_ids)} imgs, drop {len(hit)} protected "
          f"-> usable {len(usable)} (on-disk PSG train imgs)")

    # 4. ASv2 (image paths: coco/train2017/000000xxx.jpg, vg/VG_100K*/id.jpg,...)
    coco2psg = {}
    for p, c in psg2coco.items():
        coco2psg.setdefault(c, p)
    asv2_imgs = set()
    for f in sorted((DATASETS / "ASv2").glob("*.json")):
        d = json.load(open(f))
        asv2_imgs |= {r["image"] for r in d}
    as_coco, as_vg, as_other = set(), set(), set()
    for p in asv2_imgs:
        m = COCO_FN.search(p)
        if "coco" in p and m:
            as_coco.add(int(m.group(1)))
        elif "VG_100K" in p:
            as_vg.add(os.path.basename(p)[:-4])
        else:
            as_other.add(p)
    hit_c = as_coco & prot_coco
    hit_v = as_vg & prot_vg
    hit_v |= {i for i in as_vg if i in vg2coco and vg2coco[i] in prot_coco}
    usable_c = {c for c in as_coco - hit_c
                if coco2psg.get(c) in psg_splits["train"]}
    usable_v = (as_vg & vg_splits["train"]) - hit_v
    report["asv2"] = {"total_imgs": len(asv2_imgs), "coco": len(as_coco),
                      "vg": len(as_vg), "other_source": len(as_other),
                      "dropped_protected": len(hit_c) + len(hit_v),
                      "usable_coco": len(usable_c), "usable_vg": len(usable_v)}
    print(f"[asv2]         {len(asv2_imgs)} imgs ({len(as_coco)} coco, "
          f"{len(as_vg)} vg, {len(as_other)} other), drop "
          f"{len(hit_c)+len(hit_v)} protected -> usable {len(usable_c)} coco "
          f"(on-disk via PSG train) + {len(usable_v)} vg")

    # ---- write registry ----
    json.dump({
        "protected_coco_ids": sorted(prot_coco),
        "protected_vg_ids": sorted(prot_vg),
        "report": report,
    }, open(OUT / "registry.json", "w"))
    # bridges as separate files (used by converters; too big for one json)
    json.dump(vg2coco, open(OUT / "vg2coco.json", "w"))
    json.dump({str(k): v for k, v in psg2coco.items()},
              open(OUT / "psg2coco.json", "w"))
    print(f"\nwrote {OUT}/registry.json (+ vg2coco.json, psg2coco.json)")


if __name__ == "__main__":
    main()
