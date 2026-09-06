"""Convert SVG + ASv2 + VG150-train into dedup-filtered COCO-SGG jsons.

Consumes the overlap registry from build_datamix_registry.py and emits, under
runs/datamix/:

    svg_vg_train_coco.json      SVG-SG vg portion    (imgs: VG150_coco_format/train)
    svg_psg_train_coco.json     SVG-SG psg portion   (imgs: PSG_coco_format/train)
    asv2_train_coco.json        ASv2 ReC triplets    (imgs: PSG_coco_format/train)
    vg150_train_dedup_coco.json VG150 train minus protected photos

Every output is restricted to (a) images physically on disk and (b) images NOT
in the protected eval sets (PSG val/test, VG150 val/test, bridged through both
VG<->COCO id maps). Predicate strings are kept verbatim (lowercased/stripped
only) — the synonym-preserving policy applies to new sources too.

Schema matches pack_megasg.py's input: images / annotations (xywh px,
category_id, global ids) / categories / rel_categories / rel_annotations
(subject_id, object_id = global annotation ids; predicate_id).

Usage:
    python training/convert_datamix.py            # all four outputs
    python training/convert_datamix.py --only asv2
"""
from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from relsgg.paths import DATASETS, DATAMIX, RUNS, REPO  # noqa: E402

# Dataset and run roots come from relsgg.paths (RA_DATASETS / RA_RUNS).
MIX = DATAMIX

REF_RE = re.compile(r"<ref>(.*?)</ref><box>(\[\[.*?\]\])</box>")
PRED_RE = re.compile(r"<pred>(.*?)</pred><box>(\[\[.*?\]\])</box><box>(\[\[.*?\]\])</box>")


def norm_pred(p: str) -> str:
    p = re.sub(r"\s+", " ", p.strip().lower())
    return p if 0 < len(p) <= 60 else ""


class CocoSGGWriter:
    """Accumulates images/boxes/relations and writes one COCO-SGG json."""

    def __init__(self):
        self.images, self.annotations, self.rel_annotations = [], [], []
        self.cat_ids: dict[str, int] = {}
        self.pred_ids: dict[str, int] = {}
        self.pred_counts: Counter = Counter()
        self._next_ann = 1

    def cat(self, name: str) -> int:
        return self.cat_ids.setdefault(name, len(self.cat_ids))

    def pred(self, name: str) -> int:
        return self.pred_ids.setdefault(name, len(self.pred_ids))

    def add_image(self, img_id, file_name, w, h,
                  boxes, cats, rels, rel_extras=None,
                  allow_empty_rels: bool = False) -> None:
        """boxes: [[x,y,w,h] px]; cats: [name]; rels: [(si, oi, pred_str)];
        rel_extras: optional per-relation dicts (aligned with rels) merged into
        the rel_annotation records — e.g. {"source": "geometric"} to route a
        relation through the pack's FLAG_GEOMETRIC training downweight.
        allow_empty_rels: keep an image that has NO relations. Off by default
        (a relation-less image is useless for training and would silently
        dilute every source). Haystack needs it: 9,151 of its 11,368 images
        carry only NEGATIVE annotations, which live in a sidecar, and dropping
        them would make those negatives unscoreable."""
        if len(boxes) < 2 or (not rels and not allow_empty_rels):
            return
        gids = []
        for b, c in zip(boxes, cats):
            gids.append(self._next_ann)
            self.annotations.append({
                "id": self._next_ann, "image_id": img_id,
                "bbox": [float(v) for v in b], "category_id": self.cat(c),
            })
            self._next_ann += 1
        kept = 0
        seen = set()
        for k_r, (si, oi, p) in enumerate(rels):
            p = norm_pred(p)
            if not p or si == oi or (si, oi, p) in seen:
                continue
            seen.add((si, oi, p))
            rec = {
                "image_id": img_id, "subject_id": gids[si],
                "object_id": gids[oi], "predicate_id": self.pred(p),
            }
            if rel_extras is not None and rel_extras[k_r]:
                rec.update(rel_extras[k_r])
            self.rel_annotations.append(rec)
            self.pred_counts[p] += 1
            kept += 1
        if kept == 0 and not allow_empty_rels:
            # roll back boxes for an image that contributed no relations
            del self.annotations[-len(gids):]
            self._next_ann -= len(gids)
            return
        self.images.append({"id": img_id, "file_name": file_name,
                            "width": int(w), "height": int(h)})

    def write(self, path: Path) -> None:
        out = {
            "images": self.images,
            "annotations": self.annotations,
            "categories": [{"id": i, "name": n}
                           for n, i in sorted(self.cat_ids.items(), key=lambda x: x[1])],
            "rel_categories": [{"id": i, "name": n}
                               for n, i in sorted(self.pred_ids.items(), key=lambda x: x[1])],
            "rel_annotations": self.rel_annotations,
        }
        json.dump(out, open(path, "w"))
        print(f"[{path.name}] imgs={len(self.images)} boxes={len(self.annotations)} "
              f"rels={len(self.rel_annotations)} preds={len(self.pred_ids)} "
              f"cats={len(self.cat_ids)}")


def load_registry():
    reg = json.load(open(MIX / "registry.json"))
    # banned_coco_ids (psg_image_free policy: eval-protected ∪ ALL PSG photos)
    # supersedes the original eval-only protected set when present.
    prot_coco = set(reg.get("banned_coco_ids") or reg["protected_coco_ids"])
    prot_vg = set(reg["protected_vg_ids"])
    vg2coco = {k: int(v) for k, v in json.load(open(MIX / "vg2coco.json")).items()}
    psg2coco = {k: int(v) for k, v in json.load(open(MIX / "psg2coco.json")).items()}
    return prot_coco, prot_vg, vg2coco, psg2coco


def disk_ids(d: Path) -> set:
    return {f[:-4] for f in os.listdir(d) if f.endswith(".jpg")}


def convert_svg(portion: str, prot_coco, prot_vg, vg2coco, psg2coco) -> None:
    """portion: 'vg' or 'psg'."""
    import pyarrow.parquet as pq
    if portion == "vg":
        img_root = DATASETS / "VG150_coco_format/train"
        allowed = disk_ids(img_root)
        def keep(r):
            vid = r["image_id"][:-4]
            if vid not in allowed or vid in prot_vg:
                return None
            if vid in vg2coco and vg2coco[vid] in prot_coco:
                return None
            return vid, f"{vid}.jpg"
    else:
        img_root = DATASETS / "PSG_coco_format/train"
        allowed = disk_ids(img_root)
        def keep(r):
            pid = str(r["id"])
            if pid not in allowed:
                return None
            c = psg2coco.get(pid)
            if c is not None and c in prot_coco:
                return None
            return pid, f"{pid}.jpg"

    w = CocoSGGWriter()
    n_drop = 0
    shard_dir = DATASETS / f"SVG/sg_{portion}"
    for shard in sorted(shard_dir.glob("*.parquet")):
        f = pq.ParquetFile(shard)
        for batch in f.iter_batches(batch_size=512):
            for r in batch.to_pylist():
                k = keep(r)
                if k is None:
                    n_drop += 1
                    continue
                img_key, fname = k
                regions = r["regions"]
                if not regions:
                    continue
                H, W = regions[0]["segmentation"]["size"]
                sg = json.loads(r["scene_graph"])
                boxes = [reg["bbox"] for reg in regions]
                cats = [reg["object"] or "object" for reg in regions]
                rels = [(s, o, p) for s, o, p in sg["relations"]]
                w.add_image(int(img_key), fname, W, H, boxes, cats, rels)
    w.write(MIX / f"svg_{portion}_train_coco.json")
    print(f"  (dropped {n_drop} rows: protected or image not on disk)")


def parse_boxes(s: str):
    try:
        v = json.loads(s)
        return [b for b in v if isinstance(b, list) and len(b) == 4]
    except Exception:
        return []


def convert_asv2(prot_coco, prot_vg, vg2coco, psg2coco) -> None:
    coco2psg = {}
    for p, c in psg2coco.items():
        coco2psg.setdefault(c, p)
    psg_train = disk_ids(DATASETS / "PSG_coco_format/train")
    psg_meta = {str(x["image_id"]): (x["width"], x["height"])
                for x in json.load(open(DATASETS / "VG_metadata/psg.json"))["data"]}
    coco_fn = re.compile(r"(\d{12})\.jpg$")

    # group all conversations per usable image across the three files
    per_img: dict[str, list] = {}
    n_drop_prot = n_drop_nodisk = 0
    for f in sorted((DATASETS / "ASv2").glob("*.json")):
        for r in json.load(open(f)):
            m = coco_fn.search(r["image"])
            if not ("coco" in r["image"] and m):
                n_drop_nodisk += 1
                continue
            cid = int(m.group(1))
            if cid in prot_coco:
                n_drop_prot += 1
                continue
            pid = coco2psg.get(cid)
            if pid is None or pid not in psg_train:
                n_drop_nodisk += 1
                continue
            per_img.setdefault(pid, []).append(r)

    w = CocoSGGWriter()
    for pid, recs in per_img.items():
        W, H = psg_meta[pid]
        box_key_to_idx: dict[tuple, int] = {}
        boxes, cats, rels = [], [], []

        def bidx(b, name=None):
            # ASv2 boxes are normalized [0,1000) — convert to pixels
            key = tuple(int(v) for v in b)
            if key not in box_key_to_idx:
                box_key_to_idx[key] = len(boxes)
                x0, y0, x1, y1 = [v / 1000.0 for v in key]
                boxes.append([x0 * W, y0 * H, max((x1 - x0) * W, 1.0),
                              max((y1 - y0) * H, 1.0)])
                cats.append(name or "object")
            elif name and cats[box_key_to_idx[key]] == "object":
                cats[box_key_to_idx[key]] = name
            return box_key_to_idx[key]

        for r in recs:
            for turn in r.get("conversations", []):
                if turn.get("from") != "gpt":
                    continue
                text = turn.get("value", "")
                for name, bs in REF_RE.findall(text):
                    nm = re.sub(r"\s+", " ", name.strip().lower())[:60] or "object"
                    for b in parse_boxes(bs):
                        bidx(b, nm)
                for pred, subs, objs in PRED_RE.findall(text):
                    for sb in parse_boxes(subs):
                        for ob in parse_boxes(objs):
                            si, oi = bidx(sb), bidx(ob)
                            rels.append((si, oi, pred))
        w.add_image(int(pid), f"{pid}.jpg", W, H, boxes, cats, rels)
    w.write(MIX / "asv2_train_coco.json")
    print(f"  (dropped {n_drop_prot} protected + {n_drop_nodisk} "
          f"not-on-disk/non-coco records)")


def convert_gqa(prot_coco, prot_vg, vg2coco, psg2coco) -> None:
    """GQA scene graphs (train+val jsons) -> mix. GQA images ARE VG images
    (same ids); graphs are VG annotations normalized/cleaned by GQA — this
    REPLACES vg150_train in the mix (same underlying human annotations, so
    keeping both would duplicate supervision, not diversify it).

    GQA's "to the left of"/"to the right of" edges are auto-derived from box
    geometry (not human): tagged source="geometric" so the pack flags them and
    training's existing geometric downweight applies.
    """
    GEO_PREDS = {"to the left of", "to the right of"}
    img_root = DATASETS / "VG150_coco_format/train"
    allowed = disk_ids(img_root)
    w = CocoSGGWriter()
    n_drop_prot = n_drop_disk = n_geo = n_hum = 0
    for src in ("train_sceneGraphs.json", "val_sceneGraphs.json"):
        d = json.load(open(DATASETS / "GQA" / src))
        for vgid, g in d.items():
            if vgid not in allowed:
                n_drop_disk += 1
                continue
            if vgid in prot_vg or (vgid in vg2coco and vg2coco[vgid] in prot_coco):
                n_drop_prot += 1
                continue
            oid_list = list(g["objects"].keys())
            oid_to_idx = {o: i for i, o in enumerate(oid_list)}
            boxes, cats = [], []
            for o in oid_list:
                ob = g["objects"][o]
                boxes.append([float(ob["x"]), float(ob["y"]),
                              max(float(ob["w"]), 1.0), max(float(ob["h"]), 1.0)])
                cats.append(ob["name"].strip().lower() or "object")
            rels, extras = [], []
            for o in oid_list:
                si = oid_to_idx[o]
                for r in g["objects"][o].get("relations", []):
                    oi = oid_to_idx.get(r["object"])
                    if oi is None:
                        continue
                    p = r["name"].strip().lower()
                    rels.append((si, oi, p))
                    if p in GEO_PREDS:
                        extras.append({"source": "geometric"})
                        n_geo += 1
                    else:
                        extras.append(None)
                        n_hum += 1
            w.add_image(int(vgid), f"{vgid}.jpg", g["width"], g["height"],
                        boxes, cats, rels, rel_extras=extras)
    w.write(MIX / "gqa_train_coco.json")
    print(f"  (dropped {n_drop_prot} protected, {n_drop_disk} not on disk; "
          f"rels: {n_hum} human + {n_geo} geometric left/right)")


def convert_vg150_dedup(prot_coco, prot_vg, vg2coco, psg2coco) -> None:
    src = DATASETS / "VG150_coco_format/train/_annotations.coco.json"
    d = json.load(open(src))
    drop = set()
    for im in d["images"]:
        vid = os.path.splitext(im["file_name"])[0]
        if vid in prot_vg or (vid in vg2coco and vg2coco[vid] in prot_coco):
            drop.add(im["id"])
    d["images"] = [im for im in d["images"] if im["id"] not in drop]
    d["annotations"] = [a for a in d["annotations"] if a["image_id"] not in drop]
    d["rel_annotations"] = [r for r in d["rel_annotations"]
                            if r["image_id"] not in drop]
    json.dump(d, open(MIX / "vg150_train_dedup_coco.json", "w"))
    print(f"[vg150_train_dedup_coco.json] dropped {len(drop)} protected imgs "
          f"-> imgs={len(d['images'])} rels={len(d['rel_annotations'])}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["svg_vg", "svg_psg", "asv2", "vg150",
                                       "gqa"],
                    default=None)
    args = ap.parse_args()
    MIX.mkdir(parents=True, exist_ok=True)
    reg = load_registry()
    if args.only in (None, "svg_vg"):
        convert_svg("vg", *reg)
    if args.only == "svg_psg":       # retired (psg_image_free) — explicit only
        convert_svg("psg", *reg)
    if args.only == "asv2":          # retired (psg_image_free) — explicit only
        convert_asv2(*reg)
    if args.only == "vg150":         # replaced by GQA — explicit only
        convert_vg150_dedup(*reg)
    if args.only in (None, "gqa"):
        convert_gqa(*reg)


if __name__ == "__main__":
    main()
