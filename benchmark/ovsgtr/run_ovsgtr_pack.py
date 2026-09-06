"""Run an OvSGTR checkpoint over a RelSGG pack and emit the common interchange format.

WHY THIS EXISTS
---------------
A cross-model comparison is only meaningful if BOTH models are scored by the SAME
evaluator on the SAME images with the SAME vocabulary. OvSGTR's native path
(`main.py` -> `datasets/vg.py` -> `SggEvaluator`) hardwires VG150's stanford_filtered
HDF5 layout and their own metric code, so it can answer none of our benchmark
questions. Here we import only their *model*, drive it ourselves, and write a
model-agnostic record that `benchmark/ovsgtr/eval_interchange.py` scores with the identical
evaluator used for RelSGG.

VOCABULARY IS OPEN
------------------
OvSGTR scores relations by dot-product against the encoded `rel_caption` and maps
token spans back to predicate names (graph_infer.py:87-94), so an arbitrary benchmark
vocabulary can be installed with no model change. This is ONLY true for the
open-vocabulary checkpoints: `sgg_mode='full'` ignores rel_captions entirely
(groundingdino.py:420) and uses a fixed 51-way VG150 classifier plus a VG150
frequency prior. Running `vg-full-*` on a non-VG150 vocabulary is therefore refused
below rather than silently producing meaningless columns.

Runs under OvSGTR's isolated venv (python 3.11 / torch 2.1.2), NOT the RelSGG venv.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from benchmark.ovsgtr.ovsgtr_common import (OVSGTR_ROOT, build_prompts,  # noqa: E402
                                            install_vocabulary, load_ovsgtr,
                                            _ensure_ovsgtr_on_path)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def resized_hw(w: int, h: int, size: int = 800, max_size: int = 1333):
    """Byte-identical to datasets/transforms.py:108-126 (their test-time resize).

    Reimplemented rather than imported because their T.RandomResize operates on
    (image, target) pairs and would drag in the whole augmentation stack.
    """
    if max_size is not None:
        mn, mx = float(min(w, h)), float(max(w, h))
        if mx / mn * size > max_size:
            size = int(round(max_size * mn / mx))
    if (w <= h and w == size) or (h <= w and h == size):
        return h, w
    if w < h:
        return int(size * h / w), size
    return size, int(size * w / h)


class Detections:
    """Shared open-vocabulary detections from training/detect_boxes.py --set_classes.

    Feeding the SAME boxes to every SGG model is the whole point of the SGDet arm: it
    removes detector quality as a confound so the comparison isolates the relation
    head. Because the detector was prompted with the pack's own categories, `cls`
    already indexes meta["categories"] and needs no remap.
    """

    def __init__(self, path: Path, n_images: int):
        d = np.load(path)
        if int(d["n_images"]) != n_images:
            raise SystemExit(f"detection file covers {int(d['n_images'])} images but the "
                             f"pack has {n_images}; wrong split?")
        order = np.argsort(d["img_idx"], kind="stable")
        self.idx = d["img_idx"][order]
        self.xyxy = d["xyxy"][order].astype(np.float32)
        self.conf = d["conf"][order].astype(np.float32)
        self.cls = d["cls"][order].astype(np.int64)
        self.starts = np.searchsorted(self.idx, np.arange(n_images + 1))

    def boxes_for(self, i: int, W: int, H: int, conf_thr: float):
        """Return (cxcywh-normalised boxes, 0-based category ids), best-first."""
        a, b = int(self.starts[i]), int(self.starts[i + 1])
        xyxy, conf, cls = self.xyxy[a:b], self.conf[a:b], self.cls[a:b]
        keep = conf >= conf_thr
        xyxy, conf, cls = xyxy[keep], conf[keep], cls[keep]
        if len(xyxy) == 0:
            return np.zeros((0, 4), np.float32), np.zeros(0, np.int64)
        o = np.argsort(-conf)
        xyxy, cls = xyxy[o], cls[o]
        cx = (xyxy[:, 0] + xyxy[:, 2]) / 2 / W
        cy = (xyxy[:, 1] + xyxy[:, 3]) / 2 / H
        w = (xyxy[:, 2] - xyxy[:, 0]) / W
        h = (xyxy[:, 3] - xyxy[:, 1]) / H
        return np.clip(np.stack([cx, cy, w, h], 1), 0.0, 1.0).astype(np.float32), cls


class Pack:
    """Minimal reader for a RelSGG pack; deliberately free of relsgg imports so this
    file stays runnable inside OvSGTR's venv."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.meta = json.loads((self.root / "meta.json").read_text())
        self.file_names = json.loads((self.root / "file_names.json").read_text())
        self.img_meta = np.load(self.root / "img_meta.npy")
        self.boxes = np.load(self.root / "boxes.npy", mmap_mode="r")
        self.box_cats = np.load(self.root / "box_cats.npy", mmap_mode="r")
        self.img_dir = Path(self.meta["img_dir"])
        self.predicates = list(self.meta["predicates"])
        self.categories = list(self.meta["categories"])

    def __len__(self):
        return len(self.img_meta)

    def item(self, i: int):
        image_id, w, h, b0, nb, _r0, _nr = (int(x) for x in self.img_meta[i])
        boxes = np.asarray(self.boxes[b0:b0 + nb], dtype=np.float32)      # cxcywh, normalised
        cats = np.asarray(self.box_cats[b0:b0 + nb], dtype=np.int64)      # 0-based into categories
        return dict(index=i, image_id=image_id, width=w, height=h,
                    file_name=self.file_names[i], boxes=boxes, cats=cats)


@torch.no_grad()
def run(args):
    _ensure_ovsgtr_on_path()
    from util.misc import nested_tensor_from_tensor_list

    pack = Pack(args.pack)
    if args.img_dir:
        # packs built elsewhere carry that machine's image roots in meta.json
        pack.img_dir = Path(args.img_dir)
    predicates, categories = pack.predicates, pack.categories
    cat_index = {n: i for i, n in enumerate(categories)}
    if args.caption_scope == "image":
        # Per-image captions need the supplied labels (fake-predcls path) and real
        # names: the ablation modes rewrite the GLOBAL slot->name binding.
        if args.boxes not in ("gt", "external"):
            raise SystemExit("--caption_scope image needs --boxes gt|external")
        if args.caption_mode != "true":
            raise SystemExit("--caption_scope image is incompatible with --caption_mode")
    # --caption_mode ablates the OBJECT-CATEGORY channel. OvSGTR's detector is
    # prompted with the noun list, so its object queries are grounded to category
    # TEXT and the relation head reads those queries. Swapping the display names
    # while keeping the slot->index mapping intact removes the semantics without
    # touching the index space, so predicted labels still mean "pack category i"
    # and the evaluator needs no change.
    display = list(categories)
    if args.caption_mode == "pseudo":
        # 150 distinct, short, meaning-free codes. Distinct is required:
        # install_vocabulary builds a name->index dict, so duplicate names would
        # collapse slots rather than blank them.
        import string
        codes = [a + b for a in string.ascii_lowercase for b in string.ascii_lowercase]
        if len(codes) < len(categories):
            raise SystemExit("not enough 2-letter codes for this vocabulary")
        display = codes[:len(categories)]
    elif args.caption_mode == "shuffled":
        # Real nouns, permuted across slots: the vocabulary stays semantically
        # real (so detection stays good) but the name<->index binding is broken.
        rng = np.random.default_rng(args.caption_seed)
        display = [categories[i] for i in rng.permutation(len(categories))]
    if args.caption_mode != "true":
        print(f"  caption_mode={args.caption_mode}: object names replaced "
              f"(e.g. {categories[0]!r} -> {display[0]!r})")
    prompts = build_prompts(display, predicates)

    # 'external' also uses the fake-predcls path: it substitutes arbitrary boxes via
    # Hungarian matching, which is exactly what lets us install detector boxes.
    model, post, cfg, rep = load_ovsgtr(args.config, args.checkpoint, device=args.device,
                                        use_gt_box=args.boxes in ("gt", "external"))
    det = None
    if args.boxes == "external":
        if not args.det:
            raise SystemExit("--boxes external requires --det")
        det = Detections(Path(args.det), len(pack))
    # `bert.embeddings.position_ids` is HuggingFace's constant position-index buffer
    # (arange(max_position_embeddings)), which transformers >= 4.31 stopped persisting.
    # Checkpoints saved before that carry it, later ones do not: vg-ovr-swint and
    # vg-ovdr-swinb-mega-best include it, vg-ovr-swinb does not. It is a CONSTANT the
    # model rebuilds identically at construction, so its absence changes nothing --
    # verified here by comparing the LIVE buffer against arange, not assumed.
    # Everything else still fails loudly; this guard has caught real config/checkpoint
    # mismatches and must not be widened into a blanket strict=False.
    benign = set()
    for k in list(rep["missing"]):
        if not k.endswith("embeddings.position_ids"):
            continue
        buf = model
        for part in k.split("."):
            buf = getattr(buf, part, None)
            if buf is None:
                break
        if buf is not None and torch.equal(buf.flatten().cpu(),
                                           torch.arange(buf.numel())):
            benign.add(k)
    if benign:
        print(f"  ignoring {len(benign)} constant index buffer(s) absent from the "
              f"checkpoint, verified == arange: {sorted(benign)}")
    hard = [k for k in rep["missing"] if k not in benign]
    if hard or rep["unexpected"]:
        raise RuntimeError(f"checkpoint did not load cleanly: "
                           f"missing={hard} unexpected={rep['unexpected']}")

    sgg_mode = getattr(cfg, "sgg_mode", None)
    if sgg_mode == "full" and len(predicates) != 50:
        raise SystemExit(
            f"refusing to run: sgg_mode='full' has a fixed 51-way VG150 relation head and "
            f"ignores rel_captions, but this pack has {len(predicates)} predicates. Use an "
            f"open-vocabulary checkpoint (sgg_mode='ovdr'/'ovr') for cross-vocabulary eval.")
    install_vocabulary(post, display, predicates)

    if args.boxes in ("gt", "external"):
        # The configs set nms_iou_threshold=0.5, and the postprocessor applies it even
        # on the fake-predcls path where every score is tied at 1.0 — so overlapping
        # same-class GT boxes get SUPPRESSED (measured: image 1159501, 23 GT -> 21).
        # Under a GT-box protocol both models must receive the identical box set, and
        # boxes dropped here are relations OvSGTR can never recover, so this can only
        # help the baseline. -1 is PostProcess's own default (groundingdino.py:575) and
        # takes the order-preserving prefix branch instead.
        if getattr(post, "nms_iou_threshold", -1) > 0:
            print(f"  {args.boxes}-box mode: disabling NMS (was {post.nms_iou_threshold}) "
                  f"so the supplied box set is passed through intact")
            post.nms_iou_threshold = -1.0

    # Both prompts are tokenized and truncated to max_text_len independently
    # (groundingdino.py:329) before being concatenated, so a category whose span falls
    # past the cut would be unreachable by the matcher and silently unscored.
    max_text_len = getattr(cfg, "max_text_len", 512)
    checks = [("rel_caption", prompts["rel_caption"])]
    if args.caption_scope == "global":
        checks.insert(0, ("caption", prompts["caption"]))
    else:
        # SpatialSense-style packs: 1,279 category names but ~3 boxes per image.
        # One caption per image fits where the global one cannot; each is checked
        # against max_text_len inside the loop.
        print(f"  caption_scope=image: caption rebuilt per image from its own "
              f"{len(categories)}-way categories")
    for what, text in checks:
        n_tok = len(model.tokenizer(text).input_ids)
        if n_tok > max_text_len:
            raise SystemExit(f"{what} is {n_tok} tokens > max_text_len={max_text_len}; "
                             f"entries past the cut would be silently dropped")
        print(f"  {what}: {n_tok} tokens (limit {max_text_len})")

    n = len(pack) if args.limit <= 0 else min(args.limit, len(pack))
    if not (0 <= args.shard < args.num_shards):
        raise SystemExit(f"--shard {args.shard} out of range for --num_shards {args.num_shards}")
    # Strided, not blocked: consecutive images have correlated cost (box count), so a
    # stride keeps the shards balanced without measuring anything.
    todo = list(range(args.shard, n, args.num_shards))
    if args.num_shards > 1:
        print(f"shard {args.shard}/{args.num_shards}: {len(todo)} of {n} images")
    V = len(predicates) + 1  # graph_infer reserves column 0 for background

    out_index, out_pairs, out_scores = [], [], []
    out_boxes, out_labels, out_bscores = [], [], []
    pair_ptr, box_ptr = [0], [0]
    n_empty = 0
    t0 = time.time()

    for loop_i, i in enumerate(todo):
        it = pack.item(i)
        img = Image.open(pack.img_dir / it["file_name"]).convert("RGB")
        W, H = img.size
        oh, ow = resized_hw(W, H)
        t = TF.normalize(TF.to_tensor(TF.resize(img, [oh, ow])), IMAGENET_MEAN, IMAGENET_STD)
        samples = nested_tensor_from_tensor_list([t.to(args.device)])

        supplied = None  # cxcywh-normalised boxes we hand to the fake-predcls path
        cat_ids = None
        if args.boxes in ("gt", "external"):
            # The postprocessor's fake-predcls path consumes cxcywh in [0,1] and
            # rescales by the ORIGINAL size (groundingdino.py:757-760), which is what
            # the pack already stores. Labels are 1-based to match install_vocabulary.
            if args.boxes == "gt":
                supplied, cat_ids = it["boxes"], it["cats"]
            else:
                supplied, cat_ids = det.boxes_for(i, W, H, args.det_conf)
                if len(supplied) < 2:
                    # graph_infer needs >1 node to form any pair; record as empty
                    # rather than letting it fall into its 1-object warning branch.
                    n_empty += 1
                    pair_ptr.append(pair_ptr[-1]); box_ptr.append(box_ptr[-1])
                    out_index.append(it["index"])
                    continue
        local_to_global = None
        if args.caption_scope == "image":
            # This image's own categories, first appearance order, as the caption.
            # Labels index THAT list; decoded labels are mapped back below so the
            # interchange stays 1-based against the pack's global category list.
            local_names = list(dict.fromkeys(categories[c] for c in cat_ids))
            name_to_local = {n_: j for j, n_ in enumerate(local_names)}
            local_ids = np.asarray([name_to_local[categories[c]] for c in cat_ids],
                                   dtype=np.int64)
            local_to_global = np.asarray([cat_index[n_] for n_ in local_names],
                                         dtype=np.int64)
            cap = build_prompts(local_names, predicates)
            n_tok = len(model.tokenizer(cap["caption"]).input_ids)
            if n_tok > max_text_len:
                raise SystemExit(f"image {it['image_id']}: caption is {n_tok} tokens > "
                                 f"max_text_len={max_text_len}")
            install_vocabulary(post, local_names, predicates)
            labels_1based = local_ids + 1
        else:
            cap = prompts
            labels_1based = None if cat_ids is None else cat_ids + 1
        target = {"caption": cap["caption"], "rel_caption": cap["rel_caption"]}
        if supplied is not None:
            target["boxes"] = torch.as_tensor(supplied.copy(), device=args.device)
            target["labels"] = torch.as_tensor(labels_1based, device=args.device)
        targets = [target]

        outputs = model(samples, targets)
        orig_sizes = torch.as_tensor([[H, W]], device=args.device)

        if post.use_gt_box:
            # losses.py:710 populates target['input_ids'] inside the criterion, which we
            # skip entirely; the matcher (matcher.py:91-99) needs it together with
            # 'gt_names' to locate each GT category's token span in the caption. It
            # asserts on a miss, so a truncated caption fails loudly rather than
            # mis-matching.
            target["input_ids"] = outputs["input_ids"][0]
            target["gt_names"] = [categories[c] for c in cat_ids]
            match_ids = post.matcher(outputs, targets)
            gt_dicts = [{"ids": ids, "gt_boxes": target["boxes"], "gt_labels": target["labels"]}
                        for ids in match_ids]
            results = post(outputs, orig_sizes, gt_dicts=gt_dicts)
        else:
            results = post(outputs, orig_sizes)

        # PostProcess nests the SGG output under res['graph'] (groundingdino.py:827);
        # the top-level dict only carries the detection fields.
        r = results[0].get("graph", {})
        pairs = r.get("all_node_pairs")
        rel = r.get("all_relation")
        if pairs is None or rel is None or len(pairs) == 0:
            n_empty += 1
            pair_ptr.append(pair_ptr[-1])
            box_ptr.append(box_ptr[-1])
            out_index.append(it["index"])
            continue

        pairs = np.asarray(pairs, dtype=np.int32)
        rel = np.asarray(rel, dtype=np.float32)
        assert rel.shape[1] == V, f"expected {V} relation columns, got {rel.shape[1]}"
        if args.max_pairs > 0 and len(pairs) > args.max_pairs:
            # graph_infer already sorted pairs by descending score, so a prefix is
            # the top-scoring subset, not an arbitrary one.
            pairs, rel = pairs[:args.max_pairs], rel[:args.max_pairs]

        pb = np.asarray(r["pred_boxes"], dtype=np.float32)     # xyxy, absolute
        pl = np.asarray(r["pred_boxes_class"], dtype=np.int32)
        ps = np.asarray(r["pred_boxes_score"], dtype=np.float32)
        if local_to_global is not None:
            # 1-based into this image's caption -> 1-based into the pack categories;
            # 0 (background) is left alone.
            idx = np.clip(pl - 1, 0, len(local_to_global) - 1)
            pl = np.where(pl > 0, local_to_global[idx] + 1, 0).astype(np.int32)

        if supplied is not None:
            # `all_node_pairs` indexes the surviving nodes, and downstream scoring
            # assumes node i == supplied box i. That identity is NOT guaranteed: the
            # postprocessor runs batched_nms at nms_iou_threshold=0.5 (groundingdino.py
            #:774) which can drop overlapping same-class boxes and reorder the rest
            # (all fake-predcls scores are tied at 1.0). Verify per image rather than
            # trusting a spot check, since a silent permutation would corrupt every
            # triplet without changing any count.
            g = supplied.astype(np.float64)
            gxyxy = np.stack([(g[:, 0] - g[:, 2] / 2) * W, (g[:, 1] - g[:, 3] / 2) * H,
                              (g[:, 0] + g[:, 2] / 2) * W, (g[:, 1] + g[:, 3] / 2) * H], 1)
            if len(pb) != len(gxyxy) or not np.allclose(gxyxy, pb, atol=1e-2):
                raise RuntimeError(
                    f"box identity broken on image {it['image_id']} "
                    f"(n_in={len(gxyxy)} n_out={len(pb)}); NMS reordered or suppressed "
                    f"boxes, so pair indices no longer address the supplied boxes")

        out_index.append(it["index"])
        out_pairs.append(pairs)
        out_scores.append(rel.astype(np.float16))
        out_boxes.append(pb)
        out_labels.append(pl)
        out_bscores.append(ps)
        pair_ptr.append(pair_ptr[-1] + len(pairs))
        box_ptr.append(box_ptr[-1] + len(pb))

        if (loop_i + 1) % args.log_every == 0:
            el = time.time() - t0
            print(f"[{loop_i+1}/{len(todo)}] {el:.0f}s  {(loop_i+1)/el:.2f} img/s  "
                  f"empty={n_empty}", flush=True)

    def cat(chunks, dim_shape, dtype):
        return np.concatenate(chunks, 0) if chunks else np.zeros(dim_shape, dtype=dtype)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        image_index=np.asarray(out_index, dtype=np.int32),
        pair_ptr=np.asarray(pair_ptr, dtype=np.int64),
        box_ptr=np.asarray(box_ptr, dtype=np.int64),
        pairs=cat(out_pairs, (0, 2), np.int32),
        rel_scores=cat(out_scores, (0, V), np.float16),
        boxes=cat(out_boxes, (0, 4), np.float32),
        labels=cat(out_labels, (0,), np.int32),
        box_scores=cat(out_bscores, (0,), np.float32),
        predicates=np.asarray(predicates),
        categories=np.asarray(categories),
        meta=np.asarray([json.dumps({
            "model": "OvSGTR", "checkpoint": str(args.checkpoint), "config": str(args.config),
            "pack": str(args.pack), "box_source": args.boxes, "sgg_mode": sgg_mode,
            "det": str(args.det) if args.det else None,
            "det_conf": args.det_conf if args.boxes == "external" else None,
            "caption_scope": args.caption_scope,
            "img_dir": str(pack.img_dir),
            "rln_freq_bias": getattr(cfg, "rln_freq_bias", None),
            "score_semantics": "softmax" if getattr(cfg, "rln_freq_bias", None) else "sigmoid",
            "bg_column": 0, "n_images": n, "n_empty": n_empty,
            # Their postprocessor's category space reserves 0 for __background__, so the
            # labels stored here are 1-based against `categories`. Consumers that show
            # object NAMES must subtract this; anything matching on box index is
            # unaffected, which is why it stayed invisible until the LLM judge.
            "label_base": 1,
            "ovsgtr_root": str(OVSGTR_ROOT),
            "shard": args.shard, "num_shards": args.num_shards, "pack_n": n,
        })]),
)
    print(f"wrote {out}  images={n} empty={n_empty} pairs={pair_ptr[-1]} "
          f"({time.time()-t0:.0f}s)")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pack", required=True, help="RelSGG pack split dir, e.g. runs/packed/psg/test")
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--boxes", choices=["gt", "det", "external"], default="gt",
                   help="gt = their PredCls path (GT boxes AND GT labels); det = their own "
                        "detector (their native SGDet); external = shared open-vocab "
                        "detector boxes, the controlled SGDet comparison")
    p.add_argument("--det", default=None, help="npz from training/detect_boxes.py --set_classes")
    p.add_argument("--det_conf", type=float, default=0.10,
                   help="confidence floor for --boxes external")
    p.add_argument("--device", default="cuda")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--num_shards", type=int, default=1,
                   help="split the pack across N processes/GPUs. Images are independent "
                        "(batch-size-1 forward), so a shard's per-image output is "
                        "bitwise identical to the unsharded run; merge with "
                        "benchmark/ovsgtr/merge_ovsgtr_shards.py.")
    p.add_argument("--shard", type=int, default=0, help="this shard's index in [0, num_shards)")
    p.add_argument("--max_pairs", type=int, default=0, help="0 = keep all")
    p.add_argument("--caption_mode", default="true",
                   choices=["true", "pseudo", "shuffled"],
                   help="ablate the object-category channel: 'pseudo' replaces the "
                        "150 nouns with meaning-free codes, 'shuffled' permutes the "
                        "real nouns across slots. Index space is unchanged in both.")
    p.add_argument("--caption_seed", type=int, default=0)
    p.add_argument("--caption_scope", default="global", choices=["global", "image"],
                   help="'global' installs one caption holding every pack category "
                        "(VG150, PSG); 'image' rebuilds the caption per image from that "
                        "image's own box categories, for packs whose category list "
                        "exceeds max_text_len (SpatialSense: 1,279 names, ~3 boxes/image)")
    p.add_argument("--img_dir", default=None,
                   help="override meta['img_dir'] (packs built elsewhere carry that machine's roots)")
    p.add_argument("--log_every", type=int, default=50)
    run(p.parse_args())


if __name__ == "__main__":
    main()
