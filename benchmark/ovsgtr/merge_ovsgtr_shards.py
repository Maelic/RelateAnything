"""Merge sharded run_ovsgtr_pack.py outputs into one npz, ordered by image index.

Each shard ran a batch-size-1 forward over a disjoint stride of the pack, so a
merged file is bitwise identical per image to an unsharded run. This asserts
that rather than assuming it: shards must agree on vocabulary and settings, and
their image sets must be disjoint and cover the pack exactly.

    python benchmark/ovsgtr/merge_ovsgtr_shards.py \
        --shards runs/ovsgtr/vg-ovr-swint_vg150_test_det.shard*.npz \
        --out runs/ovsgtr/vg-ovr-swint_vg150_test_det.npz
"""
from __future__ import annotations
import argparse, json
import numpy as np


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--shards", nargs="+", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()

    # Materialise once. NpzFile is lazy and re-inflates the WHOLE array on every
    # __getitem__, so slicing inside the per-image loop below would decompress every
    # array 26k times -- that is what OOM-killed the first merge attempt.
    ds = [{k: v for k, v in np.load(f, allow_pickle=False).items()}
          for f in sorted(a.shards)]
    metas = [json.loads(str(d["meta"][0])) for d in ds]
    ref = metas[0]
    for f, m in zip(sorted(a.shards), metas):
        for k in ("checkpoint", "config", "pack", "box_source", "sgg_mode", "label_base"):
            if m.get(k) != ref.get(k):
                raise SystemExit(f"{f}: {k}={m.get(k)!r} != {ref.get(k)!r}; refusing to merge")
    for d in ds[1:]:
        if not np.array_equal(d["predicates"], ds[0]["predicates"]) or \
           not np.array_equal(d["categories"], ds[0]["categories"]):
            raise SystemExit("vocabulary differs between shards; refusing to merge")

    idx = np.concatenate([d["image_index"] for d in ds])
    if len(np.unique(idx)) != len(idx):
        raise SystemExit("shards overlap (duplicate image indices)")
    n_expected = ref.get("pack_n")
    if n_expected is not None and len(idx) != int(n_expected):
        raise SystemExit(f"coverage gap: merged {len(idx)} images, pack has {n_expected}. "
                         "A shard is missing or died — rerun it, do not merge partial output.")

    order = np.argsort(idx, kind="stable")
    pairs, rels, boxes, labels, bsc = [], [], [], [], []
    pair_ptr, box_ptr = [0], [0]
    # global position -> (shard, local row)
    src = np.concatenate([np.full(len(d["image_index"]), s, np.int32)
                          for s, d in enumerate(ds)])
    loc = np.concatenate([np.arange(len(d["image_index"]), dtype=np.int64) for d in ds])
    for g in order:
        d, j = ds[src[g]], int(loc[g])
        pa, pb = int(d["pair_ptr"][j]), int(d["pair_ptr"][j + 1])
        ba, bb = int(d["box_ptr"][j]), int(d["box_ptr"][j + 1])
        pairs.append(d["pairs"][pa:pb]); rels.append(d["rel_scores"][pa:pb])
        boxes.append(d["boxes"][ba:bb]); labels.append(d["labels"][ba:bb])
        bsc.append(d["box_scores"][ba:bb])
        pair_ptr.append(pair_ptr[-1] + (pb - pa)); box_ptr.append(box_ptr[-1] + (bb - ba))

    meta = dict(ref); meta.pop("shard", None)
    meta["num_shards"] = 1
    meta["merged_from"] = sorted(a.shards)
    np.savez_compressed(
        a.out,
        image_index=idx[order].astype(np.int32),
        pair_ptr=np.asarray(pair_ptr, np.int64), box_ptr=np.asarray(box_ptr, np.int64),
        pairs=np.concatenate(pairs) if pairs else np.zeros((0, 2), np.int32),
        rel_scores=np.concatenate(rels) if rels else np.zeros((0, 1), np.float16),
        boxes=np.concatenate(boxes) if boxes else np.zeros((0, 4), np.float32),
        labels=np.concatenate(labels) if labels else np.zeros((0,), np.int32),
        box_scores=np.concatenate(bsc) if bsc else np.zeros((0,), np.float32),
        predicates=ds[0]["predicates"], categories=ds[0]["categories"],
        meta=np.asarray([json.dumps(meta)]))
    print(f"wrote {a.out}: {len(idx):,} images, {pair_ptr[-1]:,} pairs, "
          f"{box_ptr[-1]:,} boxes from {len(ds)} shards")


if __name__ == "__main__":
    main()
