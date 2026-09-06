"""build_heldout_text_space.py — rebuild the text-space artifacts for a held-out
predicate vocabulary, consistently.

WHY THIS IS NEEDED, and why a partial job is worse than none. Holding predicates out
of training shrinks the vocabulary, and THREE artifacts are keyed to that vocabulary
by POSITION, not by name:

  pred_embeds_*.npz     rows of W, one per predicate. train.py asserts
                        `npz predicates == dataset vocabulary` (build_model), so a
                        stale bank fails loudly. This is the safe one.
  union_meta.json       `predicates`; PredicateOntology.from_soft_supervision
                        asserts the npz vocabulary equals THIS list.
  soft_supervision.npz  the dangerous one. pos_i/pos_j, neg_i/neg_j, inv_i/inv_j are
                        int32 INDICES into its predicate list, and `sym` is per
                        predicate. Drop 15 predicates from the model's vocabulary
                        without remapping these and every index past the first removed
                        entry points at the WRONG predicate -- pos_w/neg_lw are
                        [V, V] matrices consumed by id, so the synonym supervision is
                        silently scrambled rather than absent. Nothing would crash.

So all three are rebuilt from one index map, or none are. Pairs touching a removed
predicate are dropped (they cannot be expressed in the smaller space); every other
pair is renumbered. pair_opportunity.npz is NOT touched -- it is category-keyed
(2884^2), unrelated to the predicate vocabulary.

    python training/build_heldout_text_space.py \
        --src runs/packed/datamix_v22/text_space \
        --drop training/ovr15.json \
        --out runs/packed/datamix_v22/text_space_heldout-ovr15
"""
from __future__ import annotations

import argparse
import json
import os
import shutil

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="source text_space dir")
    ap.add_argument("--drop", required=True, help="json list of predicate strings")
    ap.add_argument("--out", required=True)
    ap.add_argument("--embeds", default="pred_embeds_studentv2_512_photo.npz")
    ap.add_argument("--meta", default="union_meta.json")
    ap.add_argument("--soft", default="soft_supervision.npz")
    a = ap.parse_args()

    drop = set(json.load(open(a.drop)))
    os.makedirs(a.out, exist_ok=True)

    # ---- the single index map every artifact is rebuilt from ----------------------
    zb = np.load(os.path.join(a.src, a.embeds), allow_pickle=False)
    old = [str(p) for p in zb["predicates"]]
    keep_mask = np.array([p not in drop for p in old])
    keep_idx = np.flatnonzero(keep_mask)
    new = [old[i] for i in keep_idx]
    remap = np.full(len(old), -1, dtype=np.int64)
    remap[keep_idx] = np.arange(len(keep_idx))
    missing = drop - set(old)
    print(f"vocabulary {len(old):,} -> {len(new):,} (removed {len(old) - len(new)})")
    if missing:
        print(f"  WARNING: not in vocabulary, nothing removed for: {sorted(missing)}")
    if len(old) == len(new):
        raise SystemExit("!! nothing was removed; refusing to write artifacts that "
                         "would be mislabelled as held-out")

    # ---- 1. embeddings -----------------------------------------------------------
    np.savez(os.path.join(a.out, a.embeds),
             predicates=np.array(new),      # <U..., never dtype=object
             embeddings=zb["embeddings"][keep_idx],
             templates=zb["templates"])
    print(f"  wrote {a.embeds}: embeddings {zb['embeddings'][keep_idx].shape}")

    # ---- 2. ontology meta --------------------------------------------------------
    m = json.load(open(os.path.join(a.src, a.meta)))
    assert m["predicates"] == old, (
        f"{a.meta} vocabulary != {a.embeds} in the SOURCE; the artifacts were "
        "already inconsistent before this script ran")
    m["predicates"] = new
    m["predicate_counts"] = {k: v for k, v in m["predicate_counts"].items()
                             if k not in drop}
    before = len(m.get("raw_links", []))
    m["raw_links"] = [l for l in m.get("raw_links", [])
                      if l.get("predicate") not in drop]
    json.dump(m, open(os.path.join(a.out, a.meta), "w"))
    print(f"  wrote {a.meta}: {len(new):,} predicates, "
          f"raw_links {before} -> {len(m['raw_links'])}")

    # ---- 3. soft supervision (the index remap) -----------------------------------
    zs = np.load(os.path.join(a.src, a.soft), allow_pickle=False)
    assert [str(x) for x in zs["predicates"]] == old, (
        f"{a.soft} vocabulary != {a.embeds} in the SOURCE")
    out = {"predicates": np.array(new),
           "sym": zs["sym"][keep_idx]}
    for i_k, j_k, w_k in (("pos_i", "pos_j", "pos_w"),
                          ("neg_i", "neg_j", "neg_lw"),
                          ("inv_i", "inv_j", "inv_w")):
        i, j = zs[i_k], zs[j_k]
        ok = (remap[i] >= 0) & (remap[j] >= 0)
        out[i_k] = remap[i[ok]].astype(np.int32)
        out[j_k] = remap[j[ok]].astype(np.int32)
        out[w_k] = zs[w_k][ok]
        print(f"  {i_k[:3]} pairs {len(i):,} -> {int(ok.sum()):,} "
              f"({100 * (1 - ok.mean()):.2f}% touched a held-out predicate)")
    for k in zs.files:
        if k not in out:
            out[k] = zs[k]
    np.savez(os.path.join(a.out, a.soft), **out)
    print(f"  wrote {a.soft}")

    # carry anything else the dir holds so --canon_groups etc. still resolve
    for f in os.listdir(a.src):
        if f not in (a.embeds, a.meta, a.soft) and \
                os.path.isfile(os.path.join(a.src, f)) and \
                not os.path.exists(os.path.join(a.out, f)):
            shutil.copy2(os.path.join(a.src, f), os.path.join(a.out, f))

    # ---- verification: exactly the asserts train.py will make --------------------
    # allow_pickle=False deliberately: this is exactly how train.py and
    # PredicateOntology.from_soft_supervision open these files, and an object array
    # loads fine with allow_pickle=True while failing in the trainer.
    e = [str(p) for p in np.load(os.path.join(a.out, a.embeds),
                                 allow_pickle=False)["predicates"]]
    mm = json.load(open(os.path.join(a.out, a.meta)))["predicates"]
    ss = [str(p) for p in np.load(os.path.join(a.out, a.soft),
                                  allow_pickle=False)["predicates"]]
    assert e == mm == ss == new, "!! rebuilt artifacts disagree with each other"
    zz = np.load(os.path.join(a.out, a.soft), allow_pickle=False)
    V = len(new)
    for k in ("pos_i", "pos_j", "neg_i", "neg_j", "inv_i", "inv_j"):
        assert zz[k].size == 0 or (zz[k].min() >= 0 and zz[k].max() < V), \
            f"!! {k} out of range for the new vocabulary"
    assert not (set(new) & drop), "!! a held-out predicate survived"
    print(f"VERIFIED: 3 artifacts agree on {V:,} predicates, all indices in range, "
          f"no held-out predicate present -> {a.out}")


if __name__ == "__main__":
    main()
