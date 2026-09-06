#!/usr/bin/env python3
"""
sgg_postprocess.py — deterministic cleanup of an unpooled relation union.

Calibration showed any LLM "write the final clean list" pass re-imposes the
model's ~4.5-rels/img prior, so density only survives in the unpooled union of
per-object/per-pair passes (anchored, anchOPEN). That union is dense but carries
mechanical junk: passive inverses, directional contradictions, support/spatial
floods. This module removes that junk with deterministic rules — preserving
density and diversity, unlike an LLM consolidation pass.

Rules (all order-preserving, only ever REMOVE):
  1. exact-duplicate drop (after canonicalization)
  2. long-predicate drop  (> 5 words → run-on hallucination)
  3. same-label physical drop  (e.g. "person wearing person" — physical/clothing
     predicates applied between two entities with the same label are almost always
     mis-attribution or subject/object confusion)
  4. passive-inverse drop  ("X worn by Y" when "Y wearing X" exists; any "... by")
  5. support-inverse drop   ("A supporting B" when "B on/sitting on A" exists)
  6. directional-contradiction drop (pair has both "in front of" and "behind", etc.)
  7. spatial-when-action drop (drop a directional-spatial edge if the pair already
     has a real action/contact relation in either direction)
  8. spatial cap (≤ max_spatial_per_subject directional-spatial edges per subject)

Use as a library (cleanup(rels)->rels) or via --postprocess in sgg_vllm_generate,
or standalone to clean an existing run dir.
"""
from __future__ import annotations

import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from sgg_canon import canonicalize, canonicalize_spatial

SPATIAL = {"above", "below", "in front of", "behind", "to the left of",
           "to the right of", "between", "inside", "near"}
OPP = {"above": "below", "below": "above", "in front of": "behind",
       "behind": "in front of", "to the left of": "to the right of",
       "to the right of": "to the left of"}
_ON = {"on", "sitting on", "resting on", "standing on", "lying on", "mounted on"}

# Physical/clothing predicates that make no sense between two entities of the same
# label (e.g. "person wearing person"). Social predicates (looking at, talking to,
# embracing) are NOT in this set — person↔person social relations are valid.
_SAME_LABEL_PHYSICAL = frozenset({
    "wearing", "part of", "containing", "covering",
})


def cleanup(rels: list, max_spatial_per_subject: int = 1,
            drop_spatial_with_action: bool = True,
            collapse_spatial: bool = True) -> list:
    """Return a cleaned copy of a relation-dict list (preserves all fields).

    Inverse + contradiction removal always apply (pure correctness). The spatial
    pruning is tunable: with directional spatial relations acceptable (30-50% OK),
    raise max_spatial_per_subject and set drop_spatial_with_action=False to keep
    them — they add layout density without hurting correctness.

    collapse_spatial=False emits the model's spatial surface form ("atop" stays
    "atop") for open-vocab predicate diversity; the fully-canonical form is still
    used internally for dedup / contradiction / cap logic, so one spatial concept
    per pair regardless of phrasing."""
    # canonical form for logic, surface form for output; drop dups on the canonical key.
    # Rels from a dedicated spatial layer use the spatial-aware canonicalizer:
    # geometry-verified proximity ("near", "next to") is signal there, not filler.
    norm, seen = [], set()
    for r in rels:
        canon_fn = (canonicalize_spatial
                    if r.get("spatial") or r.get("source") == "geometric"
                    else canonicalize)
        p = canon_fn(r.get("predicate", ""))                    # logic key (collapsed)
        if p is None:
            continue
        p_out = p if collapse_spatial else canon_fn(r.get("predicate", ""),
                                                    collapse_spatial=False)
        s, o = r.get("subject_id"), r.get("object_id")
        if (s, o, p) in seen:
            continue
        seen.add((s, o, p))
        norm.append((s, o, p, p_out, r))

    # Rule 2+3: drop long predicates and same-label physical pairs (fast pre-filter)
    filtered = []
    for s, o, p, p_out, r in norm:
        if len(p.split()) > 5:                                            # run-on predicate
            continue
        sl = r.get("subject_label", ""); ol = r.get("object_label", "")
        if sl and sl == ol and p in _SAME_LABEL_PHYSICAL:                 # e.g. person wearing person
            continue
        filtered.append((s, o, p, p_out, r))
    norm = filtered

    preds = defaultdict(set)                          # (s,o) -> {canon preds}
    for s, o, p, _, _ in norm:
        preds[(s, o)].add(p)
    has_action = lambda a, b: any(x not in SPATIAL for x in preds.get((a, b), ()))

    kept, spatial_count = [], Counter()
    for s, o, p, p_out, r in norm:
        if p.endswith(" by") and preds.get((o, s)):                      # passive inverse
            continue
        if p == "supporting" and (preds.get((o, s), set()) & _ON):       # support inverse
            continue
        if p in SPATIAL:
            opp = OPP.get(p)
            if opp and opp in preds.get((s, o), ()):                     # contradiction same dir
                continue
            if drop_spatial_with_action and (has_action(s, o) or has_action(o, s)):
                continue                                                 # action beats layout
            if spatial_count[s] >= max_spatial_per_subject:              # cap floods
                continue
            spatial_count[s] += 1
        out = dict(r)
        out["predicate"] = p_out
        kept.append(out)
    return kept


# ── Standalone: clean an existing run dir ──────────────────────────────────────
def _main():
    import argparse, json
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--max_spatial_per_subject", type=int, default=1)
    ap.add_argument("--keep_spatial_with_action", action="store_true",
                    help="Keep a directional-spatial edge even if the pair has an action.")
    ap.add_argument("--suffix", default="_clean")
    args = ap.parse_args()

    shard = next(iter(sorted(args.run.glob("shard_*.jsonl"))))
    out = args.run / f"shard_0000{args.suffix}.jsonl"
    n_before = n_after = n_img = 0
    with open(shard) as f, open(out, "w") as w:
        for line in f:
            r = json.loads(line)
            rels = r.get("relations", [])
            cl = cleanup(rels, args.max_spatial_per_subject,
                         drop_spatial_with_action=not args.keep_spatial_with_action)
            n_before += len(rels); n_after += len(cl); n_img += 1
            r["relations"] = cl
            w.write(json.dumps(r) + "\n")
    print(f"  {n_img} images: {n_before/max(n_img,1):.2f} -> {n_after/max(n_img,1):.2f} rels/img")
    print(f"  → {out}")


if __name__ == "__main__":
    _main()
