"""Does the MODEL reproduce the data's `wearing` prior, or ignore geometry?

probe_wearing_conditional.py established the prior from the training packs:
P(annotated wearing | person-garment pair) collapses to ~0 when the boxes do
not overlap (megasg 0.001%, vg_raw 1.1%) and runs 20-80% when they do. This
script measures the SAME conditional for the model, on the SAME pairs, so the
two curves are directly comparable and no cross-dataset excuse is available:

  P(GT says wearing | pair, IoU bin)        <- reference, from this eval pack
  P(model emits wearing | pair, IoU bin)    <- at the shipped tau
  mean calibrated wearing score | IoU bin   <- is the score modulated at all?

Run on VG150, which is the only eval pack carrying both a person taxonomy and a
garment taxonomy (PSG is COCO-panoptic: no shirt/hat/jacket classes at all).

GT boxes, DEPLOYMENT shapes. At max_objects=32 the ordered-pair count is
32*31 = 992 = final_budget, so the sampler is a no-op and every person-garment
pair really is scored — no pair can be missing because the cascade dropped it.
`sampled` is reported anyway rather than assumed.

Also measures FUNCTIONALITY, which the data says is near-absolute (fan-in == 1
for 95.3% / 98.7% of garments): how often does the model hand ONE garment to
TWO wearers, and does the relation-interaction stack suppress that at all.

    python training/probe_wearing_model.py --checkpoint CK \
        --data_root runs/packed/vg150 --split test --tau 0.56
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from relsgg.data.dataset import RelationDataset, collate_fn  # noqa: E402
from relsgg.model.geometry import RelGeomEncoder  # noqa: E402
from relsgg.scoring import ScoreContract  # noqa: E402
from relsgg.text.student import encode_texts_student  # noqa: E402
from relsgg.vocabulary import TRAIN_TEMPLATES  # noqa: E402
from relsgg.checkpoint import build_model_from_ckpt  # noqa: E402
from research.probe_wearing_conditional import (# noqa: E402
    BINS, BIN_NAMES, GARMENT, PERSON, cross_geometry, cxcywh_to_xyxy)


@torch.no_grad()
def collect(model, loader, ds, dev, contract, tau, wear_ids, names, amp=True,
            pair_dump=None):
    is_person = np.array([bool(PERSON.match(c)) for c in ds.meta["categories"]])
    is_garment = np.array([bool(GARMENT.match(c)) for c in ds.meta["categories"]])
    wear_t = torch.tensor(sorted(wear_ids), dtype=torch.long)

    nb = len(BINS) - 1
    n_pair = np.zeros(nb, np.int64)      # candidate person-garment pairs
    n_gt = np.zeros(nb, np.int64)        #... annotated wearing
    n_emit = np.zeros(nb, np.int64)      #... model emits wearing at tau
    n_samp = np.zeros(nb, np.int64)      #... proposed by the sampler
    s_wear = np.zeros(nb, np.float64)    # sum of calibrated wearing score
    s_top = np.zeros(nb, np.float64)     # sum of calibrated top-1 score
    n_any = np.zeros(nb, np.int64)       #... model emits ANY predicate
    # DECOMPOSITION: the contract is sigmoid(a*(z_pred + w*z_pair) + b), so if
    # z_pair is already strongly negative on disjoint pairs and z_pred simply
    # outvotes it, re-weighting w is a ZERO-RETRAIN fix. If z_pair is flat, the
    # relatedness gate is not doing its job and only training can fix it.
    z_pred_sum = np.zeros(nb, np.float64)
    z_pair_sum = np.zeros(nb, np.float64)
    # what the model actually SAYS about a disjoint person-garment pair — the
    # question the render raised. Split so the overlap bucket is the control.
    say_disjoint, say_overlap = defaultdict(int), defaultdict(int)

    gt_fanin, em_fanin = defaultdict(int), defaultdict(int)
    n_img = n_img_pg = 0
    row = -1
    for images, boxes, counts, targets in loader:
        images = images.to(dev, non_blocking=True)
        boxes_d = boxes.to(dev, non_blocking=True)
        counts_d = counts.to(dev, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=amp, dtype=torch.bfloat16):
            out = model(images, boxes_d, counts_d, targets=None)
        zp = out["logits"].float()
        za = (out["pair_logits"].float() if out.get("pair_logits") is not None
              else torch.zeros_like(zp[..., 0]))
        probs = contract.scores(zp, za)                       # [B, P, V]
        top_s, top_i = probs.max(dim=-1)
        # z_pred of the ARGMAX column, so the decomposition matches the score
        # that was actually thresholded (not a vocabulary-wide mean).
        zp_top = zp.gather(-1, top_i.unsqueeze(-1)).squeeze(-1)
        wear_s = probs[..., wear_t.to(probs.device)].max(dim=-1).values
        is_wear_top = torch.isin(top_i, wear_t.to(top_i.device))

        for b in range(zp.shape[0]):
            row += 1
            n_img += 1
            n = int(counts[b])
            cats = targets[b]["entity_labels"].cpu().numpy()[:n]
            pi = np.flatnonzero(is_person[cats])
            oi = np.flatnonzero(is_garment[cats])
            if not len(pi) or not len(oi):
                continue
            n_img_pg += 1
            bx = cxcywh_to_xyxy(boxes[b][:n].cpu().numpy().astype(np.float64))
            iou, _, _ = cross_geometry(bx[pi], bx[oi])

            gt = set()
            rels = targets[b].get("relations")
            if rels is not None and len(rels):
                for r in rels.cpu().numpy():
                    if int(r[2]) in wear_ids:
                        gt.add((int(r[0]), int(r[1])))

            mask = out["valid_mask"][b]
            sub = out["sub_idx"][b][mask].cpu().numpy()
            obj = out["obj_idx"][b][mask].cpu().numpy()
            ts = top_s[b][mask].cpu().numpy()
            ws = wear_s[b][mask].cpu().numpy()
            iw = is_wear_top[b][mask].cpu().numpy()
            ti = top_i[b][mask].cpu().numpy()
            zpt = zp_top[b][mask].cpu().numpy()
            zat = za[b][mask].cpu().numpy()
            cell = {(int(s_), int(o_)): k for k, (s_, o_) in enumerate(zip(sub, obj))}

            for a, s_ in enumerate(pi):
                for c, o_ in enumerate(oi):
                    k = int(np.digitize(iou[a, c], BINS)) - 1
                    n_pair[k] += 1
                    if (int(s_), int(o_)) in gt:
                        n_gt[k] += 1
                        gt_fanin[(row, int(o_))] += 1
                    j = cell.get((int(s_), int(o_)))
                    if j is None:
                        continue
                    n_samp[k] += 1
                    s_wear[k] += float(ws[j])
                    s_top[k] += float(ts[j])
                    z_pred_sum[k] += float(zpt[j])
                    z_pair_sum[k] += float(zat[j])
                    if pair_dump is not None:
                        # z_pair does NOT depend on the predicate, so changing w
                        # shifts every column equally and the ARGMAX IS
                        # INVARIANT — only the threshold decision moves. Five
                        # numbers per pair are therefore enough to replay any
                        # (w, tau) offline on CPU.
                        pair_dump.append((iou[a, c], zpt[j], zat[j],
                                          1 if (int(s_), int(o_)) in gt else 0,
                                          1 if iw[j] else 0))
                    if ts[j] >= tau:
                        n_any[k] += 1
                        (say_disjoint if k == 0 else say_overlap)[
                            names[int(ti[j])]] += 1
                        if iw[j]:
                            n_emit[k] += 1
                            em_fanin[(row, int(o_))] += 1

    return dict(n_pair=n_pair, n_gt=n_gt, n_emit=n_emit, n_samp=n_samp,
                s_wear=s_wear, s_top=s_top, n_any=n_any, n_img=n_img,
                z_pred=z_pred_sum, z_pair=z_pair_sum,
                n_img_pg=n_img_pg,
                say_disjoint=dict(say_disjoint), say_overlap=dict(say_overlap),
                gt_fanin=np.array(list(gt_fanin.values()) or [0]),
                em_fanin=np.array(list(em_fanin.values()) or [0]))


def report(r, tau):
    np_, ng, ne, ns = r["n_pair"], r["n_gt"], r["n_emit"], r["n_samp"]
    tot_p, tot_g, tot_e = np_.sum(), ng.sum(), ne.sum()
    print(f"\n{r['n_img']:,} images ({r['n_img_pg']:,} with >=1 person and "
          f"a garment); {tot_p:,} person-garment pairs")
    print(f"sampler proposed {ns.sum():,}/{tot_p:,} = "
          f"{100*ns.sum()/max(tot_p,1):.2f}%  (992-budget => expect ~100%)")
    print(f"GT wearing {tot_g:,} ({100*tot_g/max(tot_p,1):.2f}%)   "
          f"model emits wearing {tot_e:,} ({100*tot_e/max(tot_p,1):.2f}%)   "
          f"ratio {tot_e/max(tot_g,1):.2f}x")

    print(f"\n{'IoU bin':<12}{'pairs':>10}{'GT rate':>10}{'MODEL rate':>12}"
          f"{'ratio':>8}{'mean wear s':>13}{'mean top s':>12}{'any-emit':>10}"
          f"{'z_pred':>10}{'z_pair':>9}")
    for k, nm in enumerate(BIN_NAMES):
        if np_[k] == 0:
            continue
        gr = ng[k] / np_[k]
        mr = ne[k] / np_[k]
        mw = r["s_wear"][k] / max(ns[k], 1)
        mt = r["s_top"][k] / max(ns[k], 1)
        ar = r["n_any"][k] / np_[k]
        zp_ = r["z_pred"][k] / max(ns[k], 1)
        za_ = r["z_pair"][k] / max(ns[k], 1)
        print(f"{nm:<12}{np_[k]:>10,}{100*gr:>9.2f}%{100*mr:>11.2f}%"
              f"{mr/max(gr,1e-9):>8.2f}{mw:>13.4f}{mt:>12.4f}{100*ar:>9.2f}%"
              f"{zp_:>10.3f}{za_:>9.3f}")
    zpd = r["z_pred"][0] / max(ns[0], 1) - r["z_pred"][-1] / max(ns[-1], 1)
    zad = r["z_pair"][0] / max(ns[0], 1) - r["z_pair"][-1] / max(ns[-1], 1)
    print(f"  disjoint - overlapping:  z_pred {zpd:+.3f}   z_pair {zad:+.3f}"
          f"   -> the term carrying the geometry is "
          f"{'z_pair (reweighting w may fix it without retraining)' if abs(zad) > abs(zpd) else 'z_pred (relatedness gate is NOT gating; needs training)'}")

    for nm, bag in (("DISJOINT (IoU=0)", r["say_disjoint"]),
                    ("OVERLAPPING", r["say_overlap"])):
        tot = sum(bag.values()) or 1
        top = sorted(bag.items(), key=lambda kv: -kv[1])[:10]
        print(f"\nwhat the model SAYS about {nm} person-garment pairs "
              f"({tot:,} emitted):")
        print("  " + "  ".join(f"{k} {100*v/tot:.1f}%" for k, v in top))

    print(f"\n--- FUNCTIONALITY (one garment, one wearer) at tau={tau} ---")
    for nm, f in (("GT", r["gt_fanin"]), ("MODEL", r["em_fanin"])):
        if f.sum() == 0:
            print(f"  {nm:<6}: none")
            continue
        print(f"  {nm:<6}: {len(f):,} garments worn; fan-in mean {f.mean():.3f} "
              f"max {f.max()}  ==1 {100*(f==1).mean():.2f}%  "
              f">=2 {int((f>=2).sum()):,} ({100*(f>=2).mean():.2f}%)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data_root", default="runs/packed/vg150")
    p.add_argument("--split", default="test")
    p.add_argument("--img_size", type=int, default=448)
    p.add_argument("--max_objects", type=int, default=32)
    p.add_argument("--geo_budget", type=int, default=992)
    p.add_argument("--final_budget", type=int, default=992)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--tau", type=float, default=0.56)
    p.add_argument("--pattern", default=r"wear|dressed in|has on|clothed")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--extra_predicates", default="",
                   help="comma-separated phrases APPENDED to the pack's "
                        "vocabulary. The deployment default vocabulary "
                        "(deploy/pipeline._default_predicates) has no "
                        "'to the left/right of', and megasg labels 63%% of its "
                        "annotated-but-disjoint person-garment pairs exactly "
                        "that — so under graph-constrained argmax the correct "
                        "answer may simply be unavailable. This flag tests "
                        "that: if disjoint-wearing collapses when the phrases "
                        "are added, the bug is vocabulary composition, not the "
                        "learned prior.")
    p.add_argument("--dump_pairs", default=None,
                   help="npz of (iou, z_pred_top, z_pair, gt_wearing, argmax_is_wearing) per person-garment pair, for offline (w, tau) sweeps.")
    p.add_argument("--out", default=None)
    a = p.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    model = build_model_from_ckpt(ck, "ema").to(dev).eval()
    model.sampler.geo_budget = a.geo_budget
    model.sampler.final_budget = a.final_budget
    # required=False: proxy checkpoints have no calibration.json — identity
    # contract, and every comparison is at matched volume, never absolute tau.
    contract = ScoreContract.for_checkpoint(a.checkpoint, required=False)
    print(f"contract: {contract.describe()}  tau={a.tau}")

    ds = RelationDataset(root=a.data_root, split=a.split, resolution=a.img_size,
                         max_objects=a.max_objects)
    names = list(ds.predicate_names)
    n_native = len(names)
    if a.extra_predicates:
        extra = [s.strip() for s in a.extra_predicates.split(",") if s.strip()]
        # APPEND, never reorder: GT relation ids index into the pack's own
        # vocabulary and must keep pointing at the same phrase.
        names += [e for e in extra if e not in names]
        print(f"vocabulary {n_native} -> {len(names)}: "
              f"+{names[n_native:]}")
    import re
    rx = re.compile(a.pattern, re.I)
    wear_ids = {i for i, n in enumerate(names) if rx.search(n)}
    print(f"wearing columns: {sorted(names[i] for i in wear_ids)}")
    ck_args = ck.get("args") or {}
    ck_args = ck_args if isinstance(ck_args, dict) else vars(ck_args)
    E = encode_texts_student(names, ck_args["text_student"],
                             templates=TRAIN_TEMPLATES, device=dev)
    model.vocab_head.set_vocabulary_matrix(names, E)
    model.reparameterize()

    sub = ds
    if a.limit:
        sub = torch.utils.data.Subset(ds, list(range(min(a.limit, len(ds)))))
    loader = DataLoader(sub, batch_size=a.batch_size, shuffle=False,
                        collate_fn=collate_fn, num_workers=a.num_workers,
                        pin_memory=True)

    pair_dump = [] if a.dump_pairs else None
    r = collect(model, loader, ds, dev, contract, a.tau, wear_ids, names,
                pair_dump=pair_dump)
    report(r, a.tau)

    if a.dump_pairs and pair_dump:
        arr = np.asarray(pair_dump, dtype=np.float64)
        os.makedirs(os.path.dirname(a.dump_pairs) or ".", exist_ok=True)
        np.savez_compressed(a.dump_pairs, iou=arr[:, 0], z_pred=arr[:, 1],
                            z_pair=arr[:, 2], gt=arr[:, 3].astype(np.int8),
                            is_wear=arr[:, 4].astype(np.int8))
        print(f"dumped {len(arr):,} pairs -> {a.dump_pairs}")

    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        json.dump({k: (v.tolist() if isinstance(v, np.ndarray) else v)
                   for k, v in r.items()} | {"tau": a.tau, "bins": BINS.tolist(),
                                             "bin_names": BIN_NAMES},
                  open(a.out, "w"), indent=2)
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
