"""E0 readout: wearing-family geometry behaviour of a checkpoint, from a
dump_pair_scores.py npz — CPU only, no model, no GPU.

Replaces the GPU wearing probe for uncalibrated (proxy) checkpoints: with no
calibration.json there is no meaningful absolute tau, so every emission
number here is at COUNT-MATCHED VOLUME — emit exactly as many wearing edges
as GT annotates (the same criterion that chose the shipped tau,
) — plus 2x and 4x that volume for robustness.
Volume-free readouts (z decomposition by IoU bin, GT rates) are unchanged.

    python training/e0_report.py runs/analysis/e0/dump_base_vg150.npz \
        runs/analysis/e0/dump_bg005_vg150.npz
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict

import numpy as np

sys.path.insert(0, __file__.rsplit("/", 2)[0])
from research.probe_wearing_conditional import (# noqa: E402
    BINS, BIN_NAMES, GARMENT, PERSON)

WEAR_RX = re.compile(r"wear|dressed in|has on|clothed", re.I)


def cxcywh_to_xyxy(b):
    cx, cy, w, h = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    return np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)


def pair_iou(sb, ob):
    ix1 = np.maximum(sb[:, 0], ob[:, 0]); iy1 = np.maximum(sb[:, 1], ob[:, 1])
    ix2 = np.minimum(sb[:, 2], ob[:, 2]); iy2 = np.minimum(sb[:, 3], ob[:, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    sa = np.clip((sb[:, 2] - sb[:, 0]) * (sb[:, 3] - sb[:, 1]), 1e-9, None)
    oa = np.clip((ob[:, 2] - ob[:, 0]) * (ob[:, 3] - ob[:, 1]), 1e-9, None)
    return inter / (sa + oa - inter)


def analyse(path):
    z = np.load(path, allow_pickle=False)
    names = [str(s) for s in z["pred_names"]]
    # VG150 category names are not in the dump; box_cats index the pack's
    # category list, which the person/garment regexes need. Load it.
    cats = json.load(open("runs/packed/vg150/test/meta.json"))["categories"]
    is_person = np.array([bool(PERSON.match(c)) for c in cats])
    is_garment = np.array([bool(GARMENT.match(c)) for c in cats])
    wear_ids = np.array([i for i, n in enumerate(names) if WEAR_RX.search(n)])

    pc = z["pair_counts"]; bc = z["box_counts"]; gc = z["gt_counts"]
    img_of = np.repeat(np.arange(len(pc)), pc)
    box_off = np.concatenate([[0], np.cumsum(bc)])[:-1]
    gt_off = np.concatenate([[0], np.cumsum(gc)])[:-1]
    boxes = cxcywh_to_xyxy(z["boxes"].astype(np.float64))
    box_cats = z["box_cats"]
    sub_g = box_off[img_of] + z["sub"].astype(np.int64)
    obj_g = box_off[img_of] + z["obj"].astype(np.int64)

    # person-garment pair mask over ALL sampled pairs
    pg = is_person[box_cats[sub_g]] & is_garment[box_cats[obj_g]]
    iou = pair_iou(boxes[sub_g[pg]], boxes[obj_g[pg]])
    kbin = np.digitize(iou, BINS) - 1

    zpred = z["z_pred"].astype(np.float32)          # [P, V]
    zpair = z["z_pair"].astype(np.float32)
    top_i = zpred.argmax(axis=1)
    top_z = zpred[np.arange(len(top_i)), top_i]
    fused = top_z + zpair

    # GT wearing cells, keyed (img, sub, obj) in local indices
    gt = z["gt"]; gt_img = np.repeat(np.arange(len(gc)), gc)
    gt_wear = set()
    m = np.isin(gt[:, 2], wear_ids)
    for i, s, o in zip(gt_img[m], gt[m, 0], gt[m, 1]):
        gt_wear.add((int(i), int(s), int(o)))

    pg_idx = np.flatnonzero(pg)
    pg_img = img_of[pg_idx]
    pg_sub = z["sub"][pg_idx]; pg_obj = z["obj"][pg_idx]
    pg_gt = np.array([(int(i), int(s), int(o)) in gt_wear
                      for i, s, o in zip(pg_img, pg_sub, pg_obj)])
    n_gt = int(pg_gt.sum())
    disj = iou <= 0.0

    out = {"path": path, "n_pairs": len(pg_idx), "n_gt": n_gt}
    print(f"\n{'='*74}\n{path}")
    print(f"{len(pc)} imgs | {len(img_of):,} sampled pairs | "
          f"{len(pg_idx):,} person-garment | GT wearing {n_gt:,} "
          f"({100*n_gt/max(len(pg_idx),1):.2f}%) | disjoint "
          f"{100*disj.mean():.1f}% of pairs, GT rate on disjoint "
          f"{100*pg_gt[disj].mean():.3f}%")

    # ---- volume-free: z decomposition by IoU bin ------------------------
    print(f"\n{'IoU bin':<12}{'pairs':>9}{'GT rate':>9}{'z_pred':>9}{'z_pair':>9}")
    zp_pg = top_z[pg_idx]; za_pg = zpair[pg_idx]
    for k, nm in enumerate(BIN_NAMES):
        s = kbin == k
        if not s.any():
            continue
        print(f"{nm:<12}{int(s.sum()):>9,}{100*pg_gt[s].mean():>8.2f}%"
              f"{zp_pg[s].mean():>9.3f}{za_pg[s].mean():>9.3f}")
    lo, hi = kbin == 0, kbin == len(BIN_NAMES) - 1
    gap_p = zp_pg[lo].mean() - zp_pg[hi].mean()
    gap_a = za_pg[lo].mean() - za_pg[hi].mean()
    print(f"disjoint - overlap gap:  z_pred {gap_p:+.3f}   z_pair {gap_a:+.3f}")
    out["zpred_gap"] = float(gap_p); out["zpair_gap"] = float(gap_a)

    # ---- count-matched emission ----------------------------------------
    is_wear_top = np.isin(top_i[pg_idx], wear_ids)
    cand = np.flatnonzero(is_wear_top)
    order = cand[np.argsort(-fused[pg_idx][cand])]
    print(f"\nwearing-argmax candidates: {len(cand):,} "
          f"({100*len(cand)/max(len(pg_idx),1):.1f}% of person-garment pairs)")
    for mult in (1, 2, 4):
        M = min(mult * n_gt, len(order))
        sel = order[:M]
        d = disj[sel]; g = pg_gt[sel]
        fanin = Counter((int(pg_img[j]), int(pg_obj[j])) for j in sel)
        fi = np.array(list(fanin.values()))
        print(f"  @{mult}x GT volume ({M:,} emitted): disjoint "
              f"{int(d.sum()):,} ({100*d.mean():.2f}% of emitted, "
              f"{100*d.sum()/max(disj.sum(),1):.2f}% of disjoint population) | "
              f"recall {100*g.sum()/max(n_gt,1):.2f}% | "
              f"fan-in>=2 {100*(fi>=2).mean():.1f}%")
        if mult == 1:
            out.update(disj_share=float(d.mean()),
                       disj_rate=float(d.sum() / max(disj.sum(), 1)),
                       recall=float(g.sum() / max(n_gt, 1)),
                       fanin_ge2=float((fi >= 2).mean()))

    # what disjoint person-garment pairs get called, at 1x volume
    sel = set(order[:min(n_gt, len(order))].tolist())
    say = Counter()
    all_top = np.argsort(-(top_z[pg_idx] + za_pg))   # rank ALL pg pairs
    picked = [j for j in all_top[:min(n_gt * 3, len(all_top))] if disj[j]]
    for j in picked[:5000]:
        say[names[top_i[pg_idx[j]]]] += 1
    tot = sum(say.values()) or 1
    print("  top-ranked DISJOINT pairs get called: " + "  ".join(
        f"{k} {100*v/tot:.0f}%" for k, v in say.most_common(6)))
    # Present only on dumps written after provenance was added; older dumps
    # fall through to the conservative caveat in main().
    if "ckpt_args" in z.files:
        try:
            out["ckpt_args"] = json.loads(str(z["ckpt_args"]))
        except (ValueError, TypeError):
            pass
    return out


def main():
    rows = [analyse(p) for p in sys.argv[1:]]
    if len(rows) > 1:
        print(f"\n{'='*74}\nTWIN COMPARISON (count-matched volume)")
        print(f"{'dump':<28}{'zpred gap':>10}{'zpair gap':>10}{'disj%emit':>11}"
              f"{'disj rate':>10}{'recall':>9}{'fanin>=2':>10}")
        for r in rows:
            nm = r["path"].split("/")[-1].replace("dump_", "").replace(
                "_vg150.npz", "")
            print(f"{nm:<28}{r['zpred_gap']:>+10.3f}{r['zpair_gap']:>+10.3f}"
                  f"{100*r['disj_share']:>10.2f}%{100*r['disj_rate']:>9.2f}%"
                  f"{100*r['recall']:>8.2f}%{100*r['fanin_ge2']:>9.1f}%")
        # The caveat is derived from the two runs' own arguments, so it is
        # printed for a multi-factor pair and stays quiet for a clean
        # single-variable ablation.
        cfgs = []
        for r in rows:
            ck = r.get("ckpt_args")
            cfgs.append(ck if isinstance(ck, dict) else None)
        if len(rows) == 2 and all(cfgs):
            diff = sorted(k for k in set(cfgs[0]) | set(cfgs[1])
                          if json.dumps(cfgs[0].get(k), sort_keys=True, default=str)
                          != json.dumps(cfgs[1].get(k), sort_keys=True, default=str))
            # output_dir always differs and carries no experimental meaning.
            diff = [k for k in diff if k not in ("output_dir", "run_name")]
            if len(diff) == 1:
                print(f"\nCLEAN ABLATION: the arms differ in {diff[0]} only.")
            else:
                print(f"\nCAVEAT: arms differ in {len(diff)} args "
                      f"({', '.join(diff[:8])}{'...' if len(diff) > 8 else ''})"
                      " — the reading is directional, not an ablation.")
        else:
            print("\nCAVEAT: arm configs unavailable in these dumps; confirm by "
                  "hand what differs before reading this as an ablation.")


if __name__ == "__main__":
    main()
