"""Type-consistency of emitted `wearing` edges — the failure the user reported.

The geometry readout (e0_report.py) asks WHERE a wearing edge is placed. This
asks WHAT it connects: `wearing` has a hard selectional restriction — the object
must be a garment and the subject must be an animate wearer. A person-wearing-a-
person edge is nonsense no annotation ever supports, so its rate is a prior
violation that needs no threshold to judge.

Reported at a COUNT-MATCHED emission budget so an arm cannot look clean by
emitting less, and beside the SAME statistic measured on the pack's GT, which is
the only honest target: the model should match the data, not zero.

    python training/e3_type_consistency.py \
        runs/analysis/e3/dump_topk_vg150.npz runs/analysis/e3/dump_lse_vg150.npz \
        --labels topk lse --pack runs/packed/vg150/test
"""
from __future__ import annotations

import argparse
import json
import re

import numpy as np

WEAR_RX = re.compile(r"wear|dressed in|has on|clothed", re.I)
PERSON = re.compile(r"^(man|woman|person|people|boy|girl|child|kid|lady|guy|"
                    r"player|skier|surfer|men|women)s?$", re.I)
GARMENT = re.compile(r"^(shirt|jacket|coat|hat|cap|helmet|pant|jean|short|sock|"
                     r"shoe|sneaker|boot|glove|tie|scarf|dress|skirt|sweater|"
                     r"jersey|uniform|clothes|glass|glasses|hoodie|vest|jacket)s?$",
                     re.I)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("dumps", nargs="+")
    p.add_argument("--labels", nargs="+")
    p.add_argument("--pack", default="runs/packed/vg150/test")
    p.add_argument("--budget", choices=["global", "wearing"], default="global",
                   help="global = match TOTAL emitted edges to GT (shows how "
                        "much of its budget the model spends on wearing); "
                        "wearing = match the WEARING count to GT (isolates "
                        "type/fan-in quality from sheer over-emission)")
    a = p.parse_args()
    labels = a.labels or [f"arm{i}" for i in range(len(a.dumps))]

    meta = json.load(open(f"{a.pack}/meta.json"))
    cats = meta["categories"]
    is_p = np.array([bool(PERSON.match(c)) for c in cats])
    is_g = np.array([bool(GARMENT.match(c)) for c in cats])

    print(f"budget mode: {a.budget}")
    print(f"{'arm':<10}{'wear edges':>12}{'obj=garment':>13}{'obj=person':>12}"
          f"{'obj=other':>11}{'subj=person':>13}{'fan-in>=2':>11}")
    for path, lab in zip(a.dumps, labels):
        z = np.load(path, allow_pickle=False)
        names = [str(s) for s in z["pred_names"]]
        wear = np.array([bool(WEAR_RX.search(n)) for n in names])
        pc = z["pair_counts"].astype(np.int64)
        bc = z["box_counts"].astype(np.int64)
        gc = z["gt_counts"].astype(np.int64)
        p_off = np.concatenate([[0], np.cumsum(pc)])[:-1]
        b_off = np.concatenate([[0], np.cumsum(bc)])[:-1]
        g_off = np.concatenate([[0], np.cumsum(gc)])[:-1]
        img_of = np.repeat(np.arange(len(pc)), pc)

        zp = np.asarray(z["z_pred"], np.float32)
        best = zp.argmax(1)
        score = zp[np.arange(len(zp)), best] + z["z_pair"].astype(np.float32)
        gt = z["gt"].astype(np.int64)
        n_gt_wear = int(wear[gt[:, 2]].sum())
        if a.budget == "global":
            # Global count-matched: emit as many TOTAL edges as GT has, then
            # look at whichever ones came out wearing. This measures how much of
            # its graph budget the model CHOOSES to spend on wearing.
            budget = int(gc.sum())
            keep = np.argpartition(-score, budget)[:budget]
            keep = keep[wear[best[keep]]]
        else:
            # Wearing-matched: emit exactly as many WEARING edges as GT has, so
            # the type/fan-in rates are compared against GT at equal volume and
            # cannot be inflated by sheer over-emission.
            cand = np.flatnonzero(wear[best])
            k = min(n_gt_wear, len(cand))
            keep = cand[np.argpartition(-score[cand], k - 1)[:k]]

        gsub = b_off[img_of[keep]] + z["sub"][keep].astype(np.int64)
        gobj = b_off[img_of[keep]] + z["obj"][keep].astype(np.int64)
        cs = z["box_cats"][gsub].astype(np.int64)
        co = z["box_cats"][gobj].astype(np.int64)
        n = max(len(keep), 1)

        # fan-in: distinct subjects claiming the same object, per image
        fan = {}
        for k, s, o in zip(img_of[keep], z["sub"][keep], z["obj"][keep]):
            fan.setdefault((int(k), int(o)), set()).add(int(s))
        ge2 = sum(1 for v in fan.values() if len(v) >= 2) / max(len(fan), 1)

        print(f"{lab:<10}{len(keep):>12,}{100*is_g[co].mean():>12.1f}%"
              f"{100*is_p[co].mean():>11.1f}%"
              f"{100*(~is_g[co] & ~is_p[co]).mean():>10.1f}%"
              f"{100*is_p[cs].mean():>12.1f}%{100*ge2:>10.1f}%")

        if lab == labels[0]:   # GT reference, computed once from the same pack
            gimg = np.repeat(np.arange(len(gc)), gc)
            gw = wear[gt[:, 2]]
            gs = z["box_cats"][b_off[gimg[gw]] + gt[gw, 0]].astype(np.int64)
            go = z["box_cats"][b_off[gimg[gw]] + gt[gw, 1]].astype(np.int64)
            gfan = {}
            for k, s, o in zip(gimg[gw], gt[gw, 0], gt[gw, 1]):
                gfan.setdefault((int(k), int(o)), set()).add(int(s))
            gge2 = sum(1 for v in gfan.values() if len(v) >= 2) / max(len(gfan), 1)
            ref = (f"{'GT (target)':<10}{int(gw.sum()):>12,}"
                   f"{100*is_g[go].mean():>12.1f}%{100*is_p[go].mean():>11.1f}%"
                   f"{100*(~is_g[go] & ~is_p[go]).mean():>10.1f}%"
                   f"{100*is_p[gs].mean():>12.1f}%{100*gge2:>10.1f}%")
    print("-" * 82)
    print(ref)
    print("\nobj=person on a wearing edge is a selectional violation: no "
          "annotation supports it, so lower is strictly better.")


if __name__ == "__main__":
    main()
