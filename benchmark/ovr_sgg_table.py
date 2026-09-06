"""Assemble the two OV-SGG literature tables from runs/benchmark/ovr_sgg/*.json.

Our rows are READ from eval artifacts; literature rows are the only hand-entered
values and each carries its source string, per paper2's rule.

    python benchmark/ovr_sgg_table.py            # both tables + split stats
    python benchmark/ovr_sgg_table.py --json     # machine-readable

TABLE 1  OvR-SGG (Base+Novel / Novel relation split). VG150 test, SGDet,
         graph-constrained. Our detector is the VG150-trained yolov8m --
         protocol-matched, because OvR-SGG holds out RELATIONS only and keeps
         every object node, so the baselines' GroundingDINO is VG150-object-
         trained too.

TABLE 2  Caption-only pre-training: no VG relation annotations anywhere. Our
         detector is YOLO-World prompted with VG150's categories -- the fair
         analogue of their GroundingDINO. Using the VG150-trained yolov8m here
         would hand us object supervision no row in that table has.

CAVEAT that must travel with Table 1: for us the base/novel split is COSMETIC.
We train on no VG150 relation at all, base or novel, so nothing is "held out";
and RA-4M contains the novel predicate strings, so we are not compliant with the
OvR training restriction either. The compliant row needs the stripped-corpus
retrain. These numbers place us on the axis; they do not win the setting.
"""
from __future__ import annotations
import argparse, json, os

D = "runs/benchmark/ovr_sgg"

# --- literature rows (hand-entered; source string mandatory) -----------------
SRC_INOVA = ("OvSGTR rows: the authors' OWN README OvR-SGG table (github JosephZ/OvSGTR, "
             "read 2026-08-27) -- more complete than any third-party quote (it has R@20 and "
             "the MegaSG-pretrained rows). Other rows: INOVA arXiv:2502.03856 Tab.1 "
             "(= ACC arXiv:2511.05935 Tab.1, same paper renamed). NOTE: INOVA's table "
             "quotes only OvSGTR's NON-MegaSG rows; OvSGTR's MegaSG-pretrained variants "
             "(extra pretraining data) exceed INOVA's own numbers.")
SRC_CAP = ("INOVA arXiv:2502.03856 Tab.4 (= ACC arXiv:2511.05935 Tab.5)")

# name, backbone, all R@20/50/100, novel R@20/50/100
OVR_LIT = [
    ("IMP (CVPR'17)",      "—",      [None, None, 12.56], [None, None, 0.00]),
    ("MOTIFS (CVPR'18)",   "—",      [None, None, 15.41], [None, None, 0.00]),
    ("VCTree (CVPR'19)",   "—",      [None, None, 15.61], [None, None, 0.00]),
    ("TDE (CVPR'20)",      "—",      [None, None, 15.50], [None, None, 0.00]),
    ("VS3 (CVPR'23)",      "Swin-T", [None, 15.60, 17.30], [None, None, 0.00]),
    ("OvSGTR (ECCV'24)",   "Swin-T", [15.85, 20.50, 23.90], [10.17, 13.47, 16.20]),
    ("RAHP",               "Swin-T", [None, 20.50, 25.74], [None, 15.59, 19.92]),
    ("INOVA/ACC",          "Swin-T", [17.49, 23.22, 27.40], [12.90, 17.89, 21.70]),
    ("OvSGTR (ECCV'24)",   "Swin-B", [17.63, 22.90, 26.68], [12.09, 16.37, 19.73]),
    ("INOVA/ACC",          "Swin-B", [18.77, 24.81, 29.28], [14.72, 20.04, 24.66]),
    ("OvSGTR +MegaSG",     "Swin-T", [19.38, 25.40, 29.71], [12.23, 17.02, 21.15]),
    ("OvSGTR +MegaSG",     "Swin-B", [21.09, 27.92, 32.74], [16.59, 22.86, 27.73]),
]
CAP_LIT = [
    ("LSWS (CVPR'21)",            "—",            [None, None, 3.28]),
    ("MOTIFS + Li et al.",        "—",            [5.02, 6.40, 7.33]),
    ("Uniter + SGNLS (ECCV'20)",  "—",            [None, 5.80, 6.70]),
    ("Uniter + Li et al.",        "—",            [5.42, 6.74, 7.62]),
    ("VS3 (CVPR'23)",             "GLIP-L",       [5.59, 7.30, 8.62]),
    ("OvSGTR (ECCV'24)",          "GDINO Swin-T", [6.61, 8.92, 10.90]),
    ("OvSGTR (ECCV'24)",          "GDINO Swin-B", [6.88, 9.30, 11.48]),
    ("INOVA/ACC",                 "GDINO Swin-T", [7.86, 10.81, 13.31]),
    ("INOVA/ACC",                 "GDINO Swin-B", [8.28, 11.61, 14.33]),
]
OURS = [("zeroshot_S+", "RelateAnything ViT-S/16+ (zero-shot)"),
        ("official_S+", "RelateAnything ViT-S/16+ (released)")]


def f(x, nd=2):
    return "—" if x is None else f"{x:.{nd}f}"


def load(tag):
    p = os.path.join(D, f"{tag}.json")
    return json.load(open(p)) if os.path.exists(p) else None


def ours_row(d, proto, subset):
    """R@20/50/100 for a subset, image-macro (the SGG R@K convention: mean over
    images of per-image recall, which is what every row above reports)."""
    m = (d or {}).get("modes", {}).get(proto)
    if not m:
        return None
    if subset == "all":
        return [100 * m.get(f"R@{k}", 0.0) for k in (20, 50, 100)]
    return [100 * m.get(f"{subset}_R@{k}", 0.0) for k in (20, 50, 100)]


def split_stats():
    import benchmark.eval_ovsgtr_novel as M  # noqa
    meta = json.load(open("runs/packed/vg150/test/meta.json"))
    sup = meta["predicate_counts"]; tot = sum(sup.values())
    nov = M.VG150_NOVEL_PREDICATE
    mass = sum(sup[p] for p in nov)
    top3 = sorted(nov, key=lambda x: -sup[x])[:3]
    return dict(total=tot, novel_mass=mass, novel_frac=mass / tot,
                top3=[(p, sup[p] / mass) for p in top3],
                top3_frac=sum(sup[p] for p in top3) / mass)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--proto", default="detbox_strict",
                    choices=["detbox_strict", "detbox_lenient"],
                    help="strict = IoU>=0.5 AND detector label == GT label, which is "
                         "the SGDet convention the literature rows use")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()

    s = split_stats()
    out = {"split_composition": s, "proto": a.proto}

    print(f"\nOvSGTR OvR split composition on VG150 test "
          f"({s['total']:,} GT relations)")
    print(f"  novel carries {s['novel_mass']:,} = {100*s['novel_frac']:.1f}% of all GT; "
          f"{', '.join(f'{p} {100*v:.1f}%' for p, v in s['top3'])} "
          f"= {100*s['top3_frac']:.1f}% of the Novel evaluation")

    print(f"\n=== TABLE 1 — OvR-SGG, VG150 test, SGDet, graph-constrained "
          f"({a.proto.split('_')[1]}) ===")
    print(f"{'method':<38}{'backbone':<14}{'Base+Novel R@20/50/100':>26}"
          f"{'Novel R@20/50/100':>24}")
    print("-" * 102)
    for n, b, al, nv in OVR_LIT:
        print(f"{n:<38}{b:<14}{'/'.join(f(x) for x in al):>26}"
              f"{'/'.join(f(x) for x in nv):>24}")
    print("-" * 102)
    for tag, name in OURS:
        d = load(f"{tag}_yolov8m")
        al, nv = ours_row(d, a.proto, "all"), ours_row(d, a.proto, "novel")
        if al is None:
            print(f"{name:<38}{'ViT-S/16+':<14}{'(not yet run)':>26}"); continue
        print(f"{name:<38}{'ViT-S/16+':<14}{'/'.join(f(x) for x in al):>26}"
              f"{'/'.join(f(x) for x in nv):>24}")
        out.setdefault("table1", {})[tag] = {"all": al, "novel": nv}
        g = load(f"{tag}_gtbox")
        if g:
            gal, gnv = ours_row(g, "gtbox", "all"), ours_row(g, "gtbox", "novel")
            print(f"{'  ^ same model, GT boxes (upper bd)':<38}{'':<14}"
                  f"{'/'.join(f(x) for x in gal):>26}{'/'.join(f(x) for x in gnv):>24}")
            out["table1"][tag]["gtbox"] = {"all": gal, "novel": gnv}
    print(f"\n  literature rows: {SRC_INOVA}")
    print("  CAVEAT: for us the split is cosmetic — we train on NO VG150 relation, base "
          "or novel,\n  and RA-4M contains the novel strings, so we do not satisfy the "
          "OvR training restriction.")

    print(f"\n=== TABLE 2 — caption-only pre-training (no VG relation annotations), "
          f"VG150 SGDet ===")
    print(f"{'method':<38}{'detector':<24}{'R@20/50/100':>16}")
    print("-" * 78)
    for n, b, r in CAP_LIT:
        print(f"{n:<38}{b:<24}{'/'.join(f(x) for x in r):>16}")
    print("-" * 78)
    # The baselines' detector supervision in this table is NOT documented (the paper
    # does not say whether GroundingDINO is fine-tuned on VG150 objects before this
    # evaluation), so we BRACKET our row instead of pinning it: YOLO-World prompted
    # with VG150's categories = no VG object supervision (lower bracket), and the
    # VG150-trained yolov8m = full VG object supervision (upper bracket). The
    # baselines sit somewhere inside that interval. Do not quote one bracket alone.
    for tag, name in OURS:
        for det, lab in (("yoloworld_ov", "YOLO-World (no VG obj)"),
                         ("yolov8m", "yolov8m (VG150 obj)")):
            d = load(f"{tag}_{det}")
            al = ours_row(d, a.proto, "all")
            if al is None:
                print(f"{name:<38}{lab:<24}{'(not yet run)':>16}"); continue
            print(f"{name:<38}{lab:<24}{'/'.join(f(x) for x in al):>16}")
            out.setdefault("table2", {}).setdefault(tag, {})[det] = al
    print(f"\n  literature rows: {SRC_CAP}")
    print("  Our row is BRACKETED: the baselines' detector supervision is undocumented,")
    print("  so we report both no-VG-objects and VG150-objects detectors rather than")
    print("  quoting whichever flatters us. Read the interval, not an endpoint.")

    # --- Table 3: the column the literature never prints -----------------------
    print("\n=== TABLE 3 — the Novel split, per predicate (R@50, "
          f"{a.proto.split('_')[1]} SGDet) ===")
    print("Micro Novel R@K is 92.6% carried by on/of/in, so it is compatible with "
          "recovering\nnothing else. No OV-SGG paper reports this breakdown or a "
          "Novel mR@K.")
    shown = False
    for tag, name in OURS:
        d = load(f"{tag}_yolov8m")
        m = (d or {}).get("modes", {}).get(a.proto)
        pc = (m or {}).get("novel_per_class@50")
        if not pc:
            continue
        shown = True
        tot = sum(v["gt"] for v in pc.values())
        print(f"\n  {name}   (Novel mR@50 macro = "
              f"{100*m.get('novel_mR@50_macro', 0):.2f}, "
              f"micro R@50 = {100*m.get('novel_R@50_micro', 0):.2f})")
        print(f"    {'predicate':<15}{'GT':>8}{'% of novel':>12}{'recall@50':>11}")
        cum = 0.0
        for k, v in sorted(pc.items(), key=lambda x: -x[1]["gt"]):
            share = v["gt"] / tot if tot else 0.0
            cum += share
            print(f"    {k:<15}{v['gt']:>8,}{100*share:>11.1f}%{100*v['recall']:>10.1f}%"
                  + ("   <- 92.6% cumulative" if 0.92 < cum < 0.94 else ""))
    if not shown:
        print("  (not yet run)")

    if a.json:
        print("\n" + json.dumps(out, indent=2, default=float))


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    main()
