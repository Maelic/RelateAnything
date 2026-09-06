"""Score an OvSGTR interchange file on SpatialSense — the A6 baseline row.

SpatialSense scores (subject box, predicate, object box) cells with verified
TRUE/FALSE labels; the test split is balanced, so chance is 0.5 and a frequency
prior buys nothing. Our own model is scored by benchmark/eval_spatialsense.py; this
scores the baseline's interchange record (run_ovsgtr_pack.py --boxes gt
--caption_scope image) with the SAME metric module, relsgg/spatialsense_metrics.py,
so both rows of the table come from one implementation.

Cell semantics match ours exactly: a pair the model did not score gets 0.0 (what a
deployed system emits), and `coverage` records how often that happened. OvSGTR's
graph_infer enumerates every ordered pair, so its coverage should be ~1.0.

The global threshold for acc@valid_tau is fitted on the VALID interchange, as our
protocol does; pass --pred_valid for it, or the accuracy-at-threshold rows are
omitted and only the threshold-free AUC / AP are written.

Runs in the RELSGG venv (numpy + scikit-learn only).

    python benchmark/ovsgtr/eval_spatialsense_interchange.py \\
        --pred_test  runs/ovsgtr/ovdr_mega_spatialsense_test_gtbox.npz \\
        --pred_valid runs/ovsgtr/ovdr_mega_spatialsense_valid_gtbox.npz \\
        --out runs/ovsgtr/ovdr_mega_spatialsense_test.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from relsgg.eval.spatialsense import (best_threshold, summarise,  # noqa: E402
                                         print_summary)


def load_cells(cells_json: str, pred_names):
    side = json.load(open(cells_json))
    if list(side["predicates"]) != list(pred_names):
        raise SystemExit(f"cells predicate list != prediction vocabulary:\n"
                         f"  cells: {side['predicates']}\n  pred: {list(pred_names)}")
    # Keyed by the converter's image id, which IS the pack row index (asserted
    # round-trip in convert_spatialsense_test.py; eval_spatialsense.py relies on it too).
    return {int(k): [(int(s), int(o), int(p), int(l)) for s, o, p, l in v]
            for k, v in side["by_image_id"].items()}


def score_interchange(pred_npz: str, pack: str, cells_json: str):
    """Return (scores, labels, preds, coverage, info) for one split."""
    z = np.load(pred_npz, allow_pickle=False)
    d = {k: z[k] for k in z.files}                 # NpzFile re-inflates per access
    info = json.loads(str(d["meta"][0]))
    if info.get("box_source") != "gt":
        raise SystemExit("SpatialSense cells index GT boxes; requires --boxes gt")
    pred_names = [str(x) for x in d["predicates"]]
    meta = json.loads((Path(pack) / "meta.json").read_text())
    if list(meta["predicates"]) != pred_names:
        raise SystemExit("prediction vocabulary != pack; refusing to score")
    if info.get("score_semantics", "sigmoid") != "sigmoid":
        print("!! score_semantics is softmax: within-pair normalised probabilities are "
              "not truth scores; AUC across cells is not comparable to ours")
    bg = int(info.get("bg_column", 0))
    cells = load_cells(cells_json, pred_names)

    pair_ptr, idxs = d["pair_ptr"], d["image_index"]
    scores, labels, preds = [], [], []
    n_cells = n_cov = 0
    for i in range(len(idxs)):
        row = int(idxs[i])
        want = cells.get(row)
        if not want:
            continue
        a, b = int(pair_ptr[i]), int(pair_ptr[i + 1])
        pairs = d["pairs"][a:b]
        prob = np.delete(d["rel_scores"][a:b].astype(np.float32), bg, axis=1)
        pair_row = {}
        for k, (s_, o_) in enumerate(pairs.tolist()):
            pair_row.setdefault((s_, o_), k)
        for s_, o_, p_, lab in want:
            n_cells += 1
            k = pair_row.get((s_, o_))
            if k is None:
                scores.append(0.0)
            else:
                n_cov += 1
                scores.append(float(prob[k, p_]))
            labels.append(lab)
            preds.append(p_)
    missing_rows = set(cells) - set(int(x) for x in idxs)
    if missing_rows:
        raise SystemExit(f"{len(missing_rows)} images with cells are absent from the "
                         f"interchange (sharded run not merged?)")
    return (np.asarray(scores), np.asarray(labels), np.asarray(preds),
            n_cov / max(n_cells, 1), info, pred_names)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pred_test", required=True)
    p.add_argument("--pred_valid", default="",
                   help="valid-split interchange; fits the global threshold as ours does")
    p.add_argument("--test_pack", default="runs/packed/spatialsense_test/test")
    p.add_argument("--valid_pack", default="runs/packed/spatialsense_valid/test")
    p.add_argument("--test_cells", default="runs/datamix/spatialsense_test_cells.json")
    p.add_argument("--valid_cells", default="runs/datamix/spatialsense_valid_cells.json")
    p.add_argument("--out", required=True)
    a = p.parse_args()

    if a.pred_valid:
        vs, vl, _, vcov, _, _ = score_interchange(a.pred_valid, a.valid_pack, a.valid_cells)
        tau, vacc = best_threshold(vs, vl)
        print(f"[valid] {len(vs):,} cells  coverage {vcov:.4f}  "
              f"threshold={tau:.4f} (acc {vacc:.4f} on valid)")
    else:
        tau = 0.5
        print("[valid] no valid interchange given: acc@valid_tau uses tau=0.5")

    s, l, pr, cov, info, names = score_interchange(a.pred_test, a.test_pack, a.test_cells)
    print(f"[test ] {len(s):,} cells  coverage {cov:.4f}  "
          f"balance {int(l.sum())} true / {int((1 - l).sum())} false")
    # OvSGTR has no relatedness term; the field is kept so the two JSONs share keys.
    res = summarise(s, l, pr, names, tau, cov, use_pair_logits=False)
    print_summary(res)
    res["model"] = "OvSGTR"
    res["source"] = info
    res["valid_threshold_fitted"] = bool(a.pred_valid)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=2)
    print(f"\nsaved → {a.out}")


if __name__ == "__main__":
    main()
