"""Markdown tables for the SAME-EVALUATOR REACT comparison (sgg_benchmark's SGDet evaluator on every row,
seeds 0,1,2 -> mean; sigma printed in a footnote). Reads runs/benchmark/react_compare/same_eval/*.json
(from job_react_same_eval.sh / tools/eval_predictions_offline.py) + latency from runs/benchmark/react_latency
and react_rows.json. Usage: python benchmark/react_same_eval_table.py [--det yolov8m]"""
import argparse, glob, json, os
SE = "runs/benchmark/react_compare/same_eval"
LABELS = {"zeroshot_S+": ("RelateAnything S+ zero-shot", "no"), "official_S": ("RelateAnything S (official)", "no"),
          "official_S+": ("RelateAnything S+ (official)", "no"), "official_B": ("RelateAnything B (official)", "no"),
          "ft_S+": ("RelateAnything S+ fine-tuned", "yes"), "react": ("REACT (YOLOv8m frozen)", "yes"),
          "damp": ("REACT++ DAMP", "yes"), "dampref": ("REACT++ DAMP replicate (c0_reference)", "yes")}
ORDER = list(LABELS)
TOWER = {"zeroshot_S+": "S+", "official_S": "S", "official_S+": "S+", "official_B": "B", "ft_S+": "S+"}
KEYS = [f"{m}@{k}" for m in ("R", "mR", "F1") for k in (20, 50, 100)]

def latency(lab, ds, det):
    if lab in TOWER:
        f = f"runs/benchmark/react_latency/{TOWER[lab]}_{ds}_{det}.json"
        if os.path.exists(f):
            j = json.load(open(f)); m = j.get("total_ms_mean") or j.get("mean_ms"); s = j.get("total_ms_std") or j.get("std_ms")
            return f"{m:.1f} ± {s:.1f}" if m is not None else "—"
        return "—"
    rows = json.load(open("runs/benchmark/react_compare/react_rows.json"))["rows"]
    for r in rows:
        if r["dataset"] == ds and r["det"] == det and r.get("label", "damp" if "DAMP" in r["model"] else "react") == lab:
            g = r.get("gpu", ""); return f"{r['latency_ms']:.1f} ± {r['latency_std']:.1f}" + ("" if g.startswith("A40") else f" ({g})")
    return "—"

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--det", default=""); a = ap.parse_args()
    cells = {}
    for f in glob.glob(f"{SE}/*.json"):
        b = os.path.basename(f)[:-5]; lab, ds, det = b.rsplit("_", 2); cells[(lab, ds, det)] = json.load(open(f))
    for ds in ("psg", "vg150", "indoorvg"):
        for det in ("yolov8m", "yolo12m"):
            if a.det and det != a.det: continue
            rows = [(lab, cells[(lab, ds, det)]) for lab in ORDER if (lab, ds, det) in cells]
            if not rows: continue
            print(f"\n### {ds} test · {det} detections · sgg_benchmark SGDet evaluator (graph-constrained, seeds 0,1,2)\n")
            print(f"| model | trained on {ds} | " + " | ".join(KEYS) + " | latency ms (A40, bs1) |"); print("|---|---|" + "---:|" * (len(KEYS) + 1))
            for lab, j in rows:
                name, tr = LABELS[lab]; name = name + (f" ({det.replace('yolo', 'YOLO')})" if lab == "damp" else "")
                print(f"| {name} | {tr} | " + " | ".join(f"{j['mean'][k]:.3f}" for k in KEYS) + f" | {latency(lab, ds, det)} |")
            sig = max(max(j["std"].values()) for _, j in rows); print(f"\nmax seed σ over the table: {sig:.4f}; n_images = {rows[0][1]['n_images']}")

if __name__ == "__main__":
    main()
