"""Pre-registered release gates for the model family (deploy/release_manifest.json).

Per official model (status pending/candidate/final/rejected), against its `-zeroshot`
sibling (status `unreleased`: kept as the gate's baseline, never published):
  G1  OVS-F1 >= canonical same size - 0.003        (seed noise floor)
  G2  OVS-mR >= canonical same size - 0.003
  G3  HICO F1@50 (graph-constrained, V2 pack) >= 0.30
  G4  A6 spatial (chance-corrected axis) >= 0.349   (the v1-pack negmask S+ level)
  Hedge: a `candidates` arm ships only if OVS-F1 > default + 0.003.
Reads runs/benchmark/ovs.json + ovs_f1.json (refresh with overall_score.py first) and the run dirs.
    python benchmark/release_gate.py
"""
from __future__ import annotations
import json, os
NOISE, HICO_MIN, A6_MIN = 0.003, 0.30, 0.349


def rows(path):
    o = json.load(open(path)); return {r.get("arm"): r for r in o["rows"] if isinstance(r, dict)}


def cell(run_dir, name, key):
    p = os.path.join(run_dir, name)
    if not os.path.exists(p): return None
    return json.load(open(p)).get("metrics", {}).get(key)


def main():
    man = json.load(open("deploy/release_manifest.json"))["models"]
    by_id = {m["model_id"]: m for m in man}
    mr, f1 = rows("runs/benchmark/ovs.json"), rows("runs/benchmark/ovs_f1.json")

    def look(m):
        arm = os.path.basename(m["run_dir"]); rm, rf = mr.get(arm), f1.get(arm)
        return {"arm": arm,
                "OVS_F1": rf.get("OVS") if rf else None, "OVS_mR": rm.get("OVS") if rm else None,
                "A1": (rm or {}).get("axes", {}).get("A1 transfer"), "A2": (rm or {}).get("axes", {}).get("A2 precision"),
                "A3": (rm or {}).get("axes", {}).get("A3 open-vocab"),
                "A4": (rm or {}).get("axes", {}).get("A4 detector"),
                "A6": (rm or {}).get("axes", {}).get("A6 spatial"),
                "HICO_F1": cell(m["run_dir"], "zeroshot_hicodet_test_gc.json", "F1@50"),
                "HICO_mR": cell(m["run_dir"], "zeroshot_hicodet_test_gc.json", "mR@50"),
                "fAP": cell(m["run_dir"], "hico_fap/haystack_sigmoid.json", "mfAP")}

    f = lambda v: "  --  " if v is None else f"{v:.4f}"
    # A3 is printed but is NOT in the composite (see overall_score.py); A4 is.
    # A5 is in the composite and no ladder arm has it -- it costs a judge run per
    # arm and was run for the released tower only -- so these OVS columns are over
    # FOUR axes. The head-to-head of the report's Table 3 is over five. Stated
    # here so the two are never read as the same number.
    print("OVS columns below are over A1, A2, A4, A6 (no arm here has an A5 run)")
    print(f"{'model':<30}{'OVS-F1':>8}{'OVS-mR':>8}{'A1':>7}{'A2':>7}{'A3':>7}{'A4':>7}{'A6':>7}{'HICO F1':>9}{'fAP':>7}")
    verdicts = {}
    for m in man:
        if m["status"] not in ("pending", "candidate", "final", "rehearsal", "rejected", "unreleased"): continue
        r = look(m)
        print(f"{m['model_id']:<30}{f(r['OVS_F1']):>8}{f(r['OVS_mR']):>8}{f(r['A1']):>7}{f(r['A2']):>7}{f(r['A3']):>7}{f(r['A4']):>7}{f(r['A6']):>7}{f(r['HICO_F1']):>9}{f(r['fAP']):>7}")
        if (m["status"] in ("pending", "candidate", "final", "rejected")
                and not m["model_id"].endswith("-zeroshot")):
            base_id = (m["model_id"].replace("-blr1e-4", "")) + "-zeroshot"
            b = look(by_id[base_id]) if base_id in by_id else {}
            if r["OVS_F1"] is None or r["HICO_F1"] is None or r["A6"] is None:
                verdicts[m["model_id"]] = "PENDING (evals incomplete)"; continue
            if b.get("OVS_F1") is None:
                # G1/G2 are relative to the sibling. Absent, they would compare
                # against 0 and pass on nothing, which is worse than not running.
                verdicts[m["model_id"]] = f"PENDING (no baseline row for {base_id})"; continue
            g = {"G1 OVS-F1": r["OVS_F1"] >= (b.get("OVS_F1") or 0) - NOISE,
                 "G2 OVS-mR": (r["OVS_mR"] or 0) >= (b.get("OVS_mR") or 0) - NOISE,
                 "G3 HICO F1>=.30": r["HICO_F1"] >= HICO_MIN,
                 "G4 A6>=.349": r["A6"] >= A6_MIN}
            verdicts[m["model_id"]] = ("PASS" if all(g.values()) else "FAIL -> ship zeroshot") + "  " + ", ".join(f"{k}:{'ok' if v else 'X'}" for k, v in g.items())
    # hedge
    for m in man:
        for tag, c in (m.get("candidates") or {}).items():
            cand = [x for x in man if x["run_dir"] == c["run_dir"]]
            if not cand: continue
            rd, rc = look(m), look(cand[0])
            if rd["OVS_F1"] is None or rc["OVS_F1"] is None:
                verdicts[f"{m['model_id']} hedge {tag}"] = "PENDING"; continue
            d = rc["OVS_F1"] - rd["OVS_F1"]
            verdicts[f"{m['model_id']} hedge {tag}"] = f"{'CANDIDATE WINS' if d > NOISE else 'default ships'} (delta OVS-F1 {d:+.4f} vs +{NOISE})"
    print()
    for k, v in verdicts.items(): print(f"  {k:<44} {v}")


if __name__ == "__main__":
    main()
