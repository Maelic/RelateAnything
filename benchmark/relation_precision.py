"""Per-relation oracle for A5: score EVERY asserted relation, not the graph as a whole.

Why this exists. The pairwise oracle (llm_judge.py) returned a clean but ambiguous
result: we lose whenever our graph is longer (win rate 0.104 / 0.158, both p<0.001)
and it is a coin flip at matched length (10 and 8 decisive comparisons, p~0.75). Two
readings fit that: our extra relations are FALSE, or the judge carries a verbosity
prior. A graph-level score -- absolute or pairwise -- cannot separate them, because a
verbosity prior moves it too. Scoring each relation independently can.

TRUTH IS NOT ENOUGH, hence two axes. A baseline that emits `on` for 85% of its
relations (OvSGTR: 1.7 distinct predicates per graph, modal share 0.85, measured
judge-free by graph_stats.py) would score near-perfect PRECISION while saying almost
nothing. So each relation gets:

  true: yes | no | unclear   -- is the claim actually true of this photograph
  info: 0..3                 -- how much it says BEYOND what the two object NAMES
                                 already imply. The "beyond the names" framing is the
                                 whole point: it prices the modal predicate at ~0,
                                 because `cup on table` is guessable from `cup` and
                                 `table` alone, while `person riding horse` is not.

The headline number is therefore not precision but USEFUL RELATIONS PER IMAGE --
P(true and info>=2) x rel/graph -- which a model can raise either by being more
accurate or by saying more that is worth saying, and cannot raise by padding with
true-but-vacuous claims. Precision and rel/graph are both reported alongside it, since
the composite hides which lever moved.

Judge: Qwen3-VL-8B-Instruct, served by vLLM. NEVER Gemma -- megasg supervision came
from gemma-4-26B-A4B-it, so a Gemma judge measures self-preference (the same rule
llm_judge.py enforces; note training/judge_predictions.py IS Gemma-only and calibrates
the soft matcher, a different question). Never a "Thinking" variant either: the verdict
is the last JSON object within --max_new tokens, and a long chain-of-thought starves it.

Controls. --control_frac of the judged relations are CORRUPTED before judging: the
predicate is replaced by a different one drawn from the same image's own graph. A judge
that cannot tell an intact claim from a corrupted one is not measuring anything, so the
run reports false_accept (P(true=yes | corrupted)) and refuses to certify above a
threshold. The corruption is a BOUND, not exact: a swapped predicate is sometimes
accidentally true, which inflates false_accept rather than hiding a bad judge.

Each relation is one request, image first, so vLLM's prefix cache reuses the vision
encode across every relation of the same image (~29 relations per image over both
systems -> the image is encoded once, not 29 times).

Usage:
    python benchmark/relation_precision.py \
        --system RelateAnything=runs/judge/relsgg_psg_test_yoloworld_train.npz \
        --system OvSGTR=runs/sgdet/ovdr_mega_psg_test_yoloworld.npz \
        --pack runs/packed/psg/test --n 200 \
        --out runs/judge/relation_precision_psg.json
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmark.llm_judge import load_npz, obj_labels, render, triplets   # noqa: E402

RUBRIC = """You are checking ONE claimed relationship in a photograph.

The numbered boxes drawn on the image identify the objects. This claim is about \
box {s} (the SUBJECT) and box {o} (the OBJECT).

CLAIM: {sname} #{s} — {pred} → {oname} #{o}

Answer two independent questions.

1. TRUE — is this claim actually true of THIS photograph? Look at those two boxes. \
Answer "yes" only if the stated relationship really holds between them, in that \
direction. Answer "no" if it is false, reversed, or describes some other pair. Answer \
"unclear" only if the boxes are too small, cropped or occluded to tell.

2. INFO — how much does the claim tell you about THIS photograph BEYOND what you could \
already guess from the two object names alone? Rate 0 to 3:
  0 = nothing. It is vacuous, or would be true of almost any photo containing these \
two objects, or states only that they are near each other.
  1 = little. Slightly more than the names imply, but still generic.
  2 = something. A specific fact about this scene you could not have guessed.
  3 = a lot. It names the salient interaction between these two objects.

A claim can be TRUE and still score 0 for INFO — that is the normal case for a generic \
predicate. Judge the two questions independently and do not let one drive the other.

Reply with STRICT JSON on the last line and nothing after it:
{{"true": "yes" | "no" | "unclear", "info": 0 | 1 | 2 | 3}}"""



# ---------------------------------------------------------------------------------
# INFO, JUDGED SEPARATELY. The joint rubric above asks for `true` and `info` in one
# reply, and MEASURABLY FAILS to keep them apart: over the 200-image run,
# P(info=0 | true=no) = 0.998 and P(info>=2 | true=yes) = 0.92 for every system. The
# judge answers "is it true" twice, which makes useful_rate identically 0.92 x
# precision and prices the modal predicate at whatever its truth rate is -- exactly
# the `on` problem the axis existed to catch (OvSGTR's `on` relations scored info 1.10
# against 0.85 for their own non-`on` ones, because `on` is easy to verify).
#
# Telling the judge to decide independently did not work; the joint rubric already
# says so in as many words. So decouple it STRUCTURALLY: informativeness as defined --
# "beyond what the two object NAMES imply" -- is a property of the TRIPLE, not of the
# photograph, so it can be asked without the image and without the truth question in
# context. It then cannot collapse onto truth, and identical triples are asked once
# and reused (`on` occurs ~800 times in a single run), which makes the pass nearly
# free despite being a second one.
RUBRIC_INFO = """How specific is this relationship description?

You are given only a relationship, with no picture. Rate how much the PREDICATE adds beyond what the two object names already imply.

RELATIONSHIP: {sname} — {pred} → {oname}

  0 = nothing. Vacuous, or would hold in almost any photo containing these two objects, or says only that they are near / positioned by each other.
  1 = little. Slightly more than the names imply, but still generic.
  2 = something. A specific relationship that does not follow from the names alone.
  3 = a lot. A salient, particular interaction between these two kinds of object.

Do NOT consider whether the relationship is likely to be true, or whether it makes sense. Rate ONLY its specificity. A nonsensical relationship can still be specific.

Reply with STRICT JSON on the last line and nothing after it:
{{"info": 0 | 1 | 2 | 3}}"""


# Same two questions, but the two boxes are identified by COLOUR rather than by a
# number the judge has to find among up to a hundred identical yellow ones. The plain
# rubric's grounding burden is the likeliest source of its marginal control score
# (false_accept 0.236 against a 0.25 refusal threshold), so this is the A/B.
RUBRIC_HL = """You are checking ONE claimed relationship in a photograph.

Two boxes are highlighted. The SUBJECT is in the THICK BLUE box labelled "SUBJECT". \
The OBJECT is in the THICK ORANGE box labelled "OBJECT". Thin grey boxes are other \
objects and are NOT part of this claim.

CLAIM: the {sname} in the BLUE box — {pred} → the {oname} in the ORANGE box

Answer two independent questions.

1. TRUE — is this claim actually true of THIS photograph? Look at those two \
highlighted boxes. Answer "yes" only if the stated relationship really holds between \
them, in that direction (blue does the action to orange). Answer "no" if it is false, \
reversed, or describes some other pair. Answer "unclear" only if the boxes are too \
small, cropped or occluded to tell.

2. INFO — how much does the claim tell you about THIS photograph BEYOND what you could \
already guess from the two object names alone? Rate 0 to 3:
  0 = nothing. It is vacuous, or would be true of almost any photo containing these \
two objects, or states only that they are near each other.
  1 = little. Slightly more than the names imply, but still generic.
  2 = something. A specific fact about this scene you could not have guessed.
  3 = a lot. It names the salient interaction between these two objects.

A claim can be TRUE and still score 0 for INFO — that is the normal case for a generic \
predicate. Judge the two questions independently and do not let one drive the other.

Reply with STRICT JSON on the last line and nothing after it:
{{"true": "yes" | "no" | "unclear", "info": 0 | 1 | 2 | 3}}"""

SUBJ_RGB, OBJ_RGB, OTHER_RGB = (0, 110, 255), (255, 140, 0), (150, 150, 150)


def render_pair(img_path, boxes, s_i, o_i, max_side=1024):
    """One image per RELATION: subject blue, object orange, everything else thin grey.

    The plain renderer draws every box the same yellow and tags it with a number, so
    the judge must locate two boxes among up to a hundred identical ones before it can
    reason about the claim at all. Colour moves that from a search problem to a
    perception one. Blue and orange are chosen because they are unambiguous to NAME in
    the prompt and rare as large flat regions in PSG photographs.
    """
    from PIL import Image, ImageDraw, ImageFont
    im = Image.open(img_path).convert("RGB")
    W, H = im.size
    sc = min(1.0, max_side / max(W, H))
    if sc < 1.0:
        im = im.resize((int(W * sc), int(H * sc)), Image.LANCZOS)
    dr = ImageDraw.Draw(im)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 16)
    except Exception:
        font = ImageFont.load_default()
    for i, bx in enumerate(boxes):                    # context, de-emphasised
        if i in (s_i, o_i):
            continue
        x0, y0, x1, y1 = [v * sc for v in bx]
        dr.rectangle([x0, y0, x1, y1], outline=OTHER_RGB, width=1)
    for i, rgb, tag in ((s_i, SUBJ_RGB, "SUBJECT"), (o_i, OBJ_RGB, "OBJECT")):
        if not (0 <= i < len(boxes)):
            continue
        x0, y0, x1, y1 = [v * sc for v in boxes[i]]
        dr.rectangle([x0, y0, x1, y1], outline=rgb, width=5)
        tw = dr.textlength(tag, font=font)
        ty = y0 if y0 > 20 else y1
        dr.rectangle([x0, ty - 20, x0 + tw + 10, ty], fill=rgb)
        dr.text((x0 + 5, ty - 19), tag, fill=(255, 255, 255), font=font)
    return im


def parse(txt):
    """Last JSON object wins: any preamble may itself quote braces."""
    for m in reversed(re.findall(r"\{[^{}]*\}", txt, re.S)):
        try:
            j = json.loads(m)
        except Exception:
            continue
        t = str(j.get("true", "")).strip().lower()
        if t not in ("yes", "no", "unclear"):
            continue
        try:
            info = int(j.get("info", -1))
        except Exception:
            info = -1
        if 0 <= info <= 3:
            return t, info
    return None, -1


# Calibration anchors for the text-only info axis. The truth axis has a control
# (corrupted claims); informativeness had none, so its scale rested on assertion. These
# are scored in the same pass and reported: VACUOUS should land near 0 and SPECIFIC
# near 3. If they do not separate, the axis is not measuring specificity and no
# `useful` number derived from it means anything -- which is exactly the failure the
# joint rubric hid.
INFO_ANCHORS_VACUOUS = [("cup", "on", "table"), ("sky", "above", "road"),
                        ("person", "near", "car"), ("tree", "in", "park")]
INFO_ANCHORS_SPECIFIC = [("man", "riding", "horse"), ("woman", "slicing", "cake"),
                         ("dog", "catching", "frisbee"), ("child", "hugging", "bear")]


# Predicates whose truth is nearly implied by any plausible arrangement of two
# objects. They matter because the control CORRUPTS a claim by swapping in another
# predicate from the same image, and a swap landing on one of these is often still
# TRUE -- so the judge accepting it is not an error. Measured on the 200-image run,
# corrupted claims are accepted at 0.42 when the replacement is generic and 0.17 when
# it is specific; `beside` alone is accepted 0.57 of the time.
#
# This is why depth changed the verdict: high-ranked relations are where generic
# predicates concentrate, so the control at rank<10 reads 0.276 and at rank>=10 reads
# 0.186 IN THE SAME RUN. The top-20 run certified only by averaging the two. The
# honest gate is therefore the specific-replacement subset, where accidental truth is
# implausible and acceptance really does measure the judge.
GENERIC_PREDICATES = {
    "on", "in", "beside", "near", "above", "under", "over", "next to", "behind",
    "in front of", "attached to", "part of", "with", "against", "around", "along",
    "at", "by", "on top of", "enclosing", "surrounding", "adjacent to",
}


def parse_info(txt):
    """Info-only reply. Truth is absent by design, so `parse` cannot be reused."""
    for m in reversed(re.findall(r"\{[^{}]*\}", txt, re.S)):
        try:
            j = json.loads(m)
        except Exception:
            continue
        try:
            info = int(j.get("info", -1))
        except Exception:
            continue
        if 0 <= info <= 3:
            return info
    return -1


def corrupt(tris, j, rng):
    """Replace relation j's predicate with a DIFFERENT one from this image's own graph.

    Drawn from the same graph so the corrupted claim stays in-distribution: a predicate
    the model itself was willing to assert here, just on the wrong pair. Returns None
    when the graph has no other predicate to offer (a predicate-uniform graph cannot
    carry this control, which is itself worth knowing -- those are exactly the graphs
    the modal-share statistic flags)."""
    others = sorted({p for _, p, _ in tris} - {tris[j][1]})
    if not others:
        return None
    s, _, o = tris[j]
    return (s, others[rng.randrange(len(others))], o)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--system", action="append", required=True,
                   metavar="NAME=PATH.npz", help="repeatable; judged independently")
    p.add_argument("--pack", required=True)
    p.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    p.add_argument("--n", type=int, default=200, help="images sampled")
    p.add_argument("--max_rel", type=int, default=20)
    p.add_argument("--rel_frac", type=float, default=0.7,
                   help="same relative keep-rule as llm_judge.py, so the graph judged "
                        "here is the graph that axis compared")
    p.add_argument("--control_frac", type=float, default=0.12)
    p.add_argument("--max_false_accept", type=float, default=0.25)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max_new", type=int, default=160)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max_side", type=int, default=1024)
    p.add_argument("--highlight", action="store_true",
                   help="render ONE image per relation with the subject box blue and "
                        "the object box orange, instead of one numbered image per "
                        "photo shared by all its relations. Removes the judge's "
                        "box-search burden; costs one vision encode per relation and "
                        "defeats prefix caching, so it runs slower.")
    p.add_argument("--info_mode", choices=["joint", "text"], default="joint",
                   help="joint: one reply carries both axes (the original, and shown "
                        "to collapse onto truth). text: informativeness is a second, "
                        "image-free pass over the distinct triples -- decoupled by "
                        "construction and deduplicated.")
    p.add_argument("--chunk_rows", type=int, default=25,
                   help="images per llm.chat call. Bounds how many rendered images are "
                        "resident at once -- with --highlight there is one per "
                        "relation, and materialising all of them at 1024px would need "
                        "tens of GB.")
    p.add_argument("--tp", type=int, default=1, help="tensor-parallel size")
    # Qwen3-VL-8B advertises a 262,144-token context, for which vLLM reserves 36 GiB of
    # KV cache and then refuses to start on a 48 GB card. One request here is one image
    # plus the rubric -- ~2k tokens -- so the full context is pure waste.
    p.add_argument("--max_model_len", type=int, default=8192)
    p.add_argument("--max_num_seqs", type=int, default=64)
    p.add_argument("--gpu_frac", type=float, default=0.90)
    p.add_argument("--out", required=True)
    a = p.parse_args()

    if "gemma" in a.model.lower():
        sys.exit("refusing: megasg supervision came from gemma-4-26B, so a Gemma judge "
                 "measures self-preference. Use a different family.")

    pack = Path(a.pack)
    meta = json.loads((pack / "meta.json").read_text())
    cat_names = list(meta["categories"])
    file_names = json.loads((pack / "file_names.json").read_text())
    img_dir = Path(meta["img_dir"])

    systems = []
    for spec in a.system:
        name, _, path = spec.partition("=")
        systems.append((name, load_npz(path)))
    rows = [{int(v): i for i, v in enumerate(d["image_index"])} for _, d in systems]
    common = sorted(set.intersection(*[set(r) for r in rows]))
    rng = random.Random(a.seed)
    chosen = sorted(rng.sample(common, min(a.n, len(common))))
    print(f"{len(common)} images in common; judging {len(chosen)}")

    # Object names come from the FIRST system for every system: the boxes are shared,
    # so a naming discrepancy would read as one model asserting nonsense (llm_judge.py
    # makes the same choice for the same reason).
    from vllm import LLM, SamplingParams
    llm = LLM(model=a.model, limit_mm_per_prompt={"image": 1},
              enable_prefix_caching=True, tensor_parallel_size=a.tp,
              max_model_len=a.max_model_len, max_num_seqs=a.max_num_seqs,
              gpu_memory_utilization=a.gpu_frac, trust_remote_code=False)
    sp = SamplingParams(temperature=a.temperature, max_tokens=a.max_new)

    def build(rows_chunk):
        """Requests for one chunk of images.

        With --highlight each relation needs its OWN rendered image, so the chunk is
        what keeps memory bounded: at 1024px, materialising all ~9.5k of them at once
        would need tens of GB. Without it, one numbered image per photo is shared by
        that photo's relations exactly as before.
        """
        reqs, recs = [], []
        for row in rows_chunk:
            i0 = rows[0][row]
            lab = obj_labels(systems[0][1], i0)
            bA, bB = int(systems[0][1]["box_ptr"][i0]), int(systems[0][1]["box_ptr"][i0 + 1])
            bx = systems[0][1]["boxes"][bA:bB]
            im = None if a.highlight else render(img_dir / file_names[row], bx, lab,
                                                 cat_names, max_side=a.max_side)
            for (name, d), rmap in zip(systems, rows):
                tris = triplets(d, rmap[row], a.max_rel, a.rel_frac, cat_names, labels=lab)
                for j, t in enumerate(tris):
                    is_ctl = rng.random() < a.control_frac
                    shown = corrupt(tris, j, rng) if is_ctl else t
                    if shown is None:
                        shown, is_ctl = t, False
                    sn, pred, on = shown
                    s_i = int(sn.rsplit("#", 1)[1]); o_i = int(on.rsplit("#", 1)[1])
                    if a.highlight:
                        img = render_pair(img_dir / file_names[row], bx, s_i, o_i,
                                          max_side=a.max_side)
                        text = RUBRIC_HL.format(sname=sn.rsplit("#", 1)[0],
                                                oname=on.rsplit("#", 1)[0], pred=pred)
                    else:
                        img = im
                        text = RUBRIC.format(s=s_i, o=o_i, sname=sn.rsplit("#", 1)[0],
                                             oname=on.rsplit("#", 1)[0], pred=pred)
                    reqs.append([{"role": "user", "content": [
                        {"type": "image_pil", "image_pil": img},
                        {"type": "text", "text": text}]}])
                    recs.append({"system": name, "row": row, "rank": j,
                                 "n_rel": len(tris), "control": is_ctl,
                                 "claim": list(shown), "original": list(t)})
        return reqs, recs

    recs = []
    chunks = [chosen[i:i + a.chunk_rows] for i in range(0, len(chosen), a.chunk_rows)]
    print(f"{len(chunks)} chunks of <= {a.chunk_rows} images"
          f"{' (one rendered image PER RELATION)' if a.highlight else ''}")
    for ci, ch in enumerate(chunks):
        reqs, rc = build(ch)
        outs = llm.chat(reqs, sp)
        for r, o in zip(rc, outs):
            r["true"], r["info"] = parse(o.outputs[0].text)
            r["raw"] = o.outputs[0].text[-300:]
        recs.extend(rc)
        del reqs                      # release this chunk's rendered images
        print(f"  chunk {ci+1}/{len(chunks)}: {len(rc)} judged, {len(recs)} total",
              flush=True)
    print(f"{len(recs)} relation judgements "
          f"({sum(r['control'] for r in recs)} of them corrupted controls)")

    if a.info_mode == "text":
        # One request per DISTINCT (subject name, predicate, object name), image-free.
        # The claim as JUDGED is what gets rated -- controls included -- so a corrupted
        # claim is scored on its own text and the control stays a control.
        def key(r):
            sn, pred, on = r["claim"]
            return (sn.rsplit("#", 1)[0].lower(), pred, on.rsplit("#", 1)[0].lower())

        anchors = INFO_ANCHORS_VACUOUS + INFO_ANCHORS_SPECIFIC
        uniq = sorted({key(r) for r in recs} | set(anchors))
        print(f"info pass: {len(uniq)} distinct triples from {len(recs)} relations "
              f"({len(recs) / max(1, len(uniq)):.1f}x reuse), image-free")
        ireqs = [[{"role": "user", "content": [{"type": "text", "text":
                  RUBRIC_INFO.format(sname=k[0], pred=k[1], oname=k[2])}]}]
                 for k in uniq]
        iouts = llm.chat(ireqs, SamplingParams(temperature=a.temperature, max_tokens=64))
        table = {k: parse_info(o.outputs[0].text) for k, o in zip(uniq, iouts)}
        miss = sum(v < 0 for v in table.values())
        if miss:
            print(f"  !! {miss}/{len(uniq)} triples returned no parsable info")
        vac = [table[k] for k in INFO_ANCHORS_VACUOUS if table.get(k, -1) >= 0]
        spec = [table[k] for k in INFO_ANCHORS_SPECIFIC if table.get(k, -1) >= 0]
        mv = sum(vac) / len(vac) if vac else float("nan")
        ms = sum(spec) / len(spec) if spec else float("nan")
        summary_info_anchor = {"vacuous_mean": mv, "specific_mean": ms,
                               "separated": bool(ms - mv >= 1.5),
                               "vacuous": {"/".join(k): table.get(k) for k in INFO_ANCHORS_VACUOUS},
                               "specific": {"/".join(k): table.get(k) for k in INFO_ANCHORS_SPECIFIC}}
        print(f"  info anchors: vacuous {mv:.2f}  specific {ms:.2f}  "
              f"separated={summary_info_anchor['separated']}")
        for lab, ks in (("vacuous", INFO_ANCHORS_VACUOUS), ("specific", INFO_ANCHORS_SPECIFIC)):
            print("    " + lab + ": " + ", ".join(f"{'/'.join(k)}={table.get(k)}" for k in ks))

        for r in recs:
            r["info_joint"] = r["info"]      # keep the collapsed value for comparison
            r["info"] = table[key(r)]
        # Did decoupling work? The joint axis had P(info=0|true=no) ~ 0.998.
        nz = [r for r in recs if r["true"] in ("yes", "no") and r["info"] >= 0]
        for t in ("yes", "no"):
            g = [r for r in nz if r["true"] == t]
            if g:
                print(f"  true={t:<4} n={len(g):<5} P(info=0)="
                      f"{sum(r['info'] == 0 for r in g) / len(g):.3f}  P(info>=2)="
                      f"{sum(r['info'] >= 2 for r in g) / len(g):.3f}")

    summary = {"model": a.model, "pack": str(pack), "n_images": len(chosen),
               "max_rel": a.max_rel, "rel_frac": a.rel_frac, "seed": a.seed,
               "info_mode": a.info_mode, "highlight": bool(a.highlight),
               **({"info_anchors": summary_info_anchor} if a.info_mode == "text" else {}),
               "n_judged": len(recs)}
    ctl = [r for r in recs if r["control"] and r["true"]]
    fa = sum(r["true"] == "yes" for r in ctl) / max(1, len(ctl))
    summary["control_n"] = len(ctl)
    summary["false_accept"] = fa

    # Gate on corruptions that cannot plausibly be accidentally true (see
    # GENERIC_PREDICATES). The pooled figure is kept because it is the honest UPPER
    # bound, but it conflates judge error with the corruption failing to corrupt, and
    # its value moves with graph depth for that reason alone.
    spec = [r for r in ctl if r["claim"][1] not in GENERIC_PREDICATES]
    gen = [r for r in ctl if r["claim"][1] in GENERIC_PREDICATES]
    fa_spec = sum(r["true"] == "yes" for r in spec) / max(1, len(spec))
    fa_gen = sum(r["true"] == "yes" for r in gen) / max(1, len(gen))
    summary["control_n_specific"] = len(spec)
    summary["false_accept_specific"] = fa_spec
    summary["control_n_generic"] = len(gen)
    summary["false_accept_generic"] = fa_gen
    summary["valid"] = bool(len(spec) >= 30 and fa_spec <= a.max_false_accept)

    per = {}
    for name, _ in systems:
        rs = [r for r in recs if r["system"] == name and not r["control"] and r["true"]]
        n = len(rs)
        if not n:
            continue
        imgs = {r["row"] for r in rs}
        yes = [r for r in rs if r["true"] == "yes"]
        useful = [r for r in yes if r["info"] >= 2]
        rel_per_img = n / max(1, len(imgs))
        by_rank = {}
        for lo, hi in ((0, 4), (5, 9), (10, 14), (15, 19)):
            b = [r for r in rs if lo <= r["rank"] <= hi]
            if b:
                by_rank[f"{lo}-{hi}"] = {
                    "n": len(b),
                    "precision": sum(r["true"] == "yes" for r in b) / len(b),
                    "mean_info": float(np.mean([r["info"] for r in b])),
                }
        per[name] = {
            "n_judged": n, "n_images": len(imgs), "rel_per_image": rel_per_img,
            "precision": len(yes) / n,
            "unclear_rate": sum(r["true"] == "unclear" for r in rs) / n,
            "mean_info": float(np.mean([r["info"] for r in rs])),
            "mean_info_given_true": float(np.mean([r["info"] for r in yes])) if yes else None,
            "info_hist": {str(k): sum(r["info"] == k for r in rs) for k in range(4)},
            "useful_rate": len(useful) / n,
            # The headline: true AND info>=2, per image. Padding with true-but-vacuous
            # claims cannot raise it, and neither can terse high-precision output.
            "useful_per_image": len(useful) / max(1, len(imgs)),
            "by_rank": by_rank,
        }
    summary["per_system"] = per

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"summary": summary, "records": recs}, open(a.out, "w"), indent=1)
    print("\n=== A5b per-relation oracle ===")
    print(f"  judge {a.model}   controls n={len(ctl)}  false_accept "
          f"{fa:.3f} pooled / {fa_spec:.3f} specific (n={len(spec)}) "
          f"/ {fa_gen:.3f} generic (n={len(gen)})   valid={summary['valid']}")
    print("  gate uses the SPECIFIC subset: a corruption landing on a generic "
          "predicate is often still true, so accepting it is not judge error.")
    print(f"\n  {'system':22s} {'rel/img':>8s} {'prec':>7s} {'info':>7s} "
          f"{'useful':>7s} {'USEFUL/IMG':>11s}")
    for name, v in per.items():
        print(f"  {name:22s} {v['rel_per_image']:8.2f} {v['precision']:7.3f} "
              f"{v['mean_info']:7.2f} {v['useful_rate']:7.3f} {v['useful_per_image']:11.2f}")
    if not summary["valid"]:
        print("\n!! CONTROLS FAILED — on SPECIFIC corruptions, which cannot be "
              "excused as accidentally true; "
              "the precision column above is not interpretable.")
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
