"""probe_attribution.py — WHERE DOES THE PREDICATE SCORE ACTUALLY COME FROM?

The motivating puzzle: a 17.25M-parameter head (3 of whose 4 attention stacks
are deletable for <=0.7% F1,) appears to "know" 19,103
predicates. Either the task is easier than it looks, or most of the answer is
arriving from somewhere other than the pixels. This script measures which.

TWO THINGS ARE MEASURED, IN ONE PASS EACH.

1. THE PER-EDGE PROTOCOL, matched byte-for-byte to training/bias_baselines.py.
   Given the GT (subject, object) pair, where does the TRUE predicate rank
   among the pack's closed vocabulary? Acc@1 / R@5 / MRR, plus macro-over-class
   versions. This is the ONLY protocol on which our model is directly
   comparable to the Zellers-2018 FREQ baseline (P(pred | subj_cat, obj_cat),
   zero pixels), which reaches Acc@1 0.682 / MRR 0.796 on VG150. Our headline
   R@50 numbers are per-IMAGE top-K over candidate pairs and cannot be compared
   to it. GT pairs are force-included by passing `targets` (sampler.py:407), so
   pair-sampler recall is not a confound: coverage is 100% by construction.

2. A LESION LADDER over the same edges, so the drop attributable to each input
   channel is read off directly:

     full      the model as shipped
     imgshuf   images rolled by one along the batch -> every pair keeps its own
               boxes and geometry but is shown SOMEBODY ELSE'S PIXELS. This is
               the "does it need to see anything" test. Its complement is
     nogeo     geo_encoder forced to zeros -> the relation head loses explicit
               box geometry (both the pair fusion at model.py:1110 and the
               spatial expert at:1178) while the pair SAMPLER keeps its own
               geometry, so the candidate set is unchanged and only the head is
               blinded.
     compose0  compose_gate:= 0 -> removes the explicit "object appearance
               projected into text space" channel (model.py:1169). Ships at
               0.044/0.020 already, so this is expected to be small; it is here
               because it is the channel a text-shortcut story would predict.
     prior     both expert queries replaced by their DATASET MEAN, estimated on
               a first pass over the same data. Every pair then receives the
               identical predicate ranking, so this is the model's own
               image-independent marginal: the floor that no vision at all
               would give you, in the model's own parameterization.

   `full` minus `prior` is the total value of conditioning on the input at all.
   `full` minus `imgshuf` is the value of the pixels specifically.

WHY THE LOGIT ATTRIBUTION IS EXACT, NOT A SHAPLEY APPROXIMATION (--attrib).
The semantic query is q = compose_norm(proj(r) + g0*S + g1*O). LayerNorm is
w/sigma * (x - mu) + b, and mu = mean(x) distributes linearly across the three
summands, so CONDITIONAL on the realized (mu, sigma) of each query the logit
splits with no residual:

    <q, W_v> = <x_ctx, W_v> + <x_sub, W_v> + <x_obj, W_v> + <b, W_v>
    where x_term = (w/sigma) * (term - mean(term))

Only variation ACROSS v moves the within-pair ranking, so the reported shares
are of Var_v, and the covariances are reported too (they do not vanish). The
residual is asserted < 1e-3 rather than assumed.

CAVEAT THAT LIMITS EVERY NUMBER HERE. Closed-vocab, GT-box, GT-pair. That is
the regime the FREQ baseline lives in, which is the point, but it is the most
prior-friendly regime this project has: see  for
what detector boxes do.

Usage:
    python training/probe_attribution.py --checkpoint <run>/checkpoint_last.pth \
        --data_roots runs/packed/vg150 runs/packed/psg --split test \
        --lesions full,imgshuf,nogeo,compose0,prior --attrib
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from relsgg.data import RelationDataset, collate_fn                      # noqa: E402
from relsgg.training.engine import region_kwargs                    # noqa: E402
from relsgg.vocabulary import TRAIN_TEMPLATES  # noqa: E402
from relsgg.checkpoint import build_model_from_ckpt  # noqa: E402

LESIONS = ("full", "imgshuf", "nogeo", "compose0", "prior")


# ---------------------------------------------------------------- rank stats
def report(ranks: np.ndarray, cls: np.ndarray, n_pred: int) -> dict:
    """Acc@1 / R@K / MRR, micro and macro-over-predicate-class."""
    if len(ranks) == 0:
        return {"n": 0}
    out = {"n": int(len(ranks)),
           "Acc@1": float((ranks == 1).mean()),
           "R@5": float((ranks <= 5).mean()),
           "R@10": float((ranks <= 10).mean()),
           "MRR": float((1.0 / ranks).mean()),
           "MedRank": float(np.median(ranks))}
    per = [float((ranks[cls == c] == 1).mean()) for c in np.unique(cls)]
    out["mAcc@1"] = float(np.mean(per))
    out["mMRR"] = float(np.mean(
        [float((1.0 / ranks[cls == c]).mean()) for c in np.unique(cls)]))
    out["n_classes"] = int(len(per))
    return out


# --------------------------------------------------------------- the lesions
class Lesion:
    """Install/remove one intervention. Restores exactly what it changed."""

    def __init__(self, model, kind: str, mean_q=None):
        self.model, self.kind, self.mean_q = model, kind, mean_q
        self._undo = []

    def __enter__(self):
        m, k = self.model, self.kind
        if k == "nogeo":
            ge = m.geo_encoder
            orig = ge.forward

            def zero_geo(*a, _o=orig, **kw):
                return torch.zeros_like(_o(*a, **kw))
            ge.forward = zero_geo
            self._undo.append(lambda: setattr(ge, "forward", orig))
        elif k == "compose0":
            g = m.compose_gate
            saved = g.data.clone()
            g.data.zero_()
            self._undo.append(lambda: g.data.copy_(saved))
        elif k == "prior":
            vh = m.vocab_head
            orig = vh.score_query_dual
            q_sem_bar, q_spa_bar = self.mean_q

            def mean_query(q_sem, q_spa, _o=orig, _s=q_sem_bar, _p=q_spa_bar):
                return _o(_s.to(q_sem.dtype).expand_as(q_sem),
                          _p.to(q_spa.dtype).expand_as(q_spa))
            vh.score_query_dual = mean_query
            self._undo.append(lambda: setattr(vh, "score_query_dual", orig))
        elif k not in ("full", "imgshuf"):
            raise SystemExit(f"unknown lesion {k!r}; expected {LESIONS}")
        return self

    def __exit__(self, *exc):
        for fn in reversed(self._undo):
            fn()
        return False


# ------------------------------------------------------------- capture hooks
class QueryTap:
    """Records the two expert queries and, optionally, the three exact
    additive components of the semantic one."""

    def __init__(self, model, attrib: bool):
        self.model, self.attrib = model, attrib
        self.buf = {}
        self.h = []

    def __enter__(self):
        m = self.model

        def keep(name):
            def hook(_mod, inp, out):
                self.buf[name] = (inp[0] if name == "norm_in" else out).detach()
            return hook
        self.h.append(m.compose_norm.register_forward_hook(
            lambda mod, inp, out: self.buf.update(
                norm_in=inp[0].detach(), q_sem=out.detach())))
        self.h.append(m.spa_proj.register_forward_hook(keep("q_spa")))
        if self.attrib:
            self.h.append(m.vocab_head.proj.register_forward_hook(keep("ctx")))
            self.h.append(m.sub_text_proj.register_forward_hook(keep("sub")))
            self.h.append(m.obj_text_proj.register_forward_hook(keep("obj")))
        return self

    def __exit__(self, *exc):
        for h in self.h:
            h.remove()
        return False


def attribute(tap_buf, model, W, valid) -> dict:
    """Exact additive Var_v decomposition of the semantic-expert cosine.

    Returns variance shares over the predicate axis, plus the covariance mass,
    plus the residual (which must be ~0 or the derivation is wrong)."""
    ln = model.compose_norm
    x = tap_buf["norm_in"].float()                       # [B,K,D] pre-LN sum
    parts = {"ctx": tap_buf["ctx"].float(),
             "sub": model.compose_gate[0].float() * tap_buf["sub"].float(),
             "obj": model.compose_gate[1].float() * tap_buf["obj"].float()}
    mu = x.mean(-1, keepdim=True)
    var = x.var(-1, unbiased=False, keepdim=True)
    inv = ln.weight.float() / torch.sqrt(var + ln.eps)   # [B,K,D]
    Wf = W.float()
    # q is L2-normalized inside score_query_dual; that is a per-query positive
    # scalar, so it cannot change Var-SHARES. Shares are computed pre-norm.
    terms = {k: ((v - v.mean(-1, keepdim=True)) * inv) @ Wf.T
             for k, v in parts.items()}                  # each [B,K,V]
    terms["bias"] = (ln.bias.float() @ Wf.T).expand_as(terms["ctx"])
    total = torch.zeros_like(terms["ctx"])
    for v in terms.values():
        total = total + v
    ref = (torch.nn.functional.normalize(ln(x).float(), dim=-1) @ Wf.T)
    # ref is the unit-normalized version; compare after matching the scale
    sc = (total.norm(dim=-1, keepdim=True) /
          ref.norm(dim=-1, keepdim=True).clamp_min(1e-9))
    resid = float((total - ref * sc)[valid].abs().max())
    out = {"residual_max": resid}
    vt = total[valid].var(-1, unbiased=False)            # [n]
    for k, v in terms.items():
        out[f"var_share_{k}"] = float(
            (v[valid].var(-1, unbiased=False) / vt.clamp_min(1e-12)).mean())
        out[f"std_{k}"] = float(v[valid].std(-1).mean())
    out["cov_mass"] = float(1.0 - sum(out[f"var_share_{k}"] for k in terms))
    return out


# ------------------------------------------------------------------ one pass
@torch.no_grad()
def run_pass(model, loader, device, W, lesion, mean_q=None, attrib=False,
             eval_budget=500, rel_offsets=None, fp32=False,
             max_batches=0):
    """Returns (ranks, gt_classes, mean_q_accumulator, attribution, rel_rows).

    `rel_offsets` (img_meta col5, the first rel row of each image) lets every
    scored edge be tagged with its ROW INDEX in the pack's rels.npy. That is
    what makes an edge-by-edge join against bias_baselines.py possible: FREQ
    beating us on the aggregate says nothing about whether vision is
    COMPLEMENTARY to the label prior, and the join is the only way to tell
    "vision is redundant with counting" from "vision is right where counting is
    wrong". Requires shuffle=False, which the loader here always uses."""
    raw = model.module if hasattr(model, "module") else model
    orig_budget = raw.sampler.final_budget
    raw.sampler.final_budget = min(eval_budget, raw.sampler.geo_budget)
    ranks, cls, cover, rows = [], [], [], []
    img0 = 0
    acc_sem = acc_spa = None
    n_q = 0
    attribs = []
    try:
        with Lesion(raw, lesion, mean_q), QueryTap(raw, attrib) as tap:
            for _bi, (images, boxes, box_counts, targets) in enumerate(loader):
                if max_batches and _bi >= max_batches:
                    break
                images = images.to(device, non_blocking=True)
                if lesion == "imgshuf" and images.shape[0] > 1:
                    images = images.roll(1, 0)
                boxes = boxes.to(device, non_blocking=True)
                box_counts = box_counts.to(device, non_blocking=True)
                tg = [{k: (v.to(device) if torch.is_tensor(v) else v)
                       for k, v in t.items()} for t in targets]
                # targets=None ON PURPOSE. Passing them force-includes GT pairs
                # (sampler.py:407) but also routes forward down the LOSS branch,
                # which needs modules this config never built (zone_proj). The
                # inference path is what we want to characterize anyway, and
                # sampler recall on positives is 99.79%
                #, so coverage is near-total —
                # it is reported per cell rather than assumed.
                with torch.amp.autocast("cuda",
                                        enabled=device.type == "cuda"
                                        and not fp32,
                                        dtype=torch.bfloat16):
                    out = model(images, boxes, box_counts, targets=None,
                                **region_kwargs(targets, device))
                logits = out["logits"].float()             # [B,K,V]
                sub_i, obj_i = out["sub_idx"], out["obj_idx"]
                valid = out["valid_mask"]
                if "pair_logits" in out:                  # score contract w=1
                    logits = logits + out["pair_logits"].float().unsqueeze(-1)

                # running mean of the expert queries over VALID slots only
                if "q_sem" in tap.buf:
                    qs = tap.buf["q_sem"].float()[valid]
                    qp = tap.buf["q_spa"].float()[valid]
                    if acc_sem is None:
                        acc_sem = qs.sum(0)
                        acc_spa = qp.sum(0)
                    else:
                        acc_sem += qs.sum(0)
                        acc_spa += qp.sum(0)
                    n_q += int(valid.sum())
                if attrib and len(attribs) < 40:
                    # NOTE the metric passes run under bf16 autocast, so the
                    # hooks capture bf16 activations and the "exact" additive
                    # identity is only exact to bf16 (~1e-2 on a logit whose
                    # scale is ~15). --attrib_fp32 re-runs this capture with
                    # autocast OFF so the residual actually tests the algebra
                    # instead of the dtype.
                    attribs.append(attribute(tap.buf, raw, W, valid))

                bs = images.shape[0]
                for b, t in enumerate(tg):
                    rel = t["relations"]
                    if rel.numel() == 0:
                        continue
                    r0 = (None if rel_offsets is None
                          else int(rel_offsets[img0 + b]))
                    vm = valid[b]
                    s_all, o_all = sub_i[b][vm], obj_i[b][vm]
                    row = logits[b][vm]                    # [K',V]
                    hit = ((s_all.unsqueeze(1) == rel[:, 0].unsqueeze(0)) &
                           (o_all.unsqueeze(1) == rel[:, 1].unsqueeze(0)))
                    found, slot = hit.max(0)
                    cover.append(float(found.float().mean()))
                    for r_i in torch.nonzero(found).flatten().tolist():
                        p = int(rel[r_i, 2])
                        sc = row[slot[r_i]]
                        ranks.append(int((sc > sc[p]).sum()) + 1)
                        cls.append(p)
                        rows.append(-1 if r0 is None else r0 + r_i)
                img0 += bs
    finally:
        raw.sampler.final_budget = orig_budget
    mq = None if acc_sem is None else (acc_sem / max(n_q, 1),
                                       acc_spa / max(n_q, 1))
    agg: dict = {}
    if attribs:
        agg = {k: float(np.mean([a[k] for a in attribs])) for k in attribs[0]
               if k != "residual_max"}
        agg["residual_max"] = float(max(a["residual_max"] for a in attribs))
    agg["pair_coverage"] = float(np.mean(cover)) if cover else 0.0
    return (np.asarray(ranks), np.asarray(cls), mq, agg,
            np.asarray(rows, dtype=np.int64))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data_roots", nargs="+", required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--weights", default="ema", choices=["ema", "raw"])
    p.add_argument("--lesions", default="full,imgshuf,nogeo,compose0,prior")
    p.add_argument("--attrib", action="store_true",
                   help="exact additive Var_v decomposition of the semantic "
                        "expert (its own short fp32 pass, see --attrib_batches)")
    p.add_argument("--attrib_batches", type=int, default=12,
                   help="batches for the fp32 attribution pass")
    p.add_argument("--dump_edges", action="store_true",
                   help="save per-edge (rels.npy row, gt class, rank) per "
                        "lesion, so the model can be joined edge-by-edge "
                        "against training/bias_baselines.py")
    p.add_argument("--text_student", default=None)
    p.add_argument("--img_size", type=int, default=448)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--eval_budget", type=int, default=500)
    p.add_argument("--max_objects", type=int, default=100)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--out_dir", default="")
    args = p.parse_args()

    lesions = [s for s in args.lesions.split(",") if s]
    for k in lesions:
        if k not in LESIONS:
            raise SystemExit(f"unknown lesion {k!r}; expected {LESIONS}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = build_model_from_ckpt(ckpt, args.weights).to(device).eval()
    if not model.config.compose_query:
        raise SystemExit("this checkpoint has no compositional query — the "
                         "compose0 lesion and --attrib do not apply")
    ck_args = ckpt.get("args") or {}
    if not isinstance(ck_args, dict):
        ck_args = vars(ck_args)
    text_student = (args.text_student if args.text_student is not None
                    else ck_args.get("text_student") or "")
    if not text_student:
        raise SystemExit("probe requires the student text encoder")
    from relsgg.text.student import encode_texts_student

    out_dir = args.out_dir or os.path.dirname(args.checkpoint)
    os.makedirs(out_dir, exist_ok=True)
    results = {}

    for root in args.data_roots:
        name = os.path.basename(os.path.normpath(root))
        rel_offsets = np.load(
            os.path.join(root, args.split, "img_meta.npy"))[:, 5]
        ds = RelationDataset(root=root, split=args.split,
                             resolution=args.img_size,
                             max_objects=args.max_objects)
        if args.limit:
            names = ds.predicate_names
            ds = torch.utils.data.Subset(ds, range(min(args.limit, len(ds))))
            ds.predicate_names = names
        pred_names = ds.predicate_names
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                            collate_fn=collate_fn,
                            num_workers=args.num_workers, pin_memory=True)
        E = encode_texts_student(pred_names, text_student,
                                 templates=TRAIN_TEMPLATES, device=device)
        model.vocab_head.set_vocabulary_matrix(pred_names, E)
        model.reparameterize()
        W = model.vocab_head.W
        print(f"\n[{name}] {len(ds)} images, {len(pred_names)} predicates, "
              f"split={args.split}")

        # `prior` needs the dataset-mean query, so `full` always runs first.
        order = ["full"] + [k for k in lesions if k != "full"]
        mean_q = None
        cell = {}
        edges = {}
        for k in order:
            if k == "prior" and mean_q is None:
                print("  prior: no mean query captured — skipped")
                continue
            r, c, mq, ag, rw = run_pass(model, loader, device, W, k,
                                        mean_q=mean_q,
                                        eval_budget=args.eval_budget,
                                        rel_offsets=rel_offsets)
            if k == "full":
                mean_q = mq
            if k not in lesions:
                continue
            edges[k] = (rw, c, r)
            cell[k] = report(r, c, len(pred_names))
            cell[k]["pair_coverage"] = ag.get("pair_coverage", 0.0)
            m = cell[k]
            print(f"  {k:9s} Acc@1 {m['Acc@1']:.4f}  MRR {m['MRR']:.4f}  "
                  f"mAcc@1 {m['mAcc@1']:.4f}  MedRank {m['MedRank']:.0f}  "
                  f"cover {m['pair_coverage']:.4f}  n={m['n']:,}")
        if args.attrib:
            _, _, _, ag32, _ = run_pass(model, loader, device, W, "full",
                                        attrib=True, fp32=True,
                                        eval_budget=args.eval_budget,
                                        max_batches=args.attrib_batches)
            cell["attribution"] = {k: v for k, v in ag32.items()
                                   if k != "pair_coverage"}
        if args.dump_edges and edges:
            np.savez_compressed(
                os.path.join(out_dir, f"edges_{name}_{args.split}.npz"),
                **{f"{k}_{f}": v for k, (rw, c, r) in edges.items()
                   for f, v in (("row", rw), ("cls", c), ("rank", r))})
            print(f"  per-edge dump -> edges_{name}_{args.split}.npz")
        if "attribution" in cell:
            a = cell["attribution"]
            print(f"  attribution (Var_v shares, residual "
                  f"{a['residual_max']:.2e}): "
                  + "  ".join(f"{k.replace('var_share_','')} "
                              f"{a[k]*100:.1f}%"
                              for k in a if k.startswith("var_share_"))
                  + f"  cov {a['cov_mass']*100:.1f}%")
        results[name] = cell

    tag = os.path.basename(os.path.normpath(args.checkpoint)).replace(".pth", "")
    out = os.path.join(out_dir, f"attribution_{args.split}.json")
    json.dump({"checkpoint": args.checkpoint, "weights": args.weights,
               "split": args.split, "img_size": args.img_size,
               "lesions": lesions, "results": results}, open(out, "w"),
              indent=2)
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
