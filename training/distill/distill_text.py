#!/usr/bin/env python
"""distill_text.py — Phase B: distil dino.txt into PredicateTextStudent while
actively separating spatial antonyms.

Objective (only TRAIN-split strings, i.e. megasg; vg150/psg held out):

  L_rel  relational distillation — match the student's pairwise-cosine matrix to
         the teacher's over each random minibatch. Dimension-agnostic (teacher
         2048-d, student 512-d) and exactly what the downstream masks read
         (``W @ W.T``). This carries the synonym/random geometry.
  L_abs  absolute anchor — MSE to a PCA down-projection of the teacher target;
         small weight, speeds convergence and pins the global frame.
  L_nbr  NEIGHBOURHOOD preservation — row-wise KL between the teacher's and the
         student's softmax over batch similarities. See below.
  L_ant  antonym repulsion — hinge pushing spatial-inverse pairs' student cosine
         below a margin. The teacher cannot have this (it is frozen and embeds
         antonyms nearly identically); it deliberately deviates from the teacher
         ONLY on inverse pairs while L_rel/L_abs hold everything else in place.
  L_syn  synonym cohesion — light hinge keeping canonical-group pairs' cosine
         high (mostly redundant with L_rel; cheap insurance).

WHAT v1 GOT WRONG (measured 2026-07-28, one run, 1,967 synonym pairs on the
datamix_v22 union vocabulary). The v1 student separated antonyms beautifully
(syn-vs-inv AUC 0.988 vs the teacher's 0.669, left/right cosine 0.323 vs 0.991)
and destroyed the general neighbourhood structure paying for it: NN@1-in-own-
group 0.189 vs the teacher's 0.348, effective dimensionality 24 vs 87.

The cause is NOT the antonym term overreaching, and not the margins fighting the
teacher — v1's history ends with l_ant and l_syn both exactly 0.0 (every hinge
satisfied) and cos_rand 0.790 against the teacher's 0.80. v1 reproduced the
teacher's cosine DISTRIBUTION almost perfectly. It is L_rel that is too weak a
constraint: MSE over a batch's pairwise cosines is dominated by the bulk, and in
raw dino.txt the bulk is a narrow cone where nearly every pair sits at ~0.80.
Fitting that bulk is achievable in very few dimensions, so the student drove
l_rel to 0.0023 while collapsing to 24 effective dimensions — matching every
cosine on average and no neighbourhood in particular. Fine-grained "who is whose
nearest neighbour" was never in the objective. Two changes address it:

  1. L_nbr — NEIGHBOURHOOD PRESERVATION. L_rel's MSE treats a 0.02 error on a
     distant pair the same as on a nearest neighbour, but NN@1 (the metric that
     matches a 19k-way argmax at deployment) only cares about the top of each
     row. A KL between teacher and student row-softmaxes over batch similarities
     concentrates the gradient exactly there.
  2. ``--teacher_abtt K`` — de-hub the TARGET. Raw dino.txt has mean off-diagonal
     cosine 0.80 and hubness skew 8.13; all-but-the-top takes that to 0.00 and
     0.91, and the teacher's own NN@1 from 0.348 to 0.409. A target whose
     cosines are spread carries neighbourhood information in its cosines, so
     L_rel stops being bulk-dominated as well.

CAUTION THIS INTRODUCES. The hinges are absolute-cosine margins, so they are only
meaningful relative to the scale of the target space. m_pos=0.92 was harmless
against the raw teacher (synonyms at 0.96) but after ABTT synonyms sit at ~0.49
and inverses at ~0.29 — the same margins would then demand a large deviation
from the teacher on every pair, which IS a request to discard the geometry. The
teacher's syn/inv/rand statistics are printed at startup and margins that would
fight L_rel are warned about.

Note ABTT also makes L_ant almost free everywhere EXCEPT left/right, which is
the one antonym axis ABTT cannot fix (0.879 at every k). That is the intent: the
antonym loss should be repairing one axis, not deforming the whole space.

Per-epoch it logs mean synonym / inverse / random student cosine on TRAIN pairs
— the live guardrail: synonym cosine must stay high while inverse cosine drops.

Usage:
  python training/distill/distill_text.py --epochs 300 --lam_ant 1.0
  python training/distill/distill_text.py --teacher_abtt 3 --lam_nbr 1.0 \
      --m_pos 0.60 --m_neg 0.15
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

PROJ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJ))

from relsgg.text.student import PredicateTextStudent  # noqa: E402


def load_corpus(art: Path):
    corpus = json.load(open(art / "corpus.json"))
    strings = corpus["strings"]
    train_mask = np.array([p["split"] == "train" for p in corpus["provenance"]])
    syn = np.array(corpus["synonym_pairs"], dtype=np.int64).reshape(-1, 2)
    inv = np.array(corpus["inverse_pairs"], dtype=np.int64).reshape(-1, 2)
    # Older corpora predate the spatial grammar and carry no soft pairs.
    soft = np.array(corpus.get("soft_synonym_pairs", []),
                    dtype=np.int64).reshape(-1, 2)
    return corpus, strings, train_mask, syn, inv, soft


def load_teacher(art: Path, tset: str, strings: list[str]) -> torch.Tensor:
    z = np.load(art / f"teacher_targets_{tset}.npz", allow_pickle=True)
    assert [str(s) for s in z["strings"]] == strings, \
        "teacher_targets order != corpus order — rebuild build_corpus.py"
    T = torch.from_numpy(z["embeddings"].astype(np.float32))
    return F.normalize(T, dim=-1)


def abtt(T: torch.Tensor, k: int) -> torch.Tensor:
    """All-but-the-top: drop the mean and the top-k principal directions.

    Same transform the diagnostics score (training/diag_isotropy_sweep.py), so a
    student distilled onto this target is directly comparable to the ``+abtt``
    teacher rows there. Exact eigendecomposition rather than pca_lowrank: D=2048
    makes it cheap and the top directions are precisely what must be removed.
    """
    if k <= 0:
        return F.normalize(T, dim=-1)
    Y = T - T.mean(0, keepdim=True)
    _, evecs = torch.linalg.eigh((Y.T @ Y).double())   # ascending eigenvalues
    D = evecs[:, -k:].to(Y.dtype)                      # top-k directions
    return F.normalize(Y - (Y @ D) @ D.T, dim=-1)


def pair_stats(E: torch.Tensor, syn, inv, n_rand: int = 20_000) -> dict:
    """Mean cosine over synonym / inverse / random pairs — the scale the margins
    must be chosen against."""
    g = torch.Generator(device="cpu").manual_seed(0)
    i = torch.randint(0, len(E), (n_rand,), generator=g).to(E.device)
    j = torch.randint(0, len(E), (n_rand,), generator=g).to(E.device)
    out = {"rand": float((E[i] * E[j]).sum(-1).mean())}
    for name, p in (("syn", syn), ("inv", inv)):
        out[name] = float((E[p[:, 0]] * E[p[:, 1]]).sum(-1).mean()) if len(p) \
            else float("nan")
    return out


def pca_project(T: torch.Tensor, out_dim: int) -> torch.Tensor:
    """PCA down-projection of teacher targets to out_dim, L2-normalised.

    Uses randomized low-rank PCA (fast on the 10K×2048 teacher matrix)."""
    q = min(out_dim, T.shape[0] - 1, T.shape[1])
    _, _, V = torch.pca_lowrank(T, q=q, center=True, niter=2)   # V: [D, q]
    proj = (T - T.mean(0, keepdim=True)) @ V                    # [N, q]
    if proj.shape[1] < out_dim:                                 # pad if q<out_dim
        proj = F.pad(proj, (0, out_dim - proj.shape[1]))
    return F.normalize(proj, dim=-1)


def clip_token_init(d_tok: int, device) -> torch.Tensor:
    """PCA of CLIP's token embeddings → [49408, d_tok], scaled for init.

    Gives every CLIP-BPE subword — including ones the predicate corpus never
    contained — a real semantic starting point (no dead <unk>)."""
    from transformers import CLIPTextModel
    m = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32").eval()
    W = m.get_input_embeddings().weight.detach().float().to(device)  # [49408,512]
    _, _, V = torch.pca_lowrank(W, q=min(d_tok, W.shape[1]), center=True, niter=2)
    proj = (W - W.mean(0, keepdim=True)) @ V                         # [49408,d_tok]
    proj = (proj - proj.mean()) / (proj.std() + 1e-6) * 0.02         # match init scale
    del m
    return proj.contiguous()


def filter_pairs(pairs: np.ndarray, train_mask: np.ndarray) -> torch.Tensor:
    """Keep only pairs whose BOTH endpoints are train strings."""
    if pairs.size == 0:
        return torch.zeros(0, 2, dtype=torch.long)
    keep = train_mask[pairs[:, 0]] & train_mask[pairs[:, 1]]
    return torch.from_numpy(pairs[keep])


def sample_rows(pool: torch.Tensor, n: int, device) -> torch.Tensor:
    if pool.numel() == 0:
        return pool.to(device)
    idx = torch.randint(0, pool.shape[0], (n,), device=pool.device)
    return pool[idx].to(device)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--art", default="runs/packed/text_student")
    ap.add_argument("--tset", default="photo", choices=["plain", "carrier", "photo"])
    ap.add_argument("--out", default=None, help="student ckpt path (default <art>/student.pt)")
    ap.add_argument("--dim", type=int, default=256)
    ap.add_argument("--d_tok", type=int, default=128, help="factorised embedding width")
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--out_dim", type=int, default=512)
    ap.add_argument("--token_init", default="clip", choices=["clip", "none"],
                    help="init full-vocab embedding from CLIP token embeddings")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--batch", type=int, default=512, help="random strings per step (L_rel)")
    ap.add_argument("--pair_batch", type=int, default=512, help="syn/inv pairs per step")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--teacher_abtt", type=int, default=0,
                    help="all-but-the-top on the TEACHER targets before "
                         "distilling (0 = v1 behaviour, raw teacher). 3-10 "
                         "removes the hub directions that make L_rel spend its "
                         "budget on the bulk instead of on neighbourhoods.")
    ap.add_argument("--lam_rel", type=float, default=1.0)
    ap.add_argument("--lam_abs", type=float, default=0.5)
    ap.add_argument("--lam_nbr", type=float, default=0.0,
                    help="neighbourhood-preservation KL; 0 = v1 behaviour")
    ap.add_argument("--tau_nbr", type=float, default=0.05,
                    help="temperature for L_nbr's row softmax — lower = only the "
                         "very top neighbours matter")
    ap.add_argument("--lam_ant", type=float, default=1.0)
    ap.add_argument("--lam_syn", type=float, default=0.25)
    ap.add_argument("--m_neg", type=float, default=0.25, help="inverse cos pushed below this")
    ap.add_argument("--m_pos", type=float, default=0.75, help="synonym cos pulled above this")
    ap.add_argument("--m_pos_soft", type=float, default=0.60,
                    help="soft-synonym (modified spatial form) cos pulled "
                         "above this — below --m_pos on purpose, so 'far "
                         "above' lands in the 'above' neighbourhood without "
                         "becoming indistinguishable from it. Shares "
                         "--lam_syn.")
    ap.add_argument("--limit", type=int, default=0, help="smoke-test: cap train strings")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    art = PROJ / args.art
    out_path = Path(args.out) if args.out else art / "student.pt"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_bf16 = device.type == "cuda" and torch.cuda.get_device_capability(0)[0] >= 8

    corpus, strings, train_mask, syn_all, inv_all, soft_all = load_corpus(art)
    T = load_teacher(art, args.tset, strings).to(device)          # [N, 2048]
    N = len(strings)

    train_idx = np.nonzero(train_mask)[0]
    if args.limit:
        train_idx = train_idx[: args.limit]
        lim = np.zeros(N, dtype=bool); lim[train_idx] = True; train_mask = lim
    train_idx_t = torch.from_numpy(train_idx).to(device)
    print(f"[distill] {N} corpus strings, {len(train_idx)} train; "
          f"device={device} bf16={use_bf16}")

    syn_tr = filter_pairs(syn_all, train_mask)
    inv_tr = filter_pairs(inv_all, train_mask)
    soft_tr = filter_pairs(soft_all, train_mask)
    print(f"[distill] train pairs: {len(syn_tr)} synonym, {len(soft_tr)} "
          f"soft-synonym, {len(inv_tr)} inverse")

    # ---- target geometry: optionally de-hubbed, and ALWAYS reported ----------
    # The margins below are hinges against absolute cosines, so they are only
    # meaningful relative to the scale of the space being distilled. v1 used
    # margins tuned for the raw teacher and then changed the teacher; printing
    # both makes that class of mistake visible instead of silent.
    if args.teacher_abtt:
        before = pair_stats(T, syn_tr.to(device), inv_tr.to(device))
        T = abtt(T, args.teacher_abtt)
        after = pair_stats(T, syn_tr.to(device), inv_tr.to(device))
        T_abs_src = T
        print(f"[distill] teacher ABTT k={args.teacher_abtt}:  "
              f"syn {before['syn']:.3f}->{after['syn']:.3f}  "
              f"inv {before['inv']:.3f}->{after['inv']:.3f}  "
              f"rand {before['rand']:.3f}->{after['rand']:.3f}")
        tstat = after
    else:
        T_abs_src = T
        tstat = pair_stats(T, syn_tr.to(device), inv_tr.to(device))
        print(f"[distill] teacher (raw):  syn {tstat['syn']:.3f}  "
              f"inv {tstat['inv']:.3f}  rand {tstat['rand']:.3f}")
    if args.m_pos > tstat["syn"] + 0.20:
        print(f"[distill] !! m_pos={args.m_pos} is {args.m_pos - tstat['syn']:.2f} "
              f"above the teacher's synonym cosine ({tstat['syn']:.3f}) — L_syn "
              f"will fight L_rel on every synonym pair. This is what cost v1 its "
              f"neighbourhood structure.")
    if args.m_neg > tstat["inv"]:
        print(f"[distill] !! m_neg={args.m_neg} is already satisfied by the "
              f"teacher (inv {tstat['inv']:.3f}) — L_ant will be inactive.")

    # absolute anchor target (PCA of teacher → out_dim); follows T through ABTT
    T_abs = pca_project(T_abs_src, args.out_dim)                  # [N, out_dim]

    # full-CLIP-vocab student (OOV-free), embedding seeded from CLIP tokens
    tok_init = clip_token_init(args.d_tok, device) if args.token_init == "clip" else None
    student = PredicateTextStudent.full_vocab(
        token_init=tok_init, d_tok=args.d_tok, dim=args.dim, depth=args.depth,
        heads=args.heads, out_dim=args.out_dim,
).to(device)
    # student sees the SAME templated strings as the teacher target set
    from training.text_space_diag import TEMPLATE_SETS  # noqa: E402
    tset_templates = TEMPLATE_SETS[args.tset]
    # teacher target already template-combined; student encodes the plain string
    # under the same template average → tokenise each template variant and mean-
    # pool the student embeddings too (mirror combine_templates on the student).
    tok_cache = {
        t: student.tokenize([t.format(p=s) for s in strings], device=device)
        for t in tset_templates
    }
    print(f"[distill] student params: {student.num_parameters()/1e6:.2f}M "
          f"(full CLIP vocab {student.cfg['vocab_size']}, d_tok={student.cfg['d_tok']})")

    def embed(idx: torch.Tensor) -> torch.Tensor:
        """Template-averaged, L2-normalised student embeddings for row indices."""
        acc = 0
        for t in tset_templates:
            ids, pad = tok_cache[t]
            acc = acc + student(ids[idx], pad[idx])
        return F.normalize(acc, dim=-1)

    opt = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    history = []

    for ep in range(args.epochs):
        student.train()
        t0 = time.time()
        # --- L_rel + L_abs on a random batch of train strings ---
        b = sample_rows(train_idx_t.unsqueeze(1), min(args.batch, len(train_idx)),
                        device).squeeze(1)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=use_bf16):
            s = embed(b)                                          # [B, out]
            with torch.no_grad():
                t_sim = T[b] @ T[b].T                             # teacher [B,B]
            s_sim = s @ s.T
            eye = torch.eye(len(b), device=device, dtype=torch.bool)
            l_rel = F.mse_loss(s_sim[~eye], t_sim[~eye])
            l_abs = F.mse_loss(s, T_abs[b])

            # --- L_nbr: preserve WHO IS WHOSE NEIGHBOUR, not just the cosines.
            # L_rel's MSE weights a distant pair exactly like a nearest one; the
            # row softmax puts the gradient on the top of each row, which is what
            # a 19k-way argmax actually reads.
            if args.lam_nbr:
                # Mask AFTER the temperature divide, with a finite sentinel. Masking
                # with -inf first overflows (finfo.min / 0.05 = -inf), and KL then
                # evaluates (-inf) - (-inf) = NaN on the diagonal. With a large finite
                # value the diagonal contributes exp(-huge) * (finite) = 0 instead.
                neg = torch.finfo(s_sim.dtype).min / 2
                l_nbr = F.kl_div(
                    F.log_softmax((s_sim / args.tau_nbr).masked_fill(eye, neg), -1),
                    F.log_softmax((t_sim / args.tau_nbr).masked_fill(eye, neg), -1),
                    log_target=True, reduction="batchmean")
            else:
                l_nbr = s.new_zeros(())

            # --- L_ant: push inverse pairs apart ---
            if len(inv_tr):
                pi = sample_rows(inv_tr, args.pair_batch, device)
                si, sj = embed(pi[:, 0]), embed(pi[:, 1])
                cos_inv = (si * sj).sum(-1)
                l_ant = F.relu(cos_inv - args.m_neg).mean()
            else:
                l_ant = s.new_zeros(())
                cos_inv = s.new_zeros(1)

            # --- L_syn: keep synonym pairs together ---
            if len(syn_tr):
                ps = sample_rows(syn_tr, args.pair_batch, device)
                si, sj = embed(ps[:, 0]), embed(ps[:, 1])
                cos_syn = (si * sj).sum(-1)
                l_syn = F.relu(args.m_pos - cos_syn).mean()
            else:
                l_syn = s.new_zeros(())
                cos_syn = s.new_zeros(1)

            # --- L_syn_soft: modified spatial forms ("far above", "diagonally
            # above") pulled NEAR their base but at a lower margin, so the
            # composition stays distinguishable instead of collapsing onto it.
            if len(soft_tr):
                pf = sample_rows(soft_tr, args.pair_batch, device)
                si, sj = embed(pf[:, 0]), embed(pf[:, 1])
                cos_soft = (si * sj).sum(-1)
                l_soft = F.relu(args.m_pos_soft - cos_soft).mean()
            else:
                l_soft = s.new_zeros(())
                cos_soft = s.new_zeros(1)

            loss = (args.lam_rel * l_rel + args.lam_abs * l_abs
                    + args.lam_nbr * l_nbr
                    + args.lam_ant * l_ant + args.lam_syn * l_syn
                    + args.lam_syn * l_soft)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        opt.step()
        sched.step()

        # --- guardrail metrics on train pairs (mean cosines) ---
        if ep % 10 == 0 or ep == args.epochs - 1:
            student.eval()
            with torch.no_grad():
                rnd = sample_rows(train_idx_t.unsqueeze(1), 2000, device).squeeze(1)
                rr = embed(rnd)
                cos_rand = (rr[:1000] * rr[1000:2000]).sum(-1).mean().item() \
                    if len(rnd) >= 2000 else float("nan")
                rec = dict(
                    epoch=ep, loss=float(loss), l_rel=float(l_rel),
                    l_abs=float(l_abs), l_nbr=float(l_nbr),
                    l_ant=float(l_ant), l_syn=float(l_syn),
                    l_soft=float(l_soft),
                    cos_syn=float(cos_syn.mean()), cos_inv=float(cos_inv.mean()),
                    cos_soft=float(cos_soft.mean()),
                    cos_rand=cos_rand, sec=time.time() - t0,
)
            history.append(rec)
            print(f"[ep {ep:4d}] loss {rec['loss']:.4f} | rel {rec['l_rel']:.4f} "
                  f"abs {rec['l_abs']:.4f} nbr {rec['l_nbr']:.4f} "
                  f"ant {rec['l_ant']:.4f} syn {rec['l_syn']:.4f} "
                  f"soft {rec['l_soft']:.4f} "
                  f"| cos syn {rec['cos_syn']:.3f} soft {rec['cos_soft']:.3f} "
                  f"inv {rec['cos_inv']:.3f} rand {rec['cos_rand']:.3f}")

    student.save(str(out_path))
    json.dump({"args": vars(args), "history": history},
              open(out_path.with_suffix(".history.json"), "w"), indent=1)
    print(f"\n[distill] saved student → {out_path}")
    print(f"[distill] final: cos_syn {history[-1]['cos_syn']:.3f} "
          f"cos_inv {history[-1]['cos_inv']:.3f} cos_rand {history[-1]['cos_rand']:.3f}")


if __name__ == "__main__":
    main()
