#!/usr/bin/env python
"""text_space_diag.py — measure whether candidate text spaces separate the
10K MEGASG predicates the way training needs them separated. Runs BEFORE any
GPU training is spent.

For each text encoder (dino.txt, SigLIP2, CLIP-B/32) and template set:

  1. Synonym separation: cosine of predicate pairs sharing a canonical form
     (sgg_canon groups + canonicalization-only raw links) vs random pairs
     → ROC-AUC. High AUC ⇒ masking negatives above τ_ignore is safe.
  2. Inverse (antonym) separation: cosine of spatial-inverse pairs
     ("above"/"below" and their surface variants). Bag-of-words encoders
     embed these nearly identically — if inverse cosine ≈ synonym cosine the
     text space cannot carry directionality and the visual side must.
  3. τ_ignore calibration: smallest τ with ≥95% synonym recall; report the
     fraction of random and inverse pairs that would be wrongly ignored.

Artifacts per encoder → <out>/:
    pred_embeds_<enc>.npz   fp16 [V, D] embeddings in meta.json predicate order
                            (the training-time W matrix — reused, not recomputed)
    diag_<enc>.json         all metrics
Also writes canonical_groups.json (predicate → canonical key) used everywhere
downstream (synonym-aware loss masking, soft evaluator). Canonical forms are
LOGIC ONLY — the emitted per-relation labels are never rewritten.

Usage (login node,.venv — downloads SigLIP2/CLIP into HF_HOME on first run):
    python training/text_space_diag.py --root runs/packed/megasg \
        --encoders dinotxt siglip2 clip
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

PROJ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ))

from datagen.sgg_canon import canonicalize_spatial  # noqa: E402
from training.invert_spatial import INVERSE  # noqa: E402

TEMPLATE_SETS = {
    "plain": ["{p}"],
    "carrier": ["{p}", "one object is {p} another object"],
    "photo": ["{p}", "one object is {p} another object",
              "a photo of something {p} something"],
}

DINOTXT_CKPT = PROJ / "checkpoints/dinov3_vitl16_dinotxt_vision_head_and_text_encoder-a442d8f5.pth"


# ---------------------------------------------------------------------------
# Canonical groups + inverse pairs (the oracle)
# ---------------------------------------------------------------------------

def build_groups(predicates: list[str], raw_links: list[dict]):
    """Return (canon_key per predicate, synonym pairs, inverse pairs).

    Canonical key = canonicalize_spatial(collapse_spatial=True) — proximity
    forms map to "near" instead of being dropped; predicates the canonicalizer
    would drop keep themselves as key (they exist in the data, so they exist
    in the vocabulary).

    raw_links (predicate_raw → emitted predicate, from pack meta) are split by
    what transformed them: same canonical form ⇒ synonym; canonical forms that
    are spatial inverses ⇒ the geometry FLIPPED the relation ⇒ antonym pair.
    """
    idx = {p: i for i, p in enumerate(predicates)}
    canon = [canonicalize_spatial(p, collapse_spatial=True) or p for p in predicates]

    groups = defaultdict(list)
    for i, c in enumerate(canon):
        groups[c].append(i)

    syn_pairs = set()
    for members in groups.values():
        for a in range(len(members)):
            for b in range(a + 1, len(members)):
                syn_pairs.add((members[a], members[b]))

    inv_pairs = set()
    for link in raw_links:
        raw, pred = link["raw"], link["predicate"]
        i, j = idx.get(raw), idx.get(pred)
        c_raw = canonicalize_spatial(raw, collapse_spatial=True) or raw
        c_pred = canonicalize_spatial(pred, collapse_spatial=True) or pred
        if i is not None and j is not None and i != j:
            if c_raw == c_pred:
                syn_pairs.add(tuple(sorted((i, j))))
            elif INVERSE.get(c_raw) == c_pred:
                inv_pairs.add(tuple(sorted((i, j))))

    # Inverse pairs across whole canonical groups ("atop" vs "underneath" etc.)
    for a_canon, b_canon in INVERSE.items():
        for i in groups.get(a_canon, []):
            for j in groups.get(b_canon, []):
                if i != j:
                    inv_pairs.add(tuple(sorted((i, j))))
    syn_pairs -= inv_pairs

    return canon, sorted(syn_pairs), sorted(inv_pairs)


# ---------------------------------------------------------------------------
# Encoders — each returns L2-normalised fp32 [V, D] for a list of strings
# ---------------------------------------------------------------------------

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def encode_dinotxt(texts: list[str], batch: int = 256) -> np.ndarray:
    from transformers import CLIPTokenizer
    from training.distill.teacher import DinoTxtEncoder

    enc = DinoTxtEncoder.from_checkpoint(str(DINOTXT_CKPT), DEVICE)
    tok = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    out = []
    for i in range(0, len(texts), batch):
        ids = tok(texts[i:i + batch], return_tensors="pt", padding="max_length",
                  truncation=True, max_length=DinoTxtEncoder.CTX_LEN)["input_ids"]
        out.append(enc.encode(ids.to(DEVICE)).cpu().numpy())
        print(f"  dinotxt {i + len(ids)}/{len(texts)}", end="\r", flush=True)
    print()
    return np.concatenate(out)


def encode_siglip2(texts: list[str], batch: int = 256) -> np.ndarray:
    from transformers import AutoModel, AutoTokenizer

    name = "google/siglip2-base-patch16-224"
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModel.from_pretrained(name).eval().to(DEVICE)
    out = []
    with torch.inference_mode():
        for i in range(0, len(texts), batch):
            inputs = tok(texts[i:i + batch], return_tensors="pt",
                         padding="max_length", truncation=True, max_length=64)
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
            emb = model.get_text_features(**inputs)
            if not isinstance(emb, torch.Tensor):  # transformers 5.x output obj
                emb = emb.pooler_output
            out.append(torch.nn.functional.normalize(emb.float(), dim=-1).cpu().numpy())
            print(f"  siglip2 {i + len(inputs['input_ids'])}/{len(texts)}",
                  end="\r", flush=True)
    print()
    return np.concatenate(out)


def encode_clip(texts: list[str], batch: int = 256) -> np.ndarray:
    from transformers import CLIPModel, CLIPTokenizer

    name = "openai/clip-vit-base-patch32"
    tok = CLIPTokenizer.from_pretrained(name)
    model = CLIPModel.from_pretrained(name).eval().to(DEVICE)
    out = []
    with torch.inference_mode():
        for i in range(0, len(texts), batch):
            inputs = tok(texts[i:i + batch], return_tensors="pt", padding=True,
                         truncation=True, max_length=77)
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
            emb = model.get_text_features(**inputs)
            if not isinstance(emb, torch.Tensor):  # transformers 5.x output obj
                emb = emb.pooler_output
            out.append(torch.nn.functional.normalize(emb.float(), dim=-1).cpu().numpy())
            print(f"  clip {i + len(inputs['input_ids'])}/{len(texts)}",
                  end="\r", flush=True)
    print()
    return np.concatenate(out)


ENCODERS = {"dinotxt": encode_dinotxt, "siglip2": encode_siglip2,
            "clip": encode_clip}


def encode_all_templates(fn, predicates: list[str]) -> dict[str, np.ndarray]:
    """Encode each unique template once (the sets are nested)."""
    unique = sorted({t for ts in TEMPLATE_SETS.values() for t in ts})
    return {t: fn([t.format(p=p) for p in predicates]) for t in unique}


def combine_templates(per_template: dict[str, np.ndarray],
                      templates: list[str]) -> np.ndarray:
    embs = sum(per_template[t] for t in templates)
    return embs / np.linalg.norm(embs, axis=-1, keepdims=True)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def pair_cosines(E: np.ndarray, pairs: list[tuple[int, int]]) -> np.ndarray:
    if not pairs:
        return np.zeros(0, dtype=np.float32)
    a = np.fromiter((p[0] for p in pairs), int, len(pairs))
    b = np.fromiter((p[1] for p in pairs), int, len(pairs))
    return np.einsum("ij,ij->i", E[a], E[b])


def roc_auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """AUC via rank statistic (no sklearn dependency)."""
    scores = np.concatenate([pos, neg])
    ranks = scores.argsort().argsort().astype(np.float64) + 1
    r_pos = ranks[: len(pos)].sum()
    return float((r_pos - len(pos) * (len(pos) + 1) / 2)
                 / (len(pos) * len(neg) + 1e-9))


def evaluate(E: np.ndarray, syn, inv, rng) -> dict:
    n_rand = min(200_000, len(E) * 20)
    rand_pairs = list(zip(rng.integers(0, len(E), n_rand),
                          rng.integers(0, len(E), n_rand)))
    rand_pairs = [(a, b) for a, b in rand_pairs if a != b]

    cos_syn = pair_cosines(E, syn)
    cos_inv = pair_cosines(E, inv)
    cos_rand = pair_cosines(E, rand_pairs)

    tau95 = float(np.quantile(cos_syn, 0.05))  # 95% synonym recall
    return {
        "synonym_pairs": len(cos_syn),
        "inverse_pairs": len(cos_inv),
        "cos_synonym_mean": float(cos_syn.mean()),
        "cos_inverse_mean": float(cos_inv.mean()),
        "cos_random_mean": float(cos_rand.mean()),
        "auc_syn_vs_rand": roc_auc(cos_syn, cos_rand),
        "auc_syn_vs_inv": roc_auc(cos_syn, cos_inv),
        "tau_ignore_at_95pct_syn_recall": tau95,
        "random_frac_above_tau": float((cos_rand >= tau95).mean()),
        "inverse_frac_above_tau": float((cos_inv >= tau95).mean()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="runs/packed/megasg")
    ap.add_argument("--encoders", nargs="+", default=["dinotxt", "siglip2", "clip"],
                    choices=sorted(ENCODERS))
    ap.add_argument("--out", default=None,
                    help="Output dir (default: <root>/text_space)")
    args = ap.parse_args()

    meta = json.load(open(Path(args.root) / "train" / "meta.json"))
    predicates: list[str] = meta["predicates"]
    out_dir = Path(args.out or (Path(args.root) / "text_space"))
    out_dir.mkdir(parents=True, exist_ok=True)

    canon, syn_pairs, inv_pairs = build_groups(predicates, meta["raw_links"])
    n_groups = len(set(canon))
    print(f"{len(predicates)} predicates → {n_groups} canonical groups; "
          f"{len(syn_pairs)} synonym pairs, {len(inv_pairs)} inverse pairs")
    json.dump({p: c for p, c in zip(predicates, canon)},
              open(out_dir / "canonical_groups.json", "w"), indent=1)

    rng = np.random.default_rng(0)
    report = {"n_predicates": len(predicates), "n_canonical_groups": n_groups}

    for enc_name in args.encoders:
        fn = ENCODERS[enc_name]
        report[enc_name] = {}
        print(f"\n== encoding all templates with {enc_name} ==")
        per_template = encode_all_templates(fn, predicates)
        for tset_name, templates in TEMPLATE_SETS.items():
            print(f"\n== {enc_name} / templates={tset_name} ==")
            E = combine_templates(per_template, templates)
            m = evaluate(E, syn_pairs, inv_pairs, rng)
            report[enc_name][tset_name] = m
            for k, v in m.items():
                print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
            np.savez_compressed(
                out_dir / f"pred_embeds_{enc_name}_{tset_name}.npz",
                embeddings=E.astype(np.float16), predicates=predicates,
                templates=templates,
)
        json.dump(report[enc_name],
                  open(out_dir / f"diag_{enc_name}.json", "w"), indent=2)

    json.dump(report, open(out_dir / "diag_all.json", "w"), indent=2)
    print(f"\nAll artifacts → {out_dir}")


if __name__ == "__main__":
    main()
