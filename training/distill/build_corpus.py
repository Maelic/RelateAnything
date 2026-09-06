#!/usr/bin/env python
"""build_corpus.py — Phase A of the dino.txt text-encoder distillation.

Builds the predicate-string corpus the student will learn, the synonym/inverse
oracle pairs it is supervised with, a compact token vocabulary (so the student
is tiny — it "only cares about predicates"), and — with ``--encode-teacher`` on
a GPU — the frozen dino.txt target embeddings.

Corpus = union of predicate strings from the megasg / vg150 / psg packs PLUS
each pack's raw pre-canonicalization surface forms (``raw_links[*].raw``).
megasg is the training split; vg150/psg are held out to measure generalization
of the reparameterize()-time text space.

Reuses (never reimplements):
  training/text_space_diag.py:: build_groups, TEMPLATE_SETS,
      encode_all_templates, encode_dinotxt, combine_templates, DINOTXT_CKPT

Outputs (to --out, default runs/packed/text_student):
  corpus.json               strings, per-string provenance + split, canonical
                            key, synonym pairs, inverse pairs (index space =
                            corpus order)
  token_vocab.json          compact CLIP-BPE token id list (0=pad, 1=unk, …) —
                            the student's reduced embedding table
  teacher_targets_<tset>.npz  (only with --encode-teacher) fp16 [N, 2048]
                            L2-normalised dino.txt embeddings, corpus order

Usage:
  # CPU, login node — corpus + token vocab only:
  python training/distill/build_corpus.py
  # GPU — also cache the frozen teacher targets:
  python training/distill/build_corpus.py --encode-teacher
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJ))

from training.text_space_diag import (# noqa: E402
    TEMPLATE_SETS,
    build_groups,
    combine_templates,
    encode_all_templates,
)

PACKS = {
    "megasg": {"root": "runs/packed/megasg", "split": "train"},
    "vg150": {"root": "runs/packed/vg150", "split": "heldout"},
    "psg": {"root": "runs/packed/psg", "split": "heldout"},
}


def load_pack(root: str):
    meta = json.load(open(PROJ / root / "train" / "meta.json"))
    return meta["predicates"], meta.get("raw_links", [])


def build_corpus(holdout_frac: float = 0.1, seed: int = 0,
                 spatial_grammar: bool = False,
                 spatial_holdout: tuple = ()) -> dict:
    """Union corpus with provenance + reuse build_groups for oracle pairs.

    Because megasg's 10K vocabulary already subsumes almost all vg150/psg
    predicates, holding out only those two packs leaves a trivially small unseen
    set. To get a real generalization signal we ALSO randomly hold out
    ``holdout_frac`` of the train-split strings (the student never distills on
    them; eval reports on this slice in isolation)."""
    # provenance[str] = {"packs": set, "split": "train"|"heldout", "raw": bool}
    provenance: dict[str, dict] = {}
    merged_raw_links: list[dict] = []

    def note(s: str, pack: str, split: str, is_raw: bool):
        s = s.strip()
        if not s:
            return
        p = provenance.setdefault(
            s, {"packs": set(), "split": "heldout", "raw": True}
)
        p["packs"].add(pack)
        # a string is "train" if it appears as a real predicate in the train pack
        if split == "train":
            p["split"] = "train"
        p["raw"] = p["raw"] and is_raw  # False once seen as a real predicate

    for pack, info in PACKS.items():
        preds, raw_links = load_pack(info["root"])
        for p in preds:
            note(p, pack, info["split"], is_raw=False)
        for link in raw_links:
            note(link["raw"], pack, info["split"], is_raw=True)
            note(link["predicate"], pack, info["split"], is_raw=False)
            merged_raw_links.append(link)

    # Compositional spatial paraphrases (spatial_grammar.py). These occur in no
    # pack, so without this the student never sees "on the underside of" or
    # "diagonally above" and places them arbitrarily. Their axis/polarity is
    # known by construction, so they are supervised by oracle pairs rather than
    # distilled from the direction-blind teacher.
    grammar_meta: dict[str, dict] = {}
    if spatial_grammar:
        from training.distill.spatial_grammar import generate as _gen

        entries, bases = _gen()
        for b in bases:
            note(b, "grammar", "train", is_raw=False)
        for e in entries:
            held = e["family"] in spatial_holdout
            note(e["s"], "grammar", "heldout" if held else "train",
                 is_raw=False)
            grammar_meta[e["s"]] = e

    # Deterministic order: sorted so runs are reproducible and diffable.
    strings = sorted(provenance)

    # Random held-out slice of the train-split strings (real unseen-string test).
    rng = np.random.default_rng(seed)
    train_strings = [s for s in strings if provenance[s]["split"] == "train"]
    n_hold = int(round(holdout_frac * len(train_strings)))
    held = set(rng.choice(train_strings, size=n_hold, replace=False).tolist()) \
        if n_hold else set()
    for s in held:
        provenance[s]["split"] = "heldout"
        provenance[s]["rand_holdout"] = True

    # Oracle pairs over the *union* index space, via the exact diag logic.
    canon, syn_pairs, inv_pairs = build_groups(strings, merged_raw_links)
    syn_out = [list(p) for p in syn_pairs]
    inv_out = [list(p) for p in inv_pairs]
    soft_out: list = []

    if spatial_grammar:
        from training.distill.spatial_grammar import oracle_pairs

        index = {s: i for i, s in enumerate(strings)}
        g_hard, g_soft, g_inv = oracle_pairs(index)
        # Dedupe against what build_groups already found for the same strings.
        def merge(base: list, extra: list) -> list:
            seen = {tuple(sorted(p)) for p in base}
            return base + [p for p in extra
                           if tuple(sorted(p)) not in seen]

        syn_out = merge(syn_out, g_hard)
        inv_out = merge(inv_out, g_inv)
        soft_out = merge([], g_soft)
        print(f"[grammar] +{len(g_hard)} hard-synonym, +{len(g_soft)} "
              f"soft-synonym, +{len(g_inv)} inverse pairs over "
              f"{len(grammar_meta)} generated strings")

    prov_out = [
        {
            "s": s,
            "packs": sorted(provenance[s]["packs"]),
            "split": provenance[s]["split"],
            "raw_only": provenance[s]["raw"],
            "rand_holdout": provenance[s].get("rand_holdout", False),
            "canon": canon[i],
            **({"grammar": grammar_meta[s]} if s in grammar_meta else {}),
        }
        for i, s in enumerate(strings)
    ]
    return {
        "strings": strings,
        "provenance": prov_out,
        "canon": canon,
        "synonym_pairs": syn_out,
        "soft_synonym_pairs": soft_out,
        "inverse_pairs": inv_out,
        "n_train": sum(1 for p in prov_out if p["split"] == "train"),
        "n_heldout": sum(1 for p in prov_out if p["split"] == "heldout"),
    }


def build_token_vocab(strings: list[str]) -> dict:
    """Compact CLIP-BPE token vocabulary over all template expansions.

    The student embeds *templated* strings (same contract as the diag
    encoders), so the vocab must cover every token any template can produce.
    Compact ids: 0 = pad, 1 = unk, then sorted unique CLIP token ids.
    """
    from transformers import CLIPTokenizer

    tok = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    all_templates = sorted({t for ts in TEMPLATE_SETS.values() for t in ts})
    seen: set[int] = set()
    for t in all_templates:
        enc = tok([t.format(p=s) for s in strings], truncation=True, max_length=64)
        for ids in enc["input_ids"]:
            seen.update(int(i) for i in ids)
    clip_ids = sorted(seen)
    # compact 0=pad, 1=unk; keep the CLIP pad/eos/bos ids mapped too (they're in seen)
    compact = {cid: i + 2 for i, cid in enumerate(clip_ids)}
    return {
        "clip_ids": clip_ids,            # compact_id - 2 -> clip token id
        "compact_of_clip": compact,      # clip token id -> compact id (>=2)
        "pad": 0,
        "unk": 1,
        "size": len(clip_ids) + 2,
        "templates": all_templates,
    }


def encode_teacher(strings: list[str], out_dir: Path, force: bool = False) -> None:
    """Cache frozen dino.txt targets for every template set (GPU, inference)."""
    from training.text_space_diag import encode_dinotxt

    existing = [out_dir / f"teacher_targets_{t}.npz" for t in TEMPLATE_SETS]
    if not force and all(p.exists() for p in existing):
        # validate the cached targets still match the current corpus order
        z = np.load(existing[0], allow_pickle=True)
        if [str(s) for s in z["strings"]] == strings:
            print(f"[teacher] cached targets match corpus ({len(strings)} strings) "
                  f"— skipping dino.txt encode (use --force-teacher to redo)")
            return
        print("[teacher] cached targets stale (corpus changed) — re-encoding")

    print(f"[teacher] encoding {len(strings)} strings × "
          f"{len({t for ts in TEMPLATE_SETS.values() for t in ts})} templates "
          f"with dino.txt …")
    per_template = encode_all_templates(encode_dinotxt, strings)
    for tset, templates in TEMPLATE_SETS.items():
        E = combine_templates(per_template, templates)          # [N, 2048]
        np.savez_compressed(
            out_dir / f"teacher_targets_{tset}.npz",
            embeddings=E.astype(np.float16),
            strings=np.array(strings, dtype=object),
            templates=np.array(templates, dtype=object),
)
        print(f"[teacher]   saved teacher_targets_{tset}.npz  {E.shape}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/packed/text_student")
    ap.add_argument("--encode-teacher", action="store_true",
                    help="Also cache frozen dino.txt targets (needs a GPU).")
    ap.add_argument("--force-teacher", action="store_true",
                    help="Re-encode dino.txt targets even if cached.")
    ap.add_argument("--holdout_frac", type=float, default=0.1,
                    help="Random fraction of train strings held out for the "
                         "generalization test (student never trains on them).")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--spatial_grammar", action="store_true",
                    help="Add compositional spatial paraphrases "
                         "(spatial_grammar.py) with oracle axis/polarity "
                         "supervision — targets the open-vocabulary spatial "
                         "failures ('on the underside of', 'diagonally "
                         "above', 'lower than').")
    ap.add_argument("--spatial_holdout", nargs="*", default=[],
                    help="Grammar families to hold out entirely (e.g. "
                         "partitive diagonal): their strings enter the corpus "
                         "but no loss, so eval measures compositional "
                         "generalization rather than memorization.")
    args = ap.parse_args()

    out_dir = PROJ / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    corpus = build_corpus(holdout_frac=args.holdout_frac, seed=args.seed,
                          spatial_grammar=args.spatial_grammar,
                          spatial_holdout=tuple(args.spatial_holdout))
    strings = corpus["strings"]
    print(f"corpus: {len(strings)} strings "
          f"({corpus['n_train']} train / {corpus['n_heldout']} heldout); "
          f"{len(corpus['synonym_pairs'])} synonym pairs, "
          f"{len(corpus['soft_synonym_pairs'])} soft-synonym pairs, "
          f"{len(corpus['inverse_pairs'])} inverse pairs")
    json.dump(corpus, open(out_dir / "corpus.json", "w"), indent=1)

    vocab = build_token_vocab(strings)
    print(f"token vocab: {vocab['size']} compact tokens "
          f"(incl. pad/unk) over {len(vocab['templates'])} templates")
    # compact_of_clip has int keys — JSON needs str keys; store clip_ids only
    json.dump(
        {k: v for k, v in vocab.items() if k != "compact_of_clip"},
        open(out_dir / "token_vocab.json", "w"), indent=1,
)

    if args.encode_teacher:
        encode_teacher(strings, out_dir, force=args.force_teacher)

    print(f"\nArtifacts → {out_dir}")


if __name__ == "__main__":
    main()
