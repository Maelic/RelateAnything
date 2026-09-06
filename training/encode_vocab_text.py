"""Encode a predicate vocabulary with dino.txt or a distilled student.

WHY THIS EXISTS. The text-space scoreboard (`diag_text_spaces.py`) and the isotropy
sweep (`diag_isotropy_sweep.py`) can only compare encoders on a vocabulary where all of
them have embeddings, in the same order. That was MEGASG's 10,102 predicates — which
carries only **7 multi-member canonical groups (28 synonym pairs)**, so every
synonym-discrimination column in both diagnostics was computed on ~28 samples and could
not rank anything. The datamix_v22 union vocabulary has **544 multi-member groups (1,967
pairs)**, 70x more, and is the vocabulary the model actually trains on — but
`build_union_vocab.py` only ever writes the student's embeddings there.

Both encoders use the same "photo" template ensemble the training-time W was built with,
so a comparison between them isolates the ENCODER and not the prompt.

    # teacher
    python training/encode_vocab_text.py --encoder dinotxt \\
        --vocab runs/packed/datamix_v22/text_space/union_predicates.json \\
        --out   runs/packed/datamix_v22/text_space/pred_embeds_dinotxt_photo.npz

    # a distilled student
    python training/encode_vocab_text.py --encoder student \\
        --ckpt  runs/packed/text_student_v2/student.pt \\
        --vocab runs/packed/datamix_v22/text_space/union_predicates.json \\
        --out   runs/packed/datamix_v22/text_space/pred_embeds_studentv2_photo.npz
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PROJ = Path(__file__).resolve().parent.parent
DINOTXT_CKPT = PROJ / ("checkpoints/dinov3_vitl16_dinotxt_vision_head_and_"
                       "text_encoder-a442d8f5.pth")
# Identical to build_union_vocab.PHOTO_TEMPLATES — the student's W was built with these,
# so using anything else would confound "encoder" with "prompt" in the comparison.
PHOTO_TEMPLATES = ["{p}", "one object is {p} another object",
                   "a photo of something {p} something"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vocab", required=True,
                    help="JSON list of predicate strings, in the order to preserve")
    ap.add_argument("--out", required=True)
    ap.add_argument("--encoder", default="dinotxt", choices=["dinotxt", "student"])
    ap.add_argument("--ckpt", default="",
                    help="student checkpoint (--encoder student); defaults to the "
                         "dino.txt weights for --encoder dinotxt")
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--plain", action="store_true",
                    help="encode bare strings instead of the template ensemble")
    a = ap.parse_args()

    preds = json.load(open(a.vocab))
    assert isinstance(preds, list) and all(isinstance(p, str) for p in preds)
    assert len(set(preds)) == len(preds), "vocabulary has duplicates"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    templates = None if a.plain else PHOTO_TEMPLATES
    print(f"[{a.encoder}] {len(preds):,} predicates x {len(templates or [1])} "
          f"templates on {device}")

    if a.encoder == "dinotxt":
        from training.distill.teacher import encode_texts_dinotxt
        E = encode_texts_dinotxt(preds, a.ckpt or str(DINOTXT_CKPT),
                                 templates=templates, device=device, batch=a.batch)
    else:
        from relsgg.text.student import encode_texts_student
        assert a.ckpt, "--encoder student needs --ckpt"
        E = encode_texts_student(preds, a.ckpt, templates=templates,
                                 device=device, batch=a.batch)
    E = E.numpy()
    assert E.shape[0] == len(preds), (E.shape, len(preds))
    n = np.linalg.norm(E, axis=-1)
    print(f"[{a.encoder}] {E.shape}  norms {n.min():.4f}..{n.max():.4f}")

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.out, embeddings=E.astype(np.float16),
                        predicates=np.array(preds),
                        templates=np.array(templates or []))
    print(f"[{a.encoder}] wrote {a.out} ({Path(a.out).stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
