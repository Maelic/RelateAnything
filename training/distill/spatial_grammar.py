"""Compositional spatial paraphrase grammar for the text-student distillation.

Why this exists
---------------
The distillation corpus is the union of predicate strings that actually occur
in the packs (megasg / vg150 / psg + raw surface forms). Compositional spatial
phrasings never occur there, so the student never learns them — measured on the
live 768-d student, encoding strings absent from the 19,103-predicate vocabulary
and reading off the nearest trained spatial predicate:

    directly below        -> below 0.95, under 0.93        OK
    just underneath       -> underneath 0.95, under 0.94    OK
    on the underside of   -> beside 0.89, inside 0.86       WRONG (misses under)
    diagonally above      -> on 0.82, inside 0.79, above 0.78   WRONG (above 3rd)
    lower than            -> beneath 0.94, beside 0.93      WEAK (polarity leaks)
    north of              -> inside 0.93, on 0.92           WRONG

Since a spatial-only deployment scores `cos(q_spa, w_p)` against frozen text
embeddings, a novel predicate transfers roughly as well as its cosine to a
trained one — so these are text-encoder failures that cap open-vocabulary
spatial before the visual model is involved.

This module emits paraphrases with *known* axis and polarity, so they can be
supervised by construction (oracle pairs) rather than distilled from the
teacher — which is the point: the frozen dino.txt teacher is direction-blind
(auc_syn_vs_inv 0.65) and would re-inject the very failure L_ant removes.

Supervision emitted per generated string g (axis a, polarity p, family f):
  hard synonym  g <-> base forms of (a, p)   when g is a true paraphrase
  soft synonym  g <-> base forms of (a, p)   when g is a modified form that
                                             should stay near but distinguishable
  inverse       g <-> base forms of (a, !p)
  inverse       g <-> generated forms of (a, !p) in the SAME family
The last one is what teaches compositional polarity rather than just base
polarity: "diagonally above" must repel "diagonally below", not only "below".

Holding out a whole family (build_corpus.py --spatial_holdout) removes its
strings from every loss, so it measures compositional generalization rather
than memorization.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

# Base vocabulary per (axis, polarity). First entry is the canonical form.
# Every string here already exists in the megasg spatial set except the
# explicitly-new negatives noted below, so hard-synonym pulls land on
# embeddings the relation head was actually trained against.
AXES: Dict[str, Dict[str, List[str]]] = {
    "vertical": {
        "pos": ["above", "over", "on top of"],
        "neg": ["below", "under", "underneath", "beneath"],
    },
    "lateral": {
        "pos": ["to the left of", "on the left of", "on the left side of"],
        "neg": ["to the right of", "on the right of", "on the right side of"],
    },
    "depth": {
        "pos": ["in front of"],
        "neg": ["behind", "at the back of"],
    },
    "containment": {
        # "outside of" is NOT in the trained spatial set — it enters as a new
        # concept anchored only by its repulsion from "inside".
        "pos": ["inside"],
        "neg": ["outside of"],
    },
    "proximity": {
        # symmetric axis: "near" has no trained antonym, so the negative side
        # is new vocabulary held in place purely by the inverse hinge.
        "pos": ["near", "beside", "next to"],
        "neg": ["far from", "away from"],
    },
}

# family -> (strength, {axis: {polarity: [templates]}}).
# "hard": a true paraphrase, pulled to the base with the strong margin.
# "soft": a modified form — near the base but deliberately distinguishable.
# Templates are literal strings, not "{b}" slots, because English placement of
# a modifier depends on the base ("right behind" but "directly in front of"),
# and nonsense combinations ("far near", "high below") must never be emitted.
FAMILIES: Dict[str, Tuple[str, Dict[str, Dict[str, List[str]]]]] = {
    "precision": ("hard", {
        "vertical": {"pos": ["directly above", "right above", "just above",
                             "immediately above"],
                     "neg": ["directly below", "right below", "just below",
                             "just underneath", "immediately below"]},
        "lateral": {"pos": ["directly to the left of", "just to the left of"],
                    "neg": ["directly to the right of", "just to the right of"]},
        "depth": {"pos": ["directly in front of", "right in front of",
                          "just in front of"],
                  "neg": ["directly behind", "right behind", "just behind"]},
        "containment": {"pos": ["right inside", "just inside"],
                        "neg": ["just outside of", "right outside of"]},
        "proximity": {"pos": ["right next to", "just beside"],
                      "neg": ["well away from"]},
    }),
    "approx": ("hard", {
        "vertical": {"pos": ["slightly above", "roughly above"],
                     "neg": ["slightly below", "roughly below"]},
        "lateral": {"pos": ["slightly to the left of"],
                    "neg": ["slightly to the right of"]},
        "depth": {"pos": ["slightly in front of"], "neg": ["slightly behind"]},
        "containment": {"pos": [], "neg": []},
        "proximity": {"pos": [], "neg": []},
    }),
    "distance": ("soft", {
        "vertical": {"pos": ["far above", "high above", "way above"],
                     "neg": ["far below", "way below", "far underneath"]},
        "lateral": {"pos": ["far to the left of"],
                    "neg": ["far to the right of"]},
        "depth": {"pos": ["far in front of"], "neg": ["far behind"]},
        "containment": {"pos": ["deep inside"], "neg": ["far outside of"]},
        "proximity": {"pos": [], "neg": []},
    }),
    # Surface/partitive phrasings — the family that fails hardest today
    # ("on the underside of" lands on beside/inside).
    "partitive": ("hard", {
        "vertical": {"pos": ["at the top of", "on the top of",
                             "on the upper side of", "on the upper surface of"],
                     "neg": ["at the bottom of", "on the bottom of",
                             "on the underside of", "on the lower side of"]},
        "lateral": {"pos": ["at the left of", "on the left edge of"],
                    "neg": ["at the right of", "on the right edge of"]},
        "depth": {"pos": ["at the front of"], "neg": ["at the rear of"]},
        "containment": {"pos": ["in the middle of", "within"],
                        "neg": ["on the outside of"]},
        "proximity": {"pos": [], "neg": []},
    }),
    # Comparative phrasings — syntactically unlike a preposition, which is why
    # "lower than" currently leaks toward beside/next to.
    "comparative": ("hard", {
        "vertical": {"pos": ["higher than", "taller than"],
                     "neg": ["lower than"]},
        "lateral": {"pos": ["further left than"], "neg": ["further right than"]},
        "depth": {"pos": ["closer to the viewer than", "nearer the camera than"],
                  "neg": ["further back than", "farther from the camera than"]},
        "containment": {"pos": [], "neg": []},
        "proximity": {"pos": ["closer than"], "neg": ["further than"]},
    }),
    # Genuine two-axis composition: soft on the vertical axis it inherits,
    # and repelled from its own mirror image.
    "diagonal": ("soft", {
        "vertical": {"pos": ["diagonally above", "above and to the left of",
                             "above and to the right of"],
                     "neg": ["diagonally below", "below and to the left of",
                             "below and to the right of"]},
        "lateral": {"pos": [], "neg": []},
        "depth": {"pos": [], "neg": []},
        "containment": {"pos": [], "neg": []},
        "proximity": {"pos": [], "neg": []},
    }),
}

OPPOSITE = {"pos": "neg", "neg": "pos"}

# Base forms that are NOT in the megasg spatial set — they enter as new
# concepts held in place only by their inverse hinge. They are excluded from
# the nearest-neighbour pool in eval_spatial_compositional.py: the relation
# head was never trained to score them, so a generated string landing on one
# is not evidence about the axis it would actually be ranked into.
NEW_BASES = frozenset({"outside of", "far from", "away from"})


def generate() -> Tuple[List[dict], List[str]]:
    """Emit the compositional entries plus the base strings they anchor to.

    Returns (entries, base_strings) where each entry is
    ``{"s", "axis", "polarity", "family", "strength"}``.
    """
    entries: List[dict] = []
    seen: set = set()
    for family, (strength, per_axis) in FAMILIES.items():
        for axis, per_pol in per_axis.items():
            for pol, templates in per_pol.items():
                for s in templates:
                    if s in seen:
                        continue
                    seen.add(s)
                    entries.append({"s": s, "axis": axis, "polarity": pol,
                                    "family": family, "strength": strength})
    bases = [b for ax in AXES.values() for pol in ax.values() for b in pol]
    return entries, sorted(set(bases))


def oracle_pairs(index: Dict[str, int]) -> Tuple[list, list, list]:
    """Build (hard_synonym, soft_synonym, inverse) index pairs.

    ``index`` maps string -> corpus index. Strings absent from the corpus are
    skipped, so this is safe to call against any corpus containing a subset.
    """
    entries, _ = generate()
    hard: list = []
    soft: list = []
    inv: list = []

    def add(dst: list, a: str, b: str) -> None:
        ia, ib = index.get(a), index.get(b)
        if ia is not None and ib is not None and ia != ib:
            dst.append([ia, ib])

    for e in entries:
        same = AXES[e["axis"]][e["polarity"]]
        opp = AXES[e["axis"]][OPPOSITE[e["polarity"]]]
        bucket = hard if e["strength"] == "hard" else soft
        for b in same:
            add(bucket, e["s"], b)
        for b in opp:
            add(inv, e["s"], b)

    # Compositional polarity: same family + same axis, opposite polarity.
    by_key: Dict[tuple, List[str]] = {}
    for e in entries:
        by_key.setdefault((e["family"], e["axis"], e["polarity"]), []).append(e["s"])
    for (fam, axis, pol), group in by_key.items():
        for other in by_key.get((fam, axis, OPPOSITE[pol]), []):
            for s in group:
                add(inv, s, other)

    dedup = lambda ps: [list(t) for t in sorted({tuple(sorted(p)) for p in ps})]
    return dedup(hard), dedup(soft), dedup(inv)


if __name__ == "__main__":
    ents, bases = generate()
    print(f"{len(ents)} generated strings over {len(bases)} base forms")
    for fam in FAMILIES:
        n = sum(1 for e in ents if e["family"] == fam)
        print(f"  {fam:<12} {n:3d}  ({FAMILIES[fam][0]})")
    idx = {s: i for i, s in enumerate(sorted({e["s"] for e in ents} | set(bases)))}
    h, s, v = oracle_pairs(idx)
    print(f"pairs: {len(h)} hard-synonym, {len(s)} soft-synonym, {len(v)} inverse")
