#!/usr/bin/env python3
"""
sgg_canon.py — predicate canonicalization for SGG annotation output.

Calibration (2026-06-01) showed the caption_first strategy's apparent
diversity (955 unique predicates, entropy 5.42) was ~70% *lexical noise*:
the text-only parser faithfully copies the observer's verbose spatial prose,
producing dozens of synonyms for the same relation —

    near / positioned near / located near / stands near / placed near …
    in front of / positioned in front of / situated in front of …

After collapsing those, real diversity drops to 820 unique / entropy 4.88,
and **27% of all caption_first relations are pure low-information spatial
filler** (vs 3% for single). canonicalize() fixes three things at once:

  1. strips position *scaffolding* ("positioned/located/situated/placed …")
     while KEEPING real postures ("sitting on", "standing behind"),
  2. collapses directional spatial synonyms to a canonical surface form,
  3. DROPS vague-proximity predicates ("near", "next to", "beside" …) that
     carry no usable layout signal — these are the iter_19 BANNED list.

It returns the canonical predicate, or None to signal "drop this relation".
Apply via the --canonicalize flag in sgg_vllm_generate.py; the raw predicate
is preserved in `predicate_raw` so nothing is lost irreversibly.
"""
from __future__ import annotations

import re

# Vague proximity → no usable layout signal → DROP (mirrors prompt_eval FORBIDDEN_PREDS,
# extended with the synonyms the caption-first parser actually emits).
_DROP = {
    "near", "next to", "beside", "by", "close to", "alongside", "nearby",
    "adjacent to", "with", "has", "and", "same scene", "overlapping",
    "associated", "related", "interacts", "involves", "features", "appears",
    "around", "at", "in the background", "in the foreground", "visible",
    "present", "there",
    # has-synonyms observed in generation ("man possessing human face")
    "possessing", "possesses", "belongs to", "belonging to", "owns", "owning",
    # bare position scaffolding with nothing after it → no relation at all
    "positioned", "located", "situated", "placed", "mounted", "stationed",
    "arranged",
}

# Position scaffolding with no semantic content beyond "is somewhere" — strip the
# prefix, keep whatever spatial preposition follows. "positioned in front of" →
# "in front of"; "located behind" → "behind". A bare "positioned"/"located" with
# no following prep collapses to "" and is dropped.
# NOT scaffold: "resting/seated/mounted" carry visible posture/support content —
# "resting on", "seated on"(→sitting on), "mounted on" stay distinct surface forms
# (open-vocab diversity); all three are covered by the contact gate.
_SCAFFOLD = re.compile(
    r"^(?:positioned|located|situated|placed|set up|"
    r"stationed|arranged)\s+(?=(?:in|on|at|to|behind|above|below|"
    r"beneath|under|over|near|next|beside|close|adjacent|alongside|between|"
    r"inside|within|atop|"
    r"directly|closely|just|slightly|vertically|horizontally|centrally))",
)
_ADVERB = re.compile(
    r"^(?:directly|closely|just|slightly|vertically|horizontally|centrally|"
    r"partially|right|further|slightly)\s+")
_LEADING_COPULA = re.compile(r"^(?:is|are|was|were|be|being|been)\s+", re.I)
_WS = re.compile(r"\s+")

# Directional spatial synonyms → canonical surface form (these are KEPT — layout
# between different categories is informative; only same-category should be pruned
# upstream by the prompt).
_SPATIAL = {
    "in front of": ["in front of", "ahead of", "before", "to the front of",
                    "in front", "infront of"],
    "behind":      ["behind", "to the rear of", "at the back of"],
    "above":       ["above", "atop", "on top of", "over", "overhead", "on top"],
    "below":       ["below", "beneath", "underneath", "under"],
    "to the left of":  ["to the left of", "left of", "on the left side of",
                        "on the left of"],
    "to the right of": ["to the right of", "right of", "on the right side of",
                        "on the right of"],
    "between":     ["between", "in between", "in the middle of", "amid"],
    "inside":      ["inside", "within", "in"],
}
_SPATIAL_LOOKUP = {v: k for k, vs in _SPATIAL.items() for v in vs}

# Proximity surface forms. canonicalize() DROPS these (in a semantic pass they are
# low-information filler crowding out real predicates), but the dedicated spatial
# round emits them deliberately and geometry VERIFIES them — there they are layout
# signal, exposed via canonicalize_spatial() with canonical class "near".
_PROXIMITY = frozenset({
    "near", "close to", "next to", "beside", "adjacent to", "alongside",
    "nearby", "by",
})

# Present-tense / posture synonyms → canonical gerund, so the caption_first
# parser ("sits on", "holds") and the single prompt ("sitting on", "holding")
# share one vocabulary.
_VERB = {
    "sits": "sitting", "sit": "sitting", "seated": "sitting", "sat": "sitting",
    "stands": "standing", "stand": "standing", "stood": "standing",
    "holds": "holding", "hold": "holding", "held": "holding",
    "hangs": "hanging", "hang": "hanging", "hung": "hanging",
    "rests": "resting", "rest": "resting", "rested": "resting",
    "lies": "lying", "lie": "lying", "lay": "lying",
    "wears": "wearing", "wear": "wearing", "worn": "wearing",
    "carries": "carrying", "carry": "carrying",
    "rides": "riding", "ride": "riding",
    "leans": "leaning", "lean": "leaning",
    "covers": "covering", "cover": "covering",
    "faces": "facing", "face": "facing",
    "grows": "growing", "grow": "growing",
    "walks": "walking", "walk": "walking",
    "looks": "looking", "look": "looking",
    "watches": "watching", "watch": "watching",
}


def _normalize_first_verb(s: str) -> str:
    parts = s.split(" ", 1)
    # Don't touch passives ("worn by", "held by") — normalizing the verb would
    # corrupt them ("wearing by"); they should be inverted upstream, not here.
    if len(parts) > 1 and parts[1].startswith("by"):
        return s
    head = _VERB.get(parts[0])
    if head:
        return head + ((" " + parts[1]) if len(parts) > 1 else "")
    return s


def _clean(pred: str) -> str:
    """Lexical cleanup shared by canonicalize()/canonicalize_spatial():
    lowercase, strip copula, peel scaffolding + adverbs (possibly several layers)."""
    s = str(pred).strip().lower().replace("_", " ")
    s = _LEADING_COPULA.sub("", s)
    s = _WS.sub(" ", s).strip(".,-")
    for _ in range(3):
        new = _SCAFFOLD.sub("", s)
        new = _ADVERB.sub("", new)
        if new == s:
            break
        s = new.strip()
    return s


def canonicalize(pred: str, collapse_spatial: bool = True) -> str | None:
    """Canonical predicate, or None to drop the relation entirely.

    collapse_spatial=False keeps the model's spatial surface form ("atop",
    "on top of", "underneath" stay distinct) instead of mapping synonyms to one
    canonical token — for open-vocabulary training the synonym spread is signal,
    not noise (language-to-vision latent coverage). Scaffolding stripping and
    vague-proximity drops still apply; use collapse_spatial=True internally when
    LOGIC needs one key per spatial concept (dedup, contradictions, caps)."""
    if not pred:
        return None
    s = _clean(pred)
    if not s or s in _DROP:
        return None
    if s in _SPATIAL_LOOKUP:
        return _SPATIAL_LOOKUP[s] if collapse_spatial else s
    s = _normalize_first_verb(s)
    if s in _DROP:
        return None
    return s


def canonicalize_spatial(pred: str, collapse_spatial: bool = True) -> str | None:
    """canonicalize() for the DEDICATED spatial-layout layer (llm_v2 round,
    geometric layer, postprocess of `spatial`-tagged relations): geometry-verified
    proximity is layout signal there, not filler, so "near / next to / beside …"
    map to canonical "near" instead of being dropped. Everything else behaves
    exactly like canonicalize()."""
    if not pred:
        return None
    s = _clean(pred)
    if s in _PROXIMITY:
        return "near" if collapse_spatial else s
    return canonicalize(pred, collapse_spatial)


# Quick self-check.
if __name__ == "__main__":
    cases = {
        "positioned near": None, "near": None, "next to": None,
        "positioned in front of": "in front of", "located behind": "behind",
        "sits on": "sitting on", "holds": "holding", "stands behind": "standing behind",
        "to the left of": "to the left of", "on top of": "above",
        "wearing": "wearing", "grazing on": "grazing on", "positioned": None,
    }
    ok = True
    for inp, exp in cases.items():
        got = canonicalize(inp)
        flag = "✓" if got == exp else "✗"
        if got != exp:
            ok = False
        print(f"  {flag} {inp!r:30} → {got!r:20} (expected {exp!r})")
    # collapse_spatial=False keeps surface forms but still strips scaffolding
    cases_keep = {
        "atop": "atop", "on top of": "on top of", "underneath": "underneath",
        "positioned in front of": "in front of", "near": None,
        "sits on": "sitting on",
    }
    for inp, exp in cases_keep.items():
        got = canonicalize(inp, collapse_spatial=False)
        flag = "✓" if got == exp else "✗"
        if got != exp:
            ok = False
        print(f"  {flag} keep {inp!r:25} → {got!r:20} (expected {exp!r})")
    # canonicalize_spatial: proximity survives (canonical "near" / surface kept)
    cases_spa = {
        ("near", True): "near", ("next to", True): "near",
        ("next to", False): "next to", ("positioned close to", False): "close to",
        ("beside", False): "beside", ("on top of", True): "above",
        ("on top of", False): "on top of", ("wearing", True): "wearing",
        ("has", True): None,
    }
    for (inp, coll), exp in cases_spa.items():
        got = canonicalize_spatial(inp, collapse_spatial=coll)
        flag = "✓" if got == exp else "✗"
        if got != exp:
            ok = False
        print(f"  {flag} spatial({coll}) {inp!r:22} → {got!r:20} (expected {exp!r})")
    print("OK" if ok else "FAILURES")
