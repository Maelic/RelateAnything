#!/usr/bin/env python3
"""Collect openly-licensed video clips for the demo reel, with their credits.

Wikimedia Commons is the only source here because it is the only one that hands
back a machine-readable license per file. Stock sites (Pexels, Pixabay, Mixkit)
are free to use but ship no per-asset license record, so a credits file for them
has to be written by hand and cannot be re-derived later.

    python deploy/fetch_footage.py --out footage/              # default queries
    python deploy/fetch_footage.py --out footage/ --limit 40
    python deploy/fetch_footage.py --out footage/ --query "man riding bicycle"

Writes <out>/credits.json in the same shape the project page already uses for
assets/video/credits.json, so the reel's credits are generated, never typed.

NoDerivatives licenses are rejected: an overlay is a derivative work. Which is
also why NonCommercial is rejected by default -- a project page that carries a
paper, code and a model is not obviously non-commercial, and the argument is not
worth having. Pass --allow_nc if you disagree.
"""
from __future__ import annotations

import argparse
import html
import json
import os
import re
import sys
import urllib.parse
import urllib.request
from typing import Sequence

API = "https://commons.wikimedia.org/w/api.php"
UA = "RelateAnything-demo-footage/1.0 (https://github.com/Maelic/RelateAnything)"

# Queries chosen from the interaction predicates the model actually emits --
# riding, holding, carrying, playing, using, sitting on -- rather than from what
# makes a pretty clip. A clip with no interaction in it scores nothing later.
QUERIES = [
    "man riding bicycle", "woman riding bicycle", "cyclist street traffic",
    "skateboarder skatepark", "person riding horse", "person riding motorcycle",
    "chef cooking kitchen", "person cooking food preparation",
    "man playing guitar", "person playing musical instrument",
    "child playing playground", "dog playing park", "person walking dog",
    "people sitting cafe table", "market vendor stall customer",
    "person carrying bag street", "worker using tool", "street food cooking",
    "tennis player match", "football player match", "person reading book",
    "family dinner table eating", "person using laptop desk",
]

STRIP_TAGS = re.compile(r"<[^>]+>")


def _get(params: dict) -> dict:
    url = API + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


def _text(extmeta: dict, key: str) -> str:
    v = extmeta.get(key, {}).get("value", "") or ""
    return html.unescape(STRIP_TAGS.sub(" ", str(v))).strip()


def license_ok(slug: str, allow_nc: bool) -> bool:
    """Reject ND outright, NC unless asked. Parse components, never substrings:
    'cc-by-sa-4.0' has no 'nd' in it but a naive `'nd' in slug` says it does."""
    s = (slug or "").lower().strip()
    if not s:
        return False
    if s in ("pd", "cc0"):
        return True
    parts = re.split(r"[-_ ]", s)
    if "nd" in parts:
        return False
    if "nc" in parts and not allow_nc:
        return False
    return s.startswith("cc") or s.startswith("pd")


def by_titles(titles: Sequence[str]) -> list[dict]:
    """Look clips up by their Commons File: title instead of by search.

    This is what makes a shot list reproducible: `reel_shots.json` records the
    Commons page of every clip it cuts, so the footage can be re-materialised
    exactly rather than re-discovered by a query that may rank differently the
    next time it runs.
    """
    out = []
    for i in range(0, len(titles), 20):            # the API caps a titles batch
        try:
            d = _get({
                "action": "query", "format": "json",
                "titles": "|".join(titles[i:i + 20]), "prop": "imageinfo",
                "iiprop": "url|size|mime|mediatype|extmetadata",
            })
        except Exception as e:
            print(f"  ! lookup failed ({e})")
            continue
        for page in d.get("query", {}).get("pages", {}).values():
            if "missing" in page:
                print(f"    MISSING {page.get('title', '?')}")
                continue
            out.append(page)
    return out


def titles_from_shots(path: str) -> list[str]:
    """Pull File: titles out of a shot list's `source` URLs."""
    titles = []
    for sh in json.load(open(path)):
        u = sh.get("source", "")
        m = re.search(r"/wiki/(File:.+)$", u)
        if m:
            titles.append(urllib.parse.unquote(m.group(1)))
        else:
            print(f"    no Commons source on shot {sh.get('file', '?')}")
    return titles


def search(query: str, limit: int) -> list[dict]:
    try:
        d = _get({
            "action": "query", "format": "json", "generator": "search",
            "gsrsearch": f"filetype:video {query}", "gsrlimit": str(limit),
            "gsrnamespace": "6", "prop": "imageinfo",
            "iiprop": "url|size|mime|mediatype|extmetadata",
        })
    except Exception as e:
        print(f"  ! search failed ({e})")
        return []
    return list(d.get("query", {}).get("pages", {}).values())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="footage")
    ap.add_argument("--query", nargs="*", default=None, help="override the query list")
    ap.add_argument("--from_shots", default="",
                    help="a shot list (e.g. deploy/reel_shots.json): re-fetch "
                         "exactly the clips it cuts, by their Commons page")
    ap.add_argument("--per_query", type=int, default=8)
    ap.add_argument("--limit", type=int, default=30, help="max clips to download")
    ap.add_argument("--min_width", type=int, default=1280)
    ap.add_argument("--min_dur", type=float, default=3.0)
    ap.add_argument("--max_dur", type=float, default=180.0)
    ap.add_argument("--max_mb", type=float, default=120.0)
    ap.add_argument("--allow_nc", action="store_true")
    ap.add_argument("--dry_run", action="store_true", help="list, do not download")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    seen: dict[str, dict] = {}
    if args.from_shots:
        titles = titles_from_shots(args.from_shots)
        print(f"[shots] {len(titles)} clips named by {args.from_shots}")
        jobs = [("(shot list)", by_titles(titles))]
    else:
        jobs = [(q, None) for q in (args.query if args.query else QUERIES)]

    for q, pages in jobs:
        if pages is None:
            print(f"[search] {q}")
            pages = search(q, args.per_query)
        for page in pages:
            ii = (page.get("imageinfo") or [{}])[0]
            if ii.get("mediatype") != "VIDEO":
                continue
            em = ii.get("extmetadata", {})
            slug = em.get("License", {}).get("value", "")
            short = _text(em, "LicenseShortName")
            w, h = ii.get("width", 0), ii.get("height", 0)
            dur = float(ii.get("duration") or 0.0)
            mb = (ii.get("size") or 0) / 1e6
            title = page["title"]
            if title in seen:
                continue
            why = None
            if args.from_shots:
                pass                     # named explicitly; it already passed
            elif not license_ok(slug, args.allow_nc):
                why = f"license {slug or short or '?'}"
            elif _text(em, "Restrictions"):
                why = f"restricted: {_text(em,'Restrictions')[:40]}"
            elif w < args.min_width:
                why = f"{w}x{h}"
            elif not (args.min_dur <= dur <= args.max_dur):
                why = f"{dur:.0f}s"
            elif mb > args.max_mb:
                why = f"{mb:.0f}MB"
            elif ii.get("mime") not in ("video/webm", "video/mp4"):
                why = ii.get("mime", "?")
            if why:
                print(f"    skip {title[5:45]:45s} {why}")
                continue
            seen[title] = {
                "title": title[len("File:"):],
                "query": q,
                "url": ii["url"],
                "page": ii["descriptionurl"],
                "artist": _text(em, "Artist") or "(not stated)",
                "license": short or slug,
                "license_slug": slug,
                "license_url": em.get("LicenseUrl", {}).get("value", "") or "",
                "w": w, "h": h, "duration": round(dur, 2), "mb": round(mb, 1),
                "mime": ii.get("mime", ""),
            }
            print(f"    KEEP {title[5:45]:45s} {w}x{h} {dur:5.1f}s "
                  f"{mb:5.1f}MB  {short or slug}")

    cands = sorted(seen.values(), key=lambda c: (-c["w"], c["mb"]))[:args.limit]
    print(f"\n[fetch] {len(seen)} candidates, keeping {len(cands)}")
    if args.dry_run:
        return

    credits = []
    for i, c in enumerate(cands):
        # Extension from the MIME type, never from the URL: Commons appends a
        # utm query string to the original-file link, so `endswith(".webm")` is
        # False for every webm it serves and everything lands named .mp4.
        ext = {"video/webm": ".webm", "video/mp4": ".mp4"}.get(c["mime"], ".webm")
        # Filename from the Commons title, not a counter: the clip stays
        # traceable to its source page after it has been copied around.
        stem = re.sub(r"[^a-z0-9]+", "_",
                      os.path.splitext(c["title"])[0].lower()).strip("_")[:48]
        fn = f"{stem}{ext}"
        dst = os.path.join(args.out, fn)
        if not os.path.exists(dst):
            print(f"  [{i+1}/{len(cands)}] {fn}  ({c['mb']} MB)")
            try:
                req = urllib.request.Request(c["url"], headers={"User-Agent": UA})
                with urllib.request.urlopen(req, timeout=300) as r, \
                     open(dst, "wb") as f:
                    while chunk := r.read(1 << 20):
                        f.write(chunk)
            except Exception as e:
                print(f"      ! download failed: {e}")
                continue
        credits.append({
            "file": fn, "title": c["title"], "artist": c["artist"],
            "license": c["license"], "license_url": c["license_url"],
            "page": c["page"], "w": c["w"], "h": c["h"], "mime": c["mime"],
            "duration": c["duration"], "query": c["query"],
        })

    # Merge, never overwrite. A footage directory is normally built up over
    # several runs with different queries, and rewriting this file each time
    # silently drops the attribution for everything fetched earlier -- which
    # for CC BY material is a licensing failure, not an inconvenience.
    out = os.path.join(args.out, "credits.json")
    merged: dict[str, dict] = {}
    if os.path.exists(out):
        try:
            for c in json.load(open(out)):
                merged[c["file"]] = c
        except Exception as e:
            print(f"  ! existing credits unreadable, starting fresh ({e})")
    for c in credits:
        merged[c["file"]] = c
    # Drop entries whose clip is no longer on disk, so the file describes the
    # directory rather than accumulating ghosts.
    merged = {k: v for k, v in merged.items()
              if os.path.exists(os.path.join(args.out, k))}
    with open(out, "w") as f:
        json.dump(sorted(merged.values(), key=lambda c: c["file"]), f, indent=1)
    print(f"\n[fetch] {len(credits)} clips this run, {len(merged)} in {args.out}"
          f"\n[fetch] credits -> {out}")


if __name__ == "__main__":
    main()
