#!/usr/bin/env python3
"""Place phrases inside a Whisper transcript's 30-40s passages.

A Whisper-only meeting (Oct 6 2026: YouTube published no captions) caches
`.cache/transcript-<video>.json` as coarse passages, so a link to a passage's
start can open half a minute before the words. This interpolates by character
position within the passage. Checked against the video on Oct 6: 16:40 vs the
maintainer's 16:42, 18:33 vs 18:32. Weakest where a passage is sparse (a pause).

    python scripts/whisper_seek.py .cache/transcript-5v3reZJ1sfE.json < phrases.txt

Each stdin line is `<min-seconds>|<phrase>`; the first passage at or after
min-seconds containing the phrase wins. Subtract a couple of seconds before
linking so the first word isn't clipped.
"""

import json
import re
import sys


def norm(s):
    s = re.sub(r"\s+", " ", s.lower().replace("’", "'"))
    return re.sub(r"[^a-z0-9 ]", "", s)


def locate(passages, phrase, after=0.0):
    p = norm(phrase)
    for c in passages:
        if c["start"] < after:
            continue
        t = norm(c["text"])
        i = t.find(p)
        if i >= 0:
            return c["start"] + c["duration"] * i / len(t), c["start"]
    return None, None


def fmt(s):
    s = int(s)
    h, m, x = s // 3600, s % 3600 // 60, s % 60
    return f"{h}:{m:02d}:{x:02d}" if h else f"{m}:{x:02d}"


def main():
    passages = json.load(open(sys.argv[1]))
    for line in sys.stdin:
        line = line.rstrip("\n")
        if not line:
            continue
        after, phrase = line.split("|", 1)
        t, c = locate(passages, phrase, float(after))
        if t is None:
            print(f"    MISS       -  {phrase[:60]}")
        else:
            print(f"{fmt(t):>8} {int(t):>6}  passage {fmt(c):>8}  {phrase[:60]}")


if __name__ == "__main__":
    main()
