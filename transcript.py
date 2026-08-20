#!/usr/bin/env python3
"""Fetch a timed transcript from a YouTube URL and cache it locally.

Usage:
    python transcript.py https://www.youtube.com/live/Y-LpLQm27AI
    python transcript.py <url> --cache .cache
"""

import argparse
import bisect
import json
import pathlib
import re
import sys

PROJECT_DIR = pathlib.Path(__file__).resolve().parent
CACHE_DIR = PROJECT_DIR / ".cache"
OUT_DIR = PROJECT_DIR / "out"

_VIDEO_ID_RE = re.compile(
    r"(?:youtube\.com/(?:watch\?v=|live/)|youtu\.be/)([A-Za-z0-9_-]{11})"
)


def extract_video_id(url: str) -> str | None:
    url = re.sub(r"[?&]si=[^&]*", "", url)  # strip ?si= tracking param
    m = _VIDEO_ID_RE.search(url)
    return m.group(1) if m else None


def fetch_transcript(
    youtube_url: str,
    *,
    cache_dir: pathlib.Path = CACHE_DIR,
) -> list[dict]:
    """Fetch timed transcript snippets for a YouTube URL.

    Returns a list of {text, start, duration} dicts (start in seconds).
    Caches raw result to cache_dir/transcript-{video_id}.json.

    On failure (no captions, blocked): prints a warning and returns [].
    Pass the result to score.segment_transcript() for discussion-time scoring.
    """
    video_id = extract_video_id(youtube_url)
    if not video_id:
        print(
            f"  WARNING: could not extract video ID from {youtube_url!r}",
            file=sys.stderr,
        )
        return []

    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"transcript-{video_id}.json"

    if cache_path.is_file():
        data = json.loads(cache_path.read_text())
        print(f"  Transcript loaded from cache: {cache_path.name} ({len(data)} snippets)")
        return data

    try:
        from youtube_transcript_api import YouTubeTranscriptApi

        fetched = YouTubeTranscriptApi().fetch(video_id)
        snippets = [
            {"text": s.text, "start": s.start, "duration": s.duration}
            for s in fetched
        ]
        cache_path.write_text(json.dumps(snippets, indent=2))
        print(f"  Fetched {len(snippets)} transcript snippets -> {cache_path.name}")
        return snippets
    except Exception as exc:
        print(
            f"  WARNING: transcript fetch failed for {video_id}: {exc}\n"
            "  Discussion signal will be unavailable. "
            "Use --transcript <file> to supply manually.",
            file=sys.stderr,
        )
        return []


def _hms(seconds: float) -> str:
    """Format seconds as H:MM:SS (or M:SS under an hour)."""
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


# --- Deterministic cleanup -------------------------------------------------
# Auto-captions arrive as noisy fragments: YouTube marks speaker changes with
# `>>`, drops in bracketed sound cues ([Music]), and (especially on older
# videos) is thick with "um"/"uh" disfluencies. We tidy ONLY this mechanical
# noise — never real words — because the rendered file is the reviewer's
# verification surface for seeking the video against the recap's quotes.

_TURN_RE = re.compile(r"\s*>>+\s*")
_BRACKET_RE = re.compile(r"\[[^\]]*\]")
_FILLER_RE = re.compile(r"\b(?:um+|uh+|erm+|hmm+)\b,?", re.IGNORECASE)
_STUTTER_RE = re.compile(r"\b(\w+)(?:\s+\1\b){2,}", re.IGNORECASE)
_SPACE_BEFORE_PUNCT_RE = re.compile(r"\s+([,.;:!?])")
_WS_RE = re.compile(r"\s+")
_SENTENCE_START_RE = re.compile(r"(^|[.?!]\s+)([a-z])")


def clean_caption_text(text: str) -> str:
    """Strip mechanical caption noise from a chunk of text, preserving words.

    Removes bracketed sound cues, standalone filler words (um/uh/erm/hmm), and
    3+-word stutters ("the the the" -> "the"), then normalizes whitespace.
    Deliberately leaves doubled words alone — "had had" is usually real, and
    this file must stay a faithful transcript a reviewer can quote-check.
    """
    text = text.replace("\n", " ")
    text = _BRACKET_RE.sub(" ", text)
    text = _FILLER_RE.sub(" ", text)
    text = _STUTTER_RE.sub(r"\1", text)
    text = _SPACE_BEFORE_PUNCT_RE.sub(r"\1", text)
    return _WS_RE.sub(" ", text).strip()


def _capitalize_sentences(text: str) -> str:
    """Uppercase the chunk's first letter and each letter after .?! — cosmetic.

    Repairs lowercase sentence starts left by filler removal or by ASR that
    never capitalized in the first place. Purely for scannability.
    """
    return _SENTENCE_START_RE.sub(lambda m: m.group(1) + m.group(2).upper(), text)


def transcript_to_markdown(
    snippets: list[dict],
    title: str,
    *,
    sections: list[tuple[float, str]] | None = None,
) -> str:
    """Render snippets as a cleaned, timestamped, speaker-segmented Markdown doc.

    Captions are cleaned (see clean_caption_text), then split into paragraphs at
    each `>>` speaker change — long turns are further capped at ~30 s — so a
    reviewer can skim who-said-what and seek the video by timestamp.

    `sections` is an optional list of (start_seconds, label) anchors — typically
    each agenda item's earliest appearance from segmentation. A `## label`
    heading is inserted before the paragraph containing each anchor, turning the
    file into a topic-segmented map the reviewer can scan and jump through.
    """
    lines = [f"# {title}", ""]
    if not snippets:
        lines += ["_No transcript available._", ""]
        return "\n".join(lines)

    # Split the stream into speaker turns, tagging each fragment with the start
    # time of the snippet it came from. A `>>` splits one snippet into a
    # continuation (index 0) followed by one or more new turns (index > 0).
    tokens: list[tuple[float, str, bool]] = []  # (start, text, is_turn_start)
    for s in snippets:
        for i, part in enumerate(_TURN_RE.split(s["text"])):
            cleaned = clean_caption_text(part)
            if cleaned:
                tokens.append((s["start"], cleaned, i > 0))

    WINDOW = 30.0    # cap a single speaker's turn so long speeches stay scannable
    MIN_TURN = 5.0   # rapid sub-5s turns (e.g. roll-call) merge into one block
    paragraphs: list[tuple[float, str]] = []  # (para_start, rendered_text)
    buf: list[str] = []
    para_start = 0.0

    def flush() -> None:
        if buf:
            paragraphs.append((para_start, _capitalize_sentences(" ".join(buf))))

    for start, text, turn_start in tokens:
        if buf:
            gap = start - para_start
            if gap >= WINDOW or (turn_start and gap >= MIN_TURN):
                flush()
                buf = []
            elif turn_start:
                buf.append("—")  # speaker change kept visible within a block
        if not buf:
            para_start = start
        buf.append(text)
    flush()

    # Assign each section anchor to the paragraph that contains it (the last
    # paragraph starting at or before the anchor) so headings land in place.
    headings_before: dict[int, list[str]] = {}
    if sections and paragraphs:
        starts = [p[0] for p in paragraphs]
        for anchor, label in sorted(sections):
            idx = max(0, bisect.bisect_right(starts, anchor) - 1)
            headings_before.setdefault(idx, []).append(label)

    for i, (pstart, text) in enumerate(paragraphs):
        for label in headings_before.get(i, []):
            lines += [f"## {label}", ""]
        lines += [f"**[{_hms(pstart)}]** " + text, ""]
    return "\n".join(lines)


# --- Segmented-transcript output (shared by the orchestrators) -------------

def transcript_sections(scored: list, attr: str) -> list[tuple[float, str]]:
    """Build (start_seconds, label) heading anchors from segmentation results.

    `attr` is the Signals field holding the deep-link anchor for the transcript
    being rendered ("meeting_start_seconds" / "work_session_start_seconds").
    Items the model never located in the video have no anchor and get no
    heading.
    """
    sections: list[tuple[float, str]] = []
    for s in scored:
        anchor = getattr(s.signals, attr)
        if anchor is not None:
            sections.append((float(anchor), f"[{s.item.number or '—'}] {s.item.title}"))
    sections.sort()
    return sections


def write_transcript_md(
    date: str,
    label: str,
    snippets: list[dict],
    sections: list[tuple[float, str]] | None = None,
    *,
    out_dir: pathlib.Path = OUT_DIR,
) -> pathlib.Path:
    """Save a readable, timestamped, agenda-segmented Markdown copy of a transcript."""
    titles = {"meeting": "Meeting transcript", "worksession": "Work session transcript"}
    title = f"{titles.get(label, 'Transcript')} — {date}"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"transcript-{label}-{date}.md"
    path.write_text(transcript_to_markdown(snippets, title, sections=sections))
    extra = f", {len(sections)} sections" if sections else ""
    print(f"  Wrote {path.relative_to(PROJECT_DIR)} ({len(snippets)} snippets{extra})")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("url", help="YouTube URL")
    parser.add_argument(
        "--cache",
        type=pathlib.Path,
        default=CACHE_DIR,
        help="cache directory (default: .cache/)",
    )
    args = parser.parse_args()

    snippets = fetch_transcript(args.url, cache_dir=args.cache)
    if not snippets:
        sys.exit("No transcript available.")
    duration = snippets[-1]["start"] + snippets[-1]["duration"]
    print(f"{len(snippets)} snippets, {duration / 60:.1f} min of transcript")
    print(f"First: [{snippets[0]['start']:.0f}s] {snippets[0]['text']!r}")
    print(f"Last:  [{snippets[-1]['start']:.0f}s] {snippets[-1]['text']!r}")


if __name__ == "__main__":
    main()
