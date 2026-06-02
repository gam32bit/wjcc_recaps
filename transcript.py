#!/usr/bin/env python3
"""Fetch a timed transcript from a YouTube URL and cache it locally.

Usage:
    python transcript.py https://www.youtube.com/live/Y-LpLQm27AI
    python transcript.py <url> --cache .cache
"""

import argparse
import json
import pathlib
import re
import sys

CACHE_DIR = pathlib.Path(__file__).resolve().parent / ".cache"

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
