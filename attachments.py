#!/usr/bin/env python3
"""Fetch and cache BoardDocs attachment PDFs.

Requires the warmed session from pull_agenda.new_session() to pass the
CloudFront WAF. Files are cached by a hash of their URL so re-running the
pipeline never re-downloads.

Usage (standalone):
    python attachments.py https://go.boarddocs.com/vsba/wjcc/Board.nsf/files/...
"""

import argparse
import hashlib
import pathlib
import sys

import requests

CACHE_DIR = pathlib.Path(__file__).resolve().parent / ".cache"


def fetch_attachment(
    url: str,
    session: requests.Session,
    *,
    cache_dir: pathlib.Path = CACHE_DIR,
) -> pathlib.Path:
    """Download a BoardDocs attachment and cache it locally. Returns the path.

    Uses a 12-char URL hash as the filename stem so the cache is stable across
    runs without needing to parse BoardDocs' opaque URL structure.
    """
    url_hash = hashlib.md5(url.encode()).hexdigest()[:12]
    suffix = pathlib.PurePosixPath(url.split("?")[0]).suffix or ".pdf"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"attachment-{url_hash}{suffix}"

    if cache_path.is_file():
        print(f"  Attachment cached: {cache_path.name}")
        return cache_path

    resp = session.get(url)
    resp.raise_for_status()
    cache_path.write_bytes(resp.content)
    print(f"  Downloaded {cache_path.name} ({len(resp.content):,} bytes)")
    return cache_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("url", help="BoardDocs attachment URL")
    args = parser.parse_args()

    from pull_agenda import new_session

    session = new_session()
    path = fetch_attachment(args.url, session)
    print(f"Saved to {path}")


if __name__ == "__main__":
    main()
