#!/usr/bin/env python3
"""Orchestrate the WJCC agenda -> newsletter pipeline.

Pipeline (Phase 3 design):
    pull_agenda -> parse -> triage -> score -> [checkpoint] -> write -> render

A full interactive run:
  1. parse + triage (offline)
  2. fetch the work-session transcript (--work-session URL)
  3. score: segmentation call + rubric call -> ranked table
  4. CHECKPOINT: review/adjust the ranking and select attachments to fetch
  5. fetch approved PDFs
  6. write: prose-drafting call
  7. render -> out/newsletter-<date>.md

--dry-run stops after step 3 (deterministic signals only, no API calls).

Usage:
    python make_newsletter.py --date 20260519 --dry-run
    python make_newsletter.py --date 20260519 \\
        --work-session https://www.youtube.com/live/Y-LpLQm27AI \\
        --livestream   https://www.youtube.com/live/3np-qU3BSnQ
"""

import argparse
import dataclasses
import json
import os
import pathlib
import sys

import anthropic

from parse import agenda_preamble, parse_agenda, to_dict
from render import render_newsletter
from score import (
    ScoredItem,
    compute_deterministic,
    evidence_line,
    finalize,
    score_rubric,
    segment_transcript,
    _print_table,
)
from transcript import fetch_transcript
from triage import Logistics, kept_items, print_report, triage
from write import Draft, draft, validate_draft

PROJECT_DIR = pathlib.Path(__file__).resolve().parent
FIXTURES_DIR = PROJECT_DIR / "wjcc-fixtures"
CACHE_DIR    = PROJECT_DIR / ".cache"
OUT_DIR      = PROJECT_DIR / "out"


# --- Fixture discovery -----------------------------------------------------

def _discover_fixtures() -> dict[str, tuple[pathlib.Path, pathlib.Path]]:
    fixtures: dict[str, tuple[pathlib.Path, pathlib.Path]] = {}
    for meta_path in FIXTURES_DIR.glob("*meta*.json"):
        html_path = meta_path.with_name(
            meta_path.name.replace("meeting-meta", "agenda").replace(".json", ".html")
        )
        if not html_path.is_file():
            continue
        try:
            numberdate = json.loads(meta_path.read_text()).get("numberdate", "")
        except json.JSONDecodeError:
            continue
        if numberdate:
            fixtures[numberdate] = (html_path, meta_path)
    return fixtures


def _select_fixture(date: str | None) -> tuple[str, pathlib.Path, pathlib.Path]:
    fixtures = _discover_fixtures()
    if not fixtures:
        raise SystemExit(f"No agenda fixtures found in {FIXTURES_DIR}.")
    if date:
        if date not in fixtures:
            available = ", ".join(sorted(fixtures)) or "none"
            raise SystemExit(f"No fixture for {date}. Available: {available}")
        chosen = date
    else:
        chosen = max(fixtures)
    html_path, meta_path = fixtures[chosen]
    return chosen, html_path, meta_path


# --- Checkpoint UI ---------------------------------------------------------

def _print_checkpoint(scored: list[ScoredItem], *, dry_run: bool = False) -> None:
    _print_table(scored, dry_run=dry_run)
    if dry_run:
        print("(dry-run: discussion_minutes and rubric_score are 0; no API calls made)")
    else:
        print("Scores include: discussion time, dollar magnitude, vote type, rubric score.")
    print()


def _checkpoint_interaction(scored: list[ScoredItem], default_top: int = 5) -> tuple[list[ScoredItem], int]:
    """Interactive loop: let the user approve or adjust the ranking.

    Returns (adjusted_scored_list, top_n).
    """
    print("Commands:")
    print("  [Enter]         Approve this ranking")
    print("  top N           Change how many top items to draft (default 5)")
    print(f"  exclude N.NN    Remove an item from consideration")
    print(f"  include N.NN    Re-add an excluded item")
    print()

    top_n = default_top
    excluded: set[str] = set()
    active = list(scored)

    while True:
        try:
            raw = input(f"> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            raise SystemExit("Aborted at checkpoint.")

        if not raw:
            break

        parts = raw.split()
        cmd = parts[0].lower()

        if cmd == "top" and len(parts) == 2:
            try:
                top_n = int(parts[1])
                print(f"  Top-N set to {top_n}.")
            except ValueError:
                print(f"  Bad number: {parts[1]!r}")

        elif cmd == "exclude" and len(parts) == 2:
            num = parts[1]
            match = [s for s in active if s.item.number == num]
            if not match:
                print(f"  Item {num!r} not found in active list.")
            else:
                excluded.add(num)
                active = [s for s in active if s.item.number not in excluded]
                print(f"  Excluded {num}. {len(active)} items remain.")
                _print_checkpoint(active)

        elif cmd == "include" and len(parts) == 2:
            num = parts[1]
            excluded.discard(num)
            # Re-build active from original scored list minus excluded
            active = [s for s in scored if s.item.number not in excluded]
            finalize(active)
            print(f"  Re-included {num}. {len(active)} items remain.")
            _print_checkpoint(active)

        else:
            print(f"  Unknown command: {raw!r}. Type Enter to approve.")

    return active, top_n


def _attachment_selection(
    top_items: list[ScoredItem],
) -> list[tuple[ScoredItem, int]]:
    """Ask which attachments to fetch for the top items.

    Returns a list of (ScoredItem, attachment_index) pairs to fetch.
    """
    has_attachments = [s for s in top_items if s.item.attachments]
    if not has_attachments:
        print("No attachments available for the top items.")
        return []

    print("Attachments for top items (for quote-grounding in the prose call):")
    print()
    choices: list[tuple[str, ScoredItem, int]] = []  # (label, scored_item, att_idx)
    for s in has_attachments:
        print(f"  {s.rank}. [{s.item.number}] {s.item.title[:55]}")
        for j, att in enumerate(s.item.attachments):
            label = f"{s.rank}{chr(ord('a') + j)}"
            choices.append((label, s, j))
            print(f"        {label}) {att.name}")
    print()
    all_labels = " ".join(c[0] for c in choices)
    print(f"Fetch which attachments? (all / none / space-separated labels like '{all_labels[:6]}...')")

    try:
        raw = input("> ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print()
        return []

    if not raw or raw == "none":
        return []
    if raw == "all":
        return [(s, j) for _, s, j in choices]

    requested_labels = set(raw.split())
    selected = [(s, j) for label, s, j in choices if label in requested_labels]
    unknown = requested_labels - {label for label, _, _ in choices}
    if unknown:
        print(f"  Unknown labels ignored: {', '.join(sorted(unknown))}")
    return selected


# --- Env loading -----------------------------------------------------------

def _load_dotenv() -> None:
    env_path = PROJECT_DIR / ".env"
    if not env_path.is_file():
        return
    for raw in env_path.read_text().splitlines():
        line = raw.strip().removeprefix("export ").strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


# --- Pipeline --------------------------------------------------------------

def run(
    date: str,
    html_path: pathlib.Path,
    meta_path: pathlib.Path,
    *,
    work_session_url: str | None = None,
    livestream_url: str | None = None,
    dry_run: bool = False,
    top_n: int = 5,
) -> None:
    OUT_DIR.mkdir(exist_ok=True)
    html = html_path.read_text()
    meta = json.loads(meta_path.read_text())

    # --- Step 1: parse + triage ---
    print(f"\n{'='*60}")
    print(f"WJCC Newsletter — {date}")
    print(f"{'='*60}\n")
    items = parse_agenda(html)
    result = triage(items, meta, agenda_preamble(html))
    if livestream_url:
        result.logistics.livestream = livestream_url
    print_report(result)
    all_items = kept_items(result)

    if not all_items:
        raise SystemExit("Triage kept no substantive items.")

    scored = compute_deterministic(all_items, result)

    if dry_run:
        finalize(scored)
        _print_checkpoint(scored, dry_run=True)
        out_path = OUT_DIR / f"score-dry-{date}.json"
        out_path.write_text(json.dumps([dataclasses.asdict(s) for s in scored], indent=2))
        print(f"Wrote {out_path.relative_to(PROJECT_DIR)}")
        return

    # --- Steps 2-3: transcript + scoring ---
    _load_dotenv()
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("ANTHROPIC_API_KEY is not set (env var or .env file).")
    client = anthropic.Anthropic()

    snippets: list[dict] = []
    if work_session_url:
        print("\nFetching work-session transcript...")
        snippets = fetch_transcript(work_session_url, cache_dir=CACHE_DIR)

    print("\nScoring items...")
    if snippets:
        segment_transcript(scored, snippets, client)

    rubric_path = PROJECT_DIR / "rubric.md"
    if rubric_path.is_file():
        score_rubric(scored, rubric_path.read_text(), client)
    else:
        print("  WARNING: rubric.md not found.")

    finalize(scored)

    # --- Step 4: checkpoint ---
    _print_checkpoint(scored)
    active, top_n = _checkpoint_interaction(scored, default_top=top_n)
    top_items = active[:top_n]
    print(f"\nProceeding with top {len(top_items)} items:")
    for s in top_items:
        print(f"  {s.rank}. [{s.item.number}] {s.item.title[:60]}")
    print()

    # --- Step 5: fetch attachments ---
    selections = _attachment_selection(top_items)
    attachment_paths: dict[str, pathlib.Path] = {}
    if selections:
        from attachments import fetch_attachment
        from pull_agenda import new_session

        print("\nFetching attachments...")
        session = new_session()
        for s, att_idx in selections:
            att = s.item.attachments[att_idx]
            path = fetch_attachment(att.url, session, cache_dir=CACHE_DIR)
            attachment_paths[att.url] = path

    # --- Step 6: write ---
    print("\nDrafting newsletter with Claude...")
    feedback_text: str | None = None
    try:
        raw_feedback = input("Any reviewer notes for Claude? (Enter to skip) > ").strip()
        if raw_feedback:
            feedback_text = raw_feedback
    except (EOFError, KeyboardInterrupt):
        print()

    the_draft = draft(
        top_items,
        attachment_paths=attachment_paths or None,
        feedback=feedback_text,
        client=client,
    )

    for note in validate_draft(the_draft, top_items):
        print(f"  ! {note}")

    # --- Step 7: render + save ---
    score_path = OUT_DIR / f"score-{date}.json"
    draft_path = OUT_DIR / f"draft-{date}.json"
    newsletter_path = OUT_DIR / f"newsletter-{date}.md"

    score_path.write_text(json.dumps([dataclasses.asdict(s) for s in active], indent=2))
    draft_path.write_text(json.dumps(dataclasses.asdict(the_draft), indent=2))
    newsletter_path.write_text(render_newsletter(
        result.logistics,
        top_items,
        the_draft,
        meeting_unique=meta.get("unique"),
    ))

    print(f"\nWrote {score_path.relative_to(PROJECT_DIR)}")
    print(f"Wrote {draft_path.relative_to(PROJECT_DIR)}")
    print(f"Wrote {newsletter_path.relative_to(PROJECT_DIR)}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--date",
        help="meeting numberdate, e.g. 20260519 (default: newest fixture)",
    )
    parser.add_argument(
        "--work-session",
        metavar="URL",
        help="YouTube URL for the work-session transcript (enables discussion signal)",
    )
    parser.add_argument(
        "--livestream",
        metavar="URL",
        help="meeting livestream URL for the newsletter footer",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="deterministic signals only; no API calls, no cost",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=5,
        help="default top-N items to draft (adjustable at checkpoint, default 5)",
    )
    args = parser.parse_args()

    date, html_path, meta_path = _select_fixture(args.date)

    run(
        date,
        html_path,
        meta_path,
        work_session_url=args.work_session,
        livestream_url=args.livestream,
        dry_run=args.dry_run,
        top_n=args.top,
    )


if __name__ == "__main__":
    main()
