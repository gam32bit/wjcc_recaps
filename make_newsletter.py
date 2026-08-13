#!/usr/bin/env python3
"""Orchestrate the WJCC agenda -> newsletter pipeline.

The default product is the RECAP of a meeting that already happened. It needs the
FINALIZED agenda (with vote counts) — re-fetch it first with
`python pull_agenda.py --date <date>` — plus the meeting video:

    python make_newsletter.py --date 20260616 \\
        --meeting https://www.youtube.com/watch?v=PvOFaqmguYg \\
        --work-session https://www.youtube.com/live/Y-LpLQm27AI   # optional, for deep-links

A full interactive recap run:
  1. parse + triage (offline)
  2. fetch the meeting transcript (and optionally the work-session transcript);
     both are also saved as out/transcript-*-<date>.md
  3. score: segmentation call(s) + public-comment speaker call + rubric call
  4. CHECKPOINT: review the ranking and the public-comment tally
  5. render -> out/recap-<date>.md

The recap itself contains no model-written prose. Highlights quote the agenda's
own wording; the public-comment section is a counted tally, not a summary.

Pass --preview for the upcoming-meeting product ("The Rundown"), which scores
discussion from the prior work session instead:

    python make_newsletter.py --date 20260519 --preview \\
        --work-session https://www.youtube.com/live/Y-LpLQm27AI \\
        --livestream   https://www.youtube.com/live/3np-qU3BSnQ

--dry-run stops after step 3 (deterministic signals only, no API calls).
"""

import argparse
import dataclasses
import json
import os
import pathlib
import sys

import anthropic

import pubcomment
from parse import agenda_preamble, parse_agenda, to_dict
from render import render_newsletter, render_recap
from score import (
    ScoredItem,
    compute_deterministic,
    evidence_line,
    finalize,
    score_rubric,
    segment_transcript,
    _print_table,
)
from transcript import fetch_transcript, transcript_to_markdown
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


def _transcript_sections(scored: list[ScoredItem], attr: str) -> list[tuple[float, str]]:
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


def _write_transcript_md(
    date: str,
    label: str,
    snippets: list[dict],
    sections: list[tuple[float, str]] | None = None,
) -> None:
    """Save a readable, timestamped, agenda-segmented Markdown copy of a transcript."""
    titles = {"meeting": "Meeting transcript", "worksession": "Work session transcript"}
    title = f"{titles.get(label, 'Transcript')} — {date}"
    path = OUT_DIR / f"transcript-{label}-{date}.md"
    path.write_text(transcript_to_markdown(snippets, title, sections=sections))
    extra = f", {len(sections)} sections" if sections else ""
    print(f"  Wrote {path.relative_to(PROJECT_DIR)} ({len(snippets)} snippets{extra})")


def _public_comment_tally(
    date: str,
    scored: list[ScoredItem],
    all_items: list,
    snippets: list[dict],
    pc_ranges: list[dict],
    client: anthropic.Anthropic,
) -> tuple[list, "pubcomment.PublicCommentTally"]:
    """Identify public-comment speakers, count them, and write the audit file.

    The counting is plain Python; Claude only says where each speaker starts
    and which agenda item they addressed. The saved JSON holds the raw answer,
    the derived tally, and the period bounds, so the section can be re-rendered
    (and re-checked) later without another API call.
    """
    text = pubcomment.compact_slice(snippets, pc_ranges)
    speakers = pubcomment.classify_speakers(
        text, all_items, client, pc_ranges=pc_ranges
    )
    result = pubcomment.tally(speakers, all_items, pc_ranges)
    pubcomment.apply_to_signals(scored, result)

    path = OUT_DIR / f"recap-pubcomment-{date}.json"
    path.write_text(pubcomment.to_json(speakers, result, pc_ranges))
    print(
        f"  {result.total_speakers} public-comment speakers across "
        f"{len(result.by_item)} agenda items and {len(result.off_agenda)} "
        f"off-agenda topics. Wrote {path.relative_to(PROJECT_DIR)}"
    )
    return speakers, result


def _fetch_selected_attachments(
    top_items: list[ScoredItem],
) -> dict[str, pathlib.Path]:
    """Ask which PDFs to fetch, then fetch them (preview drafting only).

    Only the preview product sends documents to Claude. The recap links to
    attachments straight from the parsed agenda and never downloads them.
    """
    selections = _attachment_selection(top_items)
    if not selections:
        return {}

    from attachments import fetch_attachment
    from pull_agenda import new_session

    print("\nFetching attachments...")
    session = new_session()
    paths: dict[str, pathlib.Path] = {}
    for s, att_idx in selections:
        att = s.item.attachments[att_idx]
        paths[att.url] = fetch_attachment(att.url, session, cache_dir=CACHE_DIR)
    return paths


def _print_public_comment(result: "pubcomment.PublicCommentTally") -> None:
    """Show the tally at the checkpoint — including the off-agenda labels.

    Those labels are the only model-authored strings that reach the published
    recap, so they get looked at before anyone hits publish.
    """
    if not result or not result.total_speakers:
        return
    print(f"\nPublic comment — {result.total_speakers} speakers:")
    for t in result.by_item:
        print(f"  {t.count:>3}  [{t.item_number}] {t.label[:55]}")
    for t in result.off_agenda:
        print(f"  {t.count:>3}  (off-agenda) {t.label}")


# --- Pipeline --------------------------------------------------------------

def run(
    date: str,
    html_path: pathlib.Path,
    meta_path: pathlib.Path,
    *,
    work_session_url: str | None = None,
    livestream_url: str | None = None,
    meeting_url: str | None = None,
    recap: bool = False,
    dry_run: bool = False,
    top_n: int = 5,
) -> None:
    OUT_DIR.mkdir(exist_ok=True)
    html = html_path.read_text()
    meta = json.loads(meta_path.read_text())

    # --- Step 1: parse + triage ---
    kind = "Recap" if recap else "Newsletter"
    print(f"\n{'='*60}")
    print(f"WJCC {kind} — {date}")
    print(f"{'='*60}\n")
    items = parse_agenda(html)
    result = triage(items, meta, agenda_preamble(html))
    if livestream_url:
        result.logistics.livestream = livestream_url
    print_report(result)
    all_items = kept_items(result)

    if not all_items:
        raise SystemExit("Triage kept no substantive items.")

    if recap and not any(i.vote for i in all_items):
        print(
            "\n  WARNING: no Motion & Voting blocks found in this agenda. "
            "Re-fetch the FINALIZED post-meeting agenda with "
            f"`python pull_agenda.py --date {date}` before running a recap.\n"
        )

    scored = compute_deterministic(all_items, result)

    if dry_run:
        finalize(scored)
        _print_checkpoint(scored, dry_run=True)
        out_path = OUT_DIR / f"{'recap-score-dry' if recap else 'score-dry'}-{date}.json"
        out_path.write_text(json.dumps([dataclasses.asdict(s) for s in scored], indent=2))
        print(f"Wrote {out_path.relative_to(PROJECT_DIR)}")
        return

    # --- Steps 2-3: transcript(s) + scoring ---
    _load_dotenv()
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("ANTHROPIC_API_KEY is not set (env var or .env file).")
    client = anthropic.Anthropic()

    meeting_snippets: list[dict] = []
    meeting_seg_data: dict = {}
    pc_tally: pubcomment.PublicCommentTally | None = None

    if recap:
        # A recap scores discussion + public comment from the MEETING video, and
        # (when given) also segments the work session — purely to deep-link each
        # item to where it was previewed; that pass does not affect scoring.
        if not meeting_url:
            print(
                "\n  NOTE: no --meeting URL — the recap will rank on rubric + "
                "dollar + vote signals only (no transcripts, deep-links, or "
                "public-comment summary).\n"
            )
        if meeting_url:
            print("\nFetching meeting transcript...")
            meeting_snippets = fetch_transcript(meeting_url, cache_dir=CACHE_DIR)
        print("\nScoring items...")
        if meeting_snippets:
            meeting_seg_data = segment_transcript(
                scored, meeting_snippets, client, kind="meeting", score=True
            )
            # Agenda-item headings cover scored items only; public comment is
            # triaged out but its range is already segmented, so add it for free.
            sections = _transcript_sections(scored, "meeting_start_seconds")
            pc_ranges = meeting_seg_data.get("public_comment_period", [])
            if pc_ranges:
                pc_start = min(r["start_seconds"] for r in pc_ranges)
                sections.append((float(pc_start), "Public comment"))
                # Who spoke about what. This runs BEFORE the checkpoint because
                # its per-item minutes feed the composite, and therefore the
                # ranking the checkpoint asks you to approve.
                speakers, pc_tally = _public_comment_tally(
                    date, scored, all_items, meeting_snippets, pc_ranges, client
                )
                sections += pubcomment.speaker_anchors(speakers, all_items)
            _write_transcript_md(date, "meeting", meeting_snippets, sections)
        if work_session_url:
            print("\nFetching work-session transcript (for deep-links)...")
            ws_snippets = fetch_transcript(work_session_url, cache_dir=CACHE_DIR)
            if ws_snippets:
                segment_transcript(
                    scored, ws_snippets, client, kind="work_session", score=False
                )
                _write_transcript_md(
                    date, "worksession", ws_snippets,
                    _transcript_sections(scored, "work_session_start_seconds"),
                )
    else:
        # A preview measures discussion from the prior work session.
        ws_snippets = []
        if work_session_url:
            print("\nFetching work-session transcript...")
            ws_snippets = fetch_transcript(work_session_url, cache_dir=CACHE_DIR)
        print("\nScoring items...")
        if ws_snippets:
            segment_transcript(
                scored, ws_snippets, client, kind="work_session", score=True
            )
            _write_transcript_md(
                date, "worksession", ws_snippets,
                _transcript_sections(scored, "work_session_start_seconds"),
            )

    rubric_path = PROJECT_DIR / "rubric.md"
    if rubric_path.is_file():
        score_rubric(scored, rubric_path.read_text(), client)
    else:
        print("  WARNING: rubric.md not found.")

    finalize(scored)

    # --- Step 4: checkpoint ---
    _print_checkpoint(scored)
    _print_public_comment(pc_tally)
    active, top_n = _checkpoint_interaction(scored, default_top=top_n)
    top_items = active[:top_n]
    print(f"\nProceeding with top {len(top_items)} items:")
    for s in top_items:
        print(f"  {s.rank}. [{s.item.number}] {s.item.title[:60]}")
    print()

    # --- Step 5: write (preview only) ---
    # The recap needs no drafting step: every line of it is either quoted from
    # the agenda or counted from parsed data.
    the_draft: Draft | None = None
    if not recap:
        attachment_paths = _fetch_selected_attachments(top_items)
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

    # --- Step 6: render + save ---
    prefix = "recap" if recap else "newsletter"
    score_path = OUT_DIR / f"{'recap-score' if recap else 'score'}-{date}.json"
    newsletter_path = OUT_DIR / f"{prefix}-{date}.md"
    written = [score_path]

    score_path.write_text(json.dumps([dataclasses.asdict(s) for s in active], indent=2))
    if recap:
        body = render_recap(
            result.logistics,
            top_items,
            all_items,
            result.action_items,
            result.consent_agenda,
            meeting_unique=meta.get("unique"),
            meeting_video=meeting_url,
            work_session_video=work_session_url,
            public_comment=pc_tally,
        )
    else:
        draft_path = OUT_DIR / f"draft-{date}.json"
        draft_path.write_text(json.dumps(dataclasses.asdict(the_draft), indent=2))
        written.append(draft_path)
        body = render_newsletter(
            result.logistics,
            top_items,
            the_draft,
            meeting_unique=meta.get("unique"),
        )
    newsletter_path.write_text(body)
    written.append(newsletter_path)

    print()
    for path in written:
        print(f"Wrote {path.relative_to(PROJECT_DIR)}")


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
        "--preview",
        action="store_true",
        help="preview an UPCOMING meeting (the old default); recap is now the default",
    )
    parser.add_argument(
        "--work-session",
        metavar="URL",
        help="YouTube URL for the work-session video (preview discussion signal; "
        "in a recap, used to deep-link each item to where it was previewed)",
    )
    parser.add_argument(
        "--meeting",
        metavar="URL",
        help="YouTube URL for the meeting video (recap discussion + public-comment signals)",
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
        default=None,
        help="top-N items to draft (adjustable at checkpoint; default 3 for recap, 5 for preview)",
    )
    args = parser.parse_args()

    date, html_path, meta_path = _select_fixture(args.date)
    recap = not args.preview
    top_n = args.top if args.top is not None else (3 if recap else 5)

    run(
        date,
        html_path,
        meta_path,
        work_session_url=args.work_session,
        livestream_url=args.livestream,
        meeting_url=args.meeting,
        recap=recap,
        dry_run=args.dry_run,
        top_n=top_n,
    )


if __name__ == "__main__":
    main()
