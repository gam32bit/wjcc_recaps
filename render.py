#!/usr/bin/env python3
"""Render the "Most Discussed" newsletter from a scored ranking and draft.

Deterministic, no LLM, no network. Lays out the structured Draft from write.py
and the signal data from score.py into a paste-ready Markdown post.

Meeting logistics come straight from the (never-sent-to-Claude) Logistics
record, so date, times, locations, and the livestream URL are always exact.
Each item includes a signal-evidence line showing raw signal values so readers
(and the editor) can verify the ranking is grounded in real data.

Usage:
    python render.py out/draft-20260519.json \\
                     out/score-20260519.json \\
                     wjcc-fixtures/agenda-20260519.html \\
                     wjcc-fixtures/meeting-meta-20260519.json
"""

import argparse
import dataclasses
import datetime as dt
import json
import pathlib
import re
import sys

from parse import AgendaItem, agenda_preamble, parse_agenda
from score import ScoredItem, compute_deterministic, evidence_line, finalize, vote_summary
from transcript import extract_video_id
from triage import Logistics, kept_items, triage
from write import Draft, PublicCommentSummary, draft_from_dict

WJCC_MEETINGS_URL = "https://wjcc.k12.va.us/school-board/school-board-meetings/"


def _format_date(iso: str | None) -> str | None:
    if not iso:
        return None
    try:
        d = dt.date.fromisoformat(iso)
    except ValueError:
        return iso
    return f"{d:%A, %B} {d.day}, {d.year}"


def _agenda_url(meeting_unique: str | None) -> str:
    """Return the public BoardDocs URL for this meeting, or the WJCC fallback."""
    if meeting_unique:
        return (
            f"https://go.boarddocs.com/vsba/wjcc/Board.nsf/Public?open&id={meeting_unique}"
        )
    return WJCC_MEETINGS_URL


def render_newsletter(
    logistics: Logistics,
    ranked_top: list[ScoredItem],
    the_draft: Draft,
    *,
    meeting_unique: str | None = None,
) -> str:
    """Render the "Most Discussed" newsletter as Markdown."""
    date_str = _format_date(logistics.date)
    lines: list[str] = []

    # Header
    title = "The Rundown: WJCC School Board"
    if date_str:
        title += f" — {date_str}"
    lines += [f"# {title}", ""]

    # Intro
    lines += [the_draft.intro.strip(), ""]

    # Ranked items
    lines += ["## Most Discussed", ""]

    # Build a lookup from item number -> DraftItem
    draft_by_number = {di.number: di for di in the_draft.items}

    for s in ranked_top:
        di = draft_by_number.get(s.item.number)
        if di is None:
            continue

        tag = f" (Item {s.item.number})" if s.item.number else ""
        lines += [f"### {s.rank}. {di.headline.strip()}{tag}", ""]

        ev = evidence_line(s)
        lines += [f"*{ev}*", ""]

        lines += [di.what_it_is.strip(), ""]

        for q in di.quotes:
            lines += [f"> {q.quote.strip()}", f"> — *{q.source.strip()}*", ""]

    # Footer
    lines += ["---", "", "## How to weigh in", ""]
    if date_str:
        lines += [f"The WJCC School Board meets **{date_str}**.", ""]
    for s in logistics.sessions:
        detail = ", ".join(bit for bit in (s.time, s.location) if bit)
        lines.append(f"- **{s.name}**" + (f" — {detail}" if detail else ""))
    if logistics.sessions:
        lines.append("")
    if logistics.public_comment_rule:
        lines += [f"**To speak:** {logistics.public_comment_rule}", ""]
    if logistics.livestream and not logistics.livestream.startswith("www.wjccschools"):
        lines += [f"**Watch live:** {logistics.livestream}", ""]
    if logistics.next_meeting:
        lines += [f"**Next meeting:** {logistics.next_meeting}", ""]
    agenda_link = _agenda_url(meeting_unique)
    lines += [f"**Full agenda:** {agenda_link}", ""]

    return "\n".join(lines).rstrip() + "\n"


def _vote_tally_line(item: AgendaItem) -> str | None:
    """Deterministic vote-tally line for a recap highlight, e.g.
    '**Vote tally:** Approved 7-0 (moved by Randy Riffle, seconded by ...)'.

    Returns None when the item carries no recorded vote — the recap then omits
    the line entirely rather than printing 'No recorded vote'.
    """
    v = item.vote
    if not v:
        return None
    line = f"**Vote tally:** {vote_summary(item)}"
    if v.mover:
        moved = f"moved by {v.mover}"
        if v.seconder:
            moved += f", seconded by {v.seconder}"
        line += f" ({moved})"
    return line


def _watch_url(video_url: str | None, start_seconds: float | None) -> str | None:
    """Canonical YouTube watch URL, deep-linked to start_seconds when known."""
    if not video_url:
        return None
    vid = extract_video_id(video_url)
    base = f"https://www.youtube.com/watch?v={vid}" if vid else video_url
    if start_seconds is None:
        return base
    sep = "&" if "?" in base else "?"
    return f"{base}{sep}t={int(start_seconds)}s"


def _attachment_label(name: str) -> str:
    """Drop a trailing file-size annotation like ' (905 KB)' from a link label."""
    cleaned = re.sub(r"\s*\(\d[\d.,]*\s*[KMG]B\)\s*$", "", name).strip()
    return cleaned or name


def _schedule_line(entry) -> str:
    """One footer bullet for an upcoming meeting/event (comma-separated facts)."""
    detail = ", ".join(p for p in (entry.date, entry.time, entry.location) if p)
    return f"**{entry.name}**" + (f", {detail}" if detail else "")


def render_recap(
    logistics: Logistics,
    top_items: list[ScoredItem],
    the_draft: Draft,
    action_items: list[AgendaItem],
    consent_items: list[AgendaItem],
    *,
    meeting_unique: str | None = None,
    meeting_video: str | None = None,
    work_session_video: str | None = None,
    public_comment: PublicCommentSummary | None = None,
) -> str:
    """Render the post-meeting recap: top highlights + vote-count lists.

    The vote tallies in every section come straight from the deterministically
    parsed `item.vote` — never from the LLM — so they are exact against the
    finalized agenda. Per-item "watch" deep-links are built from the segmentation
    start timestamps, which a reader can verify by seeking the video.
    """
    date_str = _format_date(logistics.date)
    lines: list[str] = []

    title = "Recap: WJCC School Board"
    if date_str:
        title += f", {date_str}"
    lines += [f"# {title}", ""]
    lines += [the_draft.intro.strip(), ""]

    # --- Highlights (full prose treatment) ---
    lines += ["## Highlights", ""]
    draft_by_number = {di.number: di for di in the_draft.items}
    top_numbers = {s.item.number for s in top_items}
    for s in top_items:
        di = draft_by_number.get(s.item.number)
        if di is None:
            continue
        lines += [f"### {di.headline.strip()}", ""]
        lines += [di.what_it_is.strip(), ""]

        # Watch links — only when the item was located in that video.
        watch_links: list[str] = []
        if meeting_video and s.signals.meeting_start_seconds is not None:
            url = _watch_url(meeting_video, s.signals.meeting_start_seconds)
            watch_links.append(f"[during the meeting]({url})")
        if work_session_video and s.signals.work_session_start_seconds is not None:
            url = _watch_url(work_session_video, s.signals.work_session_start_seconds)
            watch_links.append(f"[at the work session]({url})")
        if watch_links:
            lines += ["**Watch:** " + " · ".join(watch_links), ""]

        # Attachment links straight from the parsed agenda (deterministic).
        if s.item.attachments:
            docs = " · ".join(
                f"[{_attachment_label(a.name)}]({a.url})" for a in s.item.attachments
            )
            lines += ["**Documents:** " + docs, ""]

        tally = _vote_tally_line(s.item)
        if tally:
            lines += [tally, ""]

    # --- Public comment summary (before the action-item lists) ---
    if public_comment and public_comment.summary.strip():
        lines += ["## Public comment summary", ""]
        lines += [public_comment.summary.strip(), ""]
        if public_comment.quote.strip():
            lines.append(f"> {public_comment.quote.strip()}")
            if public_comment.speaker.strip():
                lines.append(f"> — *{public_comment.speaker.strip()}*")
            lines.append("")

    # --- Other action items (vote tallies only) ---
    other_actions = [i for i in action_items if i.number not in top_numbers]
    if other_actions:
        lines += ["## Other action items", ""]
        for i in other_actions:
            if i.vote:
                lines.append(f"- {i.title}, **{vote_summary(i)}**")
            else:
                lines.append(f"- {i.title}")
        lines.append("")

    # --- Consent agenda (batch vote) ---
    consent = [i for i in consent_items if i.number not in top_numbers]
    if consent:
        voted = [i for i in consent if i.vote]
        tallies = {vote_summary(i) for i in voted}
        if len(tallies) == 1:
            lines += [
                f"## Consent agenda — {tallies.pop()}",
                "",
                "Approved together as the consent agenda:",
                "",
            ]
            for i in consent:
                lines.append(f"- {i.title}")
        else:
            # Mixed or missing outcomes — show each item's tally explicitly.
            lines += ["## Consent agenda", ""]
            for i in consent:
                if i.vote:
                    lines.append(f"- {i.title}, **{vote_summary(i)}**")
                else:
                    lines.append(f"- {i.title}")
        lines.append("")

    # --- Footer ---
    lines += ["---", "", "## Meeting details", ""]
    if logistics.upcoming_meetings:
        lines += ["**Upcoming meetings**", ""]
        for e in logistics.upcoming_meetings:
            lines.append(f"- {_schedule_line(e)}")
        lines.append("")
    if logistics.upcoming_events:
        lines += ["**Upcoming events**", ""]
        for e in logistics.upcoming_events:
            lines.append(f"- {_schedule_line(e)}")
        lines.append("")
    if logistics.public_comment_rule:
        lines += [f"**How to participate:** {logistics.public_comment_rule}", ""]
    if meeting_video:
        lines += [f"**Watch the full meeting:** {meeting_video}", ""]
    agenda_link = _agenda_url(meeting_unique)
    lines += [f"[Full agenda]({agenda_link})", ""]

    return "\n".join(lines).rstrip() + "\n"


# --- CLI -------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("draft_json", type=pathlib.Path, help="draft JSON from write.py")
    parser.add_argument("score_json", type=pathlib.Path, help="scored items JSON")
    parser.add_argument("html", type=pathlib.Path, help="agenda HTML file")
    parser.add_argument("meta", type=pathlib.Path, help="meeting-meta JSON file")
    args = parser.parse_args()

    for p in (args.draft_json, args.score_json, args.html, args.meta):
        if not p.is_file():
            sys.exit(f"No such file: {p}")

    the_draft = draft_from_dict(json.loads(args.draft_json.read_text()))
    meta = json.loads(args.meta.read_text())
    html = args.html.read_text()
    result = triage(parse_agenda(html), meta, agenda_preamble(html))

    # Rebuild ScoredItems from the saved JSON
    score_data = json.loads(args.score_json.read_text())
    ranked = compute_deterministic(kept_items(result), result)
    # Re-apply saved signal data (discussion_minutes, rubric_score)
    saved_by_number = {entry["item"]["number"]: entry for entry in score_data}
    for s in ranked:
        saved = saved_by_number.get(s.item.number)
        if saved:
            s.signals.discussion_minutes = saved["signals"].get("discussion_minutes", 0.0)
            s.signals.rubric_score = saved["signals"].get("rubric_score", -1.0)
            s.signals.rubric_justification = saved["signals"].get("rubric_justification", "")
    finalize(ranked)

    top_numbers = {di.number for di in the_draft.items}
    top = [s for s in ranked if s.item.number in top_numbers]

    print(render_newsletter(
        result.logistics,
        top,
        the_draft,
        meeting_unique=meta.get("unique"),
    ), end="")


if __name__ == "__main__":
    main()
