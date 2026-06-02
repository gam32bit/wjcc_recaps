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
import sys

from parse import agenda_preamble, parse_agenda
from score import ScoredItem, compute_deterministic, evidence_line, finalize
from triage import Logistics, kept_items, triage
from write import Draft, draft_from_dict

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
