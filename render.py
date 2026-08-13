#!/usr/bin/env python3
"""Render the recap (or the preview newsletter) into paste-ready Markdown.

Deterministic, no LLM, no network. Lays out parsed agenda data, the signal data
from score.py, and the public-comment tally from pubcomment.py.

The recap contains no model-written prose at all: each highlight is the agenda
item's own title plus its verbatim BACKGROUND text as an attributed block
quote, followed by watch links, document links, and the parsed vote tally. The
preview product still uses write.py's drafted prose.

Meeting logistics come straight from the (never-sent-to-Claude) Logistics
record, so date, times, locations, and the livestream URL are always exact.

Usage:
    python render.py out/recap-score-20260616.json \\
                     wjcc-fixtures/agenda-20260616.html \\
                     wjcc-fixtures/meeting-meta-20260616.json \\
                     --pubcomment out/recap-pubcomment-20260616.json

    python render.py out/score-20260519.json <agenda.html> <meta.json> \\
                     --preview --draft out/draft-20260519.json
"""

import argparse
import dataclasses
import datetime as dt
import json
import pathlib
import re
import sys

from parse import AgendaItem, agenda_preamble, body_sections, parse_agenda
from pubcomment import PublicCommentTally, from_json as pubcomment_from_json
from score import ScoredItem, compute_deterministic, evidence_line, finalize, vote_summary
from transcript import extract_video_id
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


# --- Recap: verbatim agenda text -------------------------------------------

# A description longer than this gets cut at a paragraph boundary. The case
# this exists for is item 8.10, whose BACKGROUND ends in a nine-bullet
# equipment scope that would swamp the section.
_DESCRIPTION_CAP = 700


def _trim_paragraphs(text: str, cap: int) -> tuple[str, bool]:
    """Keep whole paragraphs up to `cap` chars. Returns (text, was_trimmed)."""
    paragraphs = [p for p in re.split(r"\n\s*\n", text.strip()) if p.strip()]
    if not paragraphs:
        return "", False
    kept = [paragraphs[0]]
    total = len(paragraphs[0])
    for para in paragraphs[1:]:
        if total + len(para) > cap:
            break
        kept.append(para)
        total += len(para)
    # A kept paragraph ending in ':' introduces one that was cut — drop the
    # dangling lead-in rather than leave the reader hanging.
    while len(kept) > 1 and kept[-1].rstrip().endswith(":"):
        kept.pop()
    return "\n\n".join(kept), len(kept) < len(paragraphs)


def _item_description(item: AgendaItem) -> tuple[str, bool]:
    """The agenda's own description of an item, verbatim, plus a trimmed flag.

    Staff write item bodies to a labeled template, and BACKGROUND is the
    section that describes the thing itself — it holds for 202 of the 223
    substantive items across the fixture agendas. The rest of the chain covers
    the stragglers: RATIONALE tends to be written in the future tense ("staff
    will present"), which reads wrong in a recap, so it ranks below BACKGROUND;
    a handful of older items have unlabeled prose; the last resorts are the
    item's own TOPIC line, its recommended action, and finally its title.
    """
    sections, lead = body_sections(item.body)
    candidates = (
        sections.get("BACKGROUND"),
        sections.get("RATIONALE"),
        lead,
        sections.get("TOPIC"),
        item.recommended_action,
    )
    for candidate in candidates:
        text = (candidate or "").strip()
        if text and text != "N/A":
            return _trim_paragraphs(text, _DESCRIPTION_CAP)
    return item.title.strip(), False


def _cost_line(item: AgendaItem) -> str | None:
    """The agenda's COST BUDGETED text, when it says anything."""
    sections, _ = body_sections(item.body)
    cost, _ = _trim_paragraphs((sections.get("COST BUDGETED") or "").strip(), 400)
    if not cost or cost == "N/A":
        return None
    return f"**Cost budgeted:** {' '.join(cost.split())}"


def _block_quote(text: str) -> list[str]:
    """Markdown block-quote lines, with the blank lines quoted too.

    Substack drops out of the quote at an unprefixed blank line, which would
    silently turn the second paragraph of an excerpt into newsletter voice.
    """
    lines: list[str] = []
    for line in text.split("\n"):
        lines.append(f"> {line}".rstrip() if line.strip() else ">")
    return lines


def _short_title(title: str, limit: int = 50) -> str:
    """Shorten a long agenda title at a word boundary, for the summary line."""
    title = " ".join(title.split())
    if len(title) <= limit:
        return title
    return title[:limit].rsplit(" ", 1)[0].rstrip(",;:") + "…"


def _by_the_numbers(
    all_items: list[AgendaItem],
    action_items: list[AgendaItem],
    public_comment: PublicCommentTally | None,
) -> str | None:
    """The recap's opening stat line — every figure counted, none written.

    Each clause is dropped when its data is absent (a run without a --meeting
    URL has no speaker figures), so the line degrades instead of lying.
    """
    parts = [f"{len(all_items)} items"]

    voted = [i for i in action_items if i.vote]
    if voted:
        split = [i for i in voted if i.vote.contested]
        outcome = "all unanimous" if not split else f"{len(split)} not unanimous"
        parts.append(f"{len(voted)} action vote{'s' if len(voted) != 1 else ''}, {outcome}")

    if public_comment and public_comment.total_speakers:
        n = public_comment.total_speakers
        clause = f"{n} resident{'s' if n != 1 else ''} spoke"
        top = public_comment.top_item
        if top and top.count > 1:
            clause += f", {top.count} of them on {_short_title(top.label)}"
        parts.append(clause)

    return "**By the numbers:** " + " · ".join(parts) if parts else None


def _public_comment_section(public_comment: PublicCommentTally) -> list[str]:
    """The speaker tally — a count per topic, ranked. No prose, no names."""
    lines = ["## What people came to talk about", ""]
    n = public_comment.total_speakers
    lines += [f"{n} resident{'s' if n != 1 else ''} spoke during public comment.", ""]

    def bullet(t) -> str:
        # Agenda titles arrive capitalized; off-agenda labels come back in the
        # model's lowercase phrasing, so lift the first letter to match.
        label = t.label if t.item_number else t.label[:1].upper() + t.label[1:]
        tag = f" (Item {t.item_number})" if t.item_number else ""
        return f"- **{label}**{tag} — {t.count} speaker{'s' if t.count != 1 else ''}"

    lines += [bullet(t) for t in public_comment.by_item]
    if public_comment.by_item:
        lines.append("")
    if public_comment.off_agenda:
        lines += ["Other topics raised (not on the agenda):", ""]
        lines += [bullet(t) for t in public_comment.off_agenda]
        lines.append("")
    return lines


def render_recap(
    logistics: Logistics,
    top_items: list[ScoredItem],
    all_items: list[AgendaItem],
    action_items: list[AgendaItem],
    consent_items: list[AgendaItem],
    *,
    meeting_unique: str | None = None,
    meeting_video: str | None = None,
    work_session_video: str | None = None,
    public_comment: PublicCommentTally | None = None,
) -> str:
    """Render the post-meeting recap: top highlights + vote-count lists.

    Every word here is either quoted from the agenda or counted from parsed
    data — no model writes any of it. Descriptions are the agenda's own
    BACKGROUND text, block-quoted and attributed so a reader can tell the
    district's wording from the newsletter's. Vote tallies come from the
    deterministically parsed `item.vote`, so they are exact against the
    finalized agenda. Per-item "watch" deep-links are built from the
    segmentation start timestamps, which a reader can verify by seeking.
    """
    date_str = _format_date(logistics.date)
    lines: list[str] = []

    title = "Recap: WJCC School Board"
    if date_str:
        title += f", {date_str}"
    lines += [f"# {title}", ""]
    stat_line = _by_the_numbers(all_items, action_items, public_comment)
    if stat_line:
        lines += [stat_line, ""]

    # --- Highlights (agenda title + the agenda's own description) ---
    lines += ["## Highlights", ""]
    top_numbers = {s.item.number for s in top_items}
    for s in top_items:
        head = f"Item {s.item.number} — {s.item.title}" if s.item.number else s.item.title
        lines += [f"### {head.strip()}", ""]

        description, trimmed = _item_description(s.item)
        label = "**From the agenda item (excerpt):**" if trimmed else "**From the agenda item:**"
        lines += [label, ""]
        lines += _block_quote(description)
        lines.append("")

        cost = _cost_line(s.item)
        if cost:
            lines += [cost, ""]

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

    # --- Public comment tally (before the action-item lists) ---
    if public_comment and public_comment.total_speakers:
        lines += _public_comment_section(public_comment)

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
    parser.add_argument("score_json", type=pathlib.Path, help="scored items JSON")
    parser.add_argument("html", type=pathlib.Path, help="agenda HTML file")
    parser.add_argument("meta", type=pathlib.Path, help="meeting-meta JSON file")
    parser.add_argument("--draft", type=pathlib.Path,
                        help="draft JSON from write.py (required for --preview)")
    parser.add_argument("--preview", action="store_true",
                        help="render the preview newsletter instead of the recap")
    parser.add_argument("--pubcomment", type=pathlib.Path,
                        help="recap-pubcomment JSON, for the speaker tally")
    parser.add_argument("--top", type=int, default=3,
                        help="how many highlights to render in a recap (default 3)")
    parser.add_argument("--meeting", help="meeting video URL, for watch links")
    parser.add_argument("--work-session", help="work-session video URL")
    args = parser.parse_args()

    paths = [args.score_json, args.html, args.meta]
    if args.draft:
        paths.append(args.draft)
    if args.pubcomment:
        paths.append(args.pubcomment)
    for p in paths:
        if not p.is_file():
            sys.exit(f"No such file: {p}")

    meta = json.loads(args.meta.read_text())
    html = args.html.read_text()
    result = triage(parse_agenda(html), meta, agenda_preamble(html))

    # Rebuild ScoredItems from the saved JSON: the deterministic signals are
    # recomputed from the agenda, the API-derived ones are restored.
    score_data = json.loads(args.score_json.read_text())
    ranked = compute_deterministic(kept_items(result), result)
    saved_by_number = {entry["item"]["number"]: entry for entry in score_data}
    for s in ranked:
        saved = saved_by_number.get(s.item.number)
        if not saved:
            continue
        sig = saved["signals"]
        s.signals.discussion_minutes = sig.get("discussion_minutes", 0.0)
        s.signals.rubric_score = sig.get("rubric_score", -1.0)
        s.signals.rubric_justification = sig.get("rubric_justification", "")
        s.signals.public_comment_minutes = sig.get("public_comment_minutes", 0.0)
        s.signals.public_comment_speakers = sig.get("public_comment_speakers", 0)
        s.signals.meeting_start_seconds = sig.get("meeting_start_seconds")
        s.signals.work_session_start_seconds = sig.get("work_session_start_seconds")
    finalize(ranked)

    if args.preview:
        if not args.draft:
            sys.exit("--preview needs --draft (the preview product is still drafted).")
        the_draft = draft_from_dict(json.loads(args.draft.read_text()))
        top_numbers = {di.number for di in the_draft.items}
        top = [s for s in ranked if s.item.number in top_numbers]
        print(render_newsletter(
            result.logistics, top, the_draft, meeting_unique=meta.get("unique"),
        ), end="")
        return

    public_comment = None
    if args.pubcomment:
        _, public_comment, _ = pubcomment_from_json(
            json.loads(args.pubcomment.read_text())
        )

    print(render_recap(
        result.logistics,
        ranked[: args.top],
        kept_items(result),
        result.action_items,
        result.consent_agenda,
        meeting_unique=meta.get("unique"),
        meeting_video=args.meeting,
        work_session_video=args.work_session,
        public_comment=public_comment,
    ), end="")


if __name__ == "__main__":
    main()
