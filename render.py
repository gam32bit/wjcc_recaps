#!/usr/bin/env python3
"""Render the recap (or the preview newsletter) into paste-ready Markdown.

Deterministic, no LLM, no network. Lays out parsed agenda data, the signal data
from score.py, and the public-comment tally from pubcomment.py.

The recap contains no model-written prose at all. A highlight carries the
maintainer's chosen quote from the meeting video (quotes-<period>.json), the
signals that ranked the item, the agenda item's own title and verbatim
BACKGROUND text as an attributed block quote, the parsed vote tally, and the
counted public-comment speakers. The preview product still uses write.py's
drafted prose.

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
from merge import display_number
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


# Where `pdfslice.py`'s per-item PDFs are served from. GitHub Pages, not a
# github.com blob URL or raw.githubusercontent.com — those render the file
# through their own viewer instead of handing over the district's document.
#
# EDIT THIS to match wherever the repo is published. Until docs/ is actually
# on Pages the links resolve to nothing, and `_read_more` emits no link at all
# when the constant is blank, so a run before publishing degrades rather than
# printing dead links.
PACKET_BASE_URL = "https://gam32bit.github.io/wjcc_recaps/packet/"


def _read_more(number: str, packet_paths: dict[str, str] | None) -> str | None:
    """Link from a trimmed excerpt to the item's own page in the packet.

    `packet_paths` is keyed by the item's namespaced number, so a period recap
    links each item into the meeting it actually came from.
    """
    if not PACKET_BASE_URL or not packet_paths:
        return None
    rel = packet_paths.get(number)
    return f"{PACKET_BASE_URL}{rel}" if rel else None


def _attachment_url(
    item: AgendaItem, attachment_paths: dict[str, str] | None
) -> str | None:
    """Link to the slice holding this item's attachments, when one was cut.

    Keyed by PAGE, not by item number, because a folded review's documents move
    with their page: `merge.py` gives the vote its review's attachments and
    stamps each one with the review's `source_page`, so the merged 8.07 carries
    documents that live at p.7 under 5.01. The item's own `source_page` is the
    fallback for an item whose attachments were never inherited.
    """
    if not PACKET_BASE_URL or not attachment_paths or not item.attachments:
        return None
    page = item.attachments[0].page or item.source_page
    if page is None:
        return None
    # Namespaced by meeting, exactly as `make_newsletter._packet_paths` writes
    # the map. Only a period recap reads a Diligent packet, so an item number
    # here always carries its prefix.
    prefix = item.number.partition("-")[0]
    rel = attachment_paths.get(f"{prefix}-{page}")
    return f"{PACKET_BASE_URL}{rel}" if rel else None


def _vote_tally_line(item: AgendaItem, video_url: str | None = None) -> str | None:
    """Deterministic vote-tally line for a recap highlight, e.g.
    '**Vote tally:** Approved 7-0 (moved by Randy Riffle, seconded by ...)'.

    Returns None when the item carries no recorded vote — the recap then omits
    the line entirely rather than printing 'No recorded vote'.

    A vote recovered from the video (`source="transcript"`, the only source a
    Diligent packet leaves) prints its counts and never the members' names:
    those are auto-caption spellings, and the same seven people come through as
    Hodgees/Hodes/Hodges.

    `video_url` used to deep-link the roll call. It is still accepted, and the
    caller still resolves it, so restoring the link is a one-line change — but
    the link is not printed today: see the note at the return below.
    """
    v = item.vote
    if not v:
        return None
    line = f"**Vote tally:** {vote_summary(item)}"
    if v.source == "transcript":
        detail = []
        if v.abstain:
            detail.append(f"{len(v.abstain)} abstaining")
        if v.absent:
            detail.append(f"{len(v.absent)} absent")
        if detail:
            line += f" ({', '.join(detail)})"
        # No roll-call deep link. Across the 27 votes recovered from the two
        # August videos, the roll-call timestamp and the item's discussion
        # timestamp are identical for 18 and within one 30-second segmentation
        # window for the other 9 — so the link landed on the same place in the
        # video as the "Watch discussion" line directly above it.
        return line
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


# Standing procedural facts about public comment. These are NOT in the agenda
# packet — the packet only carries the per-speaker time limit — so they are
# maintainer-owned text, edited here by a human when the board changes its
# practice. Nothing about them is generated.
#
# NOTE: the 6:30 p.m. start is duplicated from the board's standing schedule and
# will go stale silently if the board moves it. The "Upcoming meetings" list
# printed directly above this note carries the real open-session time for each
# meeting, so a mismatch is at least visible on the page.
_PARTICIPATION_NOTE = (
    "Public comment is taken at the regular meeting only, not at the work "
    "session. Speakers must sign up before the meeting begins at 6:30 p.m."
)


# Words the packet Title-Cases because the rule is an agenda item TITLE, not a
# sentence. Lowercased so the note reads as prose. Only common nouns are listed:
# "School Board" is a proper name and keeps its capitals.
_TITLE_CASE_WORDS = ("Minutes", "Address")


def _participation_note(logistics: Logistics) -> str:
    """The standing note, plus the packet's own speaking rule when it has one.

    The packet prints the rule as an agenda item title — "Public Comment - Each
    speaker may have 2 Minutes to Address the School Board" — so the label is
    stripped, the title-case artifacts are lowered, and the rule is appended.
    """
    rule = " ".join((logistics.public_comment_rule or "").split())
    rule = re.sub(r"(?i)^public\s+comment\s*[-–—:]\s*", "", rule).strip(" .")
    if not rule:
        return _PARTICIPATION_NOTE
    for word in _TITLE_CASE_WORDS:
        rule = re.sub(rf"\b{word}\b", word.lower(), rule)
    return f"{_PARTICIPATION_NOTE} {rule[:1].upper() + rule[1:]}."


def _schedule_line(entry) -> str:
    """One footer bullet for an upcoming meeting/event (comma-separated facts)."""
    detail = ", ".join(p for p in (entry.date, entry.time, entry.location) if p)
    return f"**{entry.name}**" + (f", {detail}" if detail else "")


# --- Recap: verbatim agenda text -------------------------------------------

# How much of an item's own description the recap shows before handing the
# reader off to the packet slice. Two sentences: enough to say what the item is,
# short enough that a highlight stays scannable. Everything cut is one click
# away in the district's own page, which is what "read more" links to.
_EXCERPT_SENTENCES = 2

# Where a sentence may end: terminal punctuation, optional closing quote or
# bracket, then whitespace.
_SENTENCE_END_RE = re.compile(r"(?<=[.!?])[\"'”’)\]]*\s+")

# Words whose trailing period does NOT end a sentence. Taken from the agenda
# text itself — "Sonny Merryman, Inc.", "2 Minutes", "6:30 p.m.", "Case #R2627-01
# ... No. 4" — plus the honorifics the board's own items use. A single letter
# ("R. Smith") is handled separately, by length.
_ABBREVIATIONS = {
    "inc", "co", "corp", "ltd", "llc", "dept", "div", "est", "approx", "no",
    "vs", "etc", "fig", "mr", "mrs", "ms", "dr", "jr", "sr", "st", "ave", "rd",
    "u.s", "a.m", "p.m", "e.g", "i.e",
}


# A line that reads as one cell of a table pdftotext flattened into a column,
# rather than a line of prose. Pure figures, and short label fragments that
# carry no sentence punctuation. Bulleted lines are excluded explicitly: the
# playground item's evaluation criteria ("• Compliance with ADA and national
# standards") are real content and would otherwise look the same.
_FIGURE_RE = re.compile(r"^[\s$().,%–—-]*[\d,.]+[\s$()%–—-]*$")
_BULLET_RE = re.compile(r"^\s*(?:[•·*\-–—]|\(?\d+\\?[.)])\s")
# How many tabular lines in a row before the text is judged to have stopped
# being prose. Three, so a single short line ending a paragraph is safe.
_TABLE_RUN = 3


def _is_tabular(line: str) -> bool:
    text = line.strip()
    if not text or _BULLET_RE.match(line):
        return False
    if _FIGURE_RE.match(text):
        return True
    return len(text.split()) <= 4 and not text.endswith((".", "?", "!"))


def _strip_flattened_table(text: str) -> tuple[str, bool]:
    """Cut a description where its prose gives way to a flattened table.

    A Diligent packet is a PDF, and `pdftotext` turns a table into one cell per
    line down a column. The FY27 Operating Fund amendment is the case: four
    lines of prose, then "Revenue Type / Local Revenue / James City County /
    ... / Total", then a column of bare figures. `_trim_paragraphs` cannot see
    it — the extracted text has no blank line between the prose and the table,
    so the whole thing is one paragraph and the cap lands mid-column.

    Returns (text, was_cut). Nothing is rewritten; the prose is kept verbatim
    and the table is dropped, which is what the "read more" link is for.
    """
    lines = text.split("\n")
    run = 0
    start = 0     # index the current run began at — tracked, not derived from
                  # `n - run`, because blank lines advance n without adding to
                  # run and the CNS table ("Cooperative" / "Contract" / blank /
                  # figures) left one header line stranded in the excerpt.
    for n, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue          # blank lines neither start nor break a run
        if not _is_tabular(line):
            # Any prose line resets. Letting a long line CONTINUE a run (on the
            # theory that it is a wrapped table cell) was tried and reverted: it
            # took the cut count across the fixtures from 8 genuine tables to
            # 29, swallowing real numbered lists whose lead-in happened to be
            # short ("The project includes:").
            run = 0
            continue
        if run == 0:
            start = n
        run += 1
        if run >= _TABLE_RUN:
            head = "\n".join(lines[:start]).rstrip()
            # Nothing but table: leave the text alone rather than return an
            # empty description.
            return (head, True) if head else (text, False)
    return text, False


def _trim_sentences(text: str, count: int) -> tuple[str, bool]:
    """Keep the first `count` sentences. Returns (text, was_trimmed).

    Splitting on "period then space" alone cuts inside "Sonny Merryman, Inc.
    for ten buses" and "before 6:30 p.m. on the day of", so a candidate break is
    rejected when the word before it is a known abbreviation or a single
    initial. Nothing is rewritten — the kept sentences are the agenda's own
    characters, so the block quote stays verbatim.
    """
    text = text.strip()
    if not text:
        return "", False
    kept: list[str] = []
    start = 0
    for m in _SENTENCE_END_RE.finditer(text):
        head = text[start:m.start()]
        word = head.split()[-1] if head.split() else ""
        core = word.rstrip(".!?\"'”’)]").casefold()
        if core in _ABBREVIATIONS or (len(core) == 1 and core.isalpha()):
            continue
        kept.append(head)
        start = m.end()
        if len(kept) >= count:
            return " ".join(kept).strip(), bool(text[start:].strip())
    return text, False


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
            text, cut = _strip_flattened_table(text)
            kept, trimmed = _trim_sentences(text, _EXCERPT_SENTENCES)
            return kept, trimmed or cut
    return item.title.strip(), False


# A COST BUDGETED value that carries its OWN label, e.g. "Revised Operating
# Budget Fund: $203,859,000". Every one of these across the fixtures is a fund
# TOTAL rather than the price of the action, which is why "Cost budgeted:
# Revised Operating Budget Fund: $203,859,000" read as nonsense — two labels
# deep, and the outer one contradicting the inner one. The agenda already says
# what the number is; use its label instead of stacking ours on top.
_LABELLED_COST_RE = re.compile(r"^([A-Z][^:$]{2,60}):\s*(\$.+)$")


def _cost_line(item: AgendaItem) -> str | None:
    """The agenda's COST BUDGETED text, when it says anything."""
    sections, _ = body_sections(item.body)
    cost, _ = _trim_paragraphs((sections.get("COST BUDGETED") or "").strip(), 400)
    if not cost or cost == "N/A":
        return None
    cost = " ".join(cost.split())
    if (m := _LABELLED_COST_RE.match(cost)):
        return f"**{m.group(1).strip()}:** {m.group(2).strip().rstrip('.')}"
    return f"**Cost budgeted:** {cost}"


def _signal_line(
    s: ScoredItem,
    *,
    dollars: bool,
    meeting_video: str | None,
    work_session_video: str | None,
    attachment_url: str | None,
) -> str | None:
    """Why this item is a highlight, as the measured signals that ranked it.

    A recap-local twin of `score.evidence_line`, deliberately NOT that function:
    `evidence_line` also feeds the forecast product and the checkpoint printout,
    and this layout hangs links off individual clauses — the watch link on the
    minutes it measures, the packet link on the attachment count — which only
    makes sense in a reader-facing recap.

    Dropped relative to `evidence_line`: the vote outcome (the full tally gets
    its own line below) and the rubric score (an internal ranking input a reader
    has no way to check). Clauses whose data is absent are dropped, so the line
    degrades rather than lying.
    """
    parts: list[str] = []

    def timed(minutes: float, label: str, video: str | None, start: float | None) -> None:
        # The verb goes in front of the FIRST timing clause only. An item
        # measured at both meetings would otherwise read "Discussed 6.3 min at
        # the meeting · Discussed 48.4 min at the work session".
        verb = "Discussed " if not parts else ""
        clause = f"{verb}{minutes:.1f} min {label}"
        if (url := _watch_url(video, start)) and start is not None:
            clause += f" [(Watch)]({url})"
        parts.append(clause)

    if s.signals.meeting_minutes > 0:
        timed(s.signals.meeting_minutes, "at the meeting",
              meeting_video, s.signals.meeting_start_seconds)
    if s.signals.work_session_minutes > 0:
        timed(s.signals.work_session_minutes, "at the work session",
              work_session_video, s.signals.work_session_start_seconds)
    # Older saved score JSON predates the meeting/work-session split, and knows
    # only a total. No "at the ..." to append, and the verb is already in front.
    if not parts and s.signals.discussion_minutes > 0:
        parts.append(f"Discussed {s.signals.discussion_minutes:.1f} min")

    if (n := s.signals.public_comment_speakers):
        parts.append(f"{n} {'person' if n == 1 else 'people'} spoke")
    if s.signals.carry_forward_note:
        parts.append(f"prior comment: {s.signals.carry_forward_note}")
    if dollars and s.signals.dollar_raw:
        parts.append(s.signals.dollar_raw)
    if s.signals.is_routine:
        parts.append("routine")
    if (n := s.signals.attachment_count):
        clause = f"{n} attachment{'s' if n != 1 else ''}"
        # Unlinked when no slice was cut for this item — a Diligent packet
        # embeds its documents, so the only linkable copy is one pdfslice.py
        # wrote. Plain text beats a link that goes nowhere.
        parts.append(f"[{clause}]({attachment_url})" if attachment_url else clause)

    if not parts:
        return None
    return "**Why it's here:** " + " · ".join(parts)


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


def _meeting_label(number: str, period_meetings: list[dict] | None) -> str:
    """Which meeting a namespaced item number belongs to, e.g. "Aug 4 work session".

    `merge.py` prefixes every item number with its meeting ("0804-8.01") because
    both August meetings have an item 7.01. Returns "" for an un-namespaced
    number, which is every single-meeting recap.
    """
    if not period_meetings:
        return ""
    prefix, sep, _ = number.partition("-")
    if not sep:
        return ""
    return next(
        (m["label"] for m in period_meetings if m["numberdate"][4:] == prefix), ""
    )


def _meeting_video(
    number: str,
    period_meetings: list[dict] | None,
    *,
    meeting_video: str | None,
    work_session_video: str | None,
) -> str | None:
    """The video an item's own meeting was recorded on.

    A period recap deep-links each roll call into the meeting that took it, so
    the video is looked up by the item's number prefix — the same key
    `merge.py` namespaced it with. Reading the meeting's LABEL and matching
    "work session" in the prose would break silently the day a label is
    reworded, and the failure is a link into the wrong meeting rather than an
    error.
    """
    if period_meetings:
        prefix = number.partition("-")[0]
        for m in period_meetings:
            if m["numberdate"][4:] == prefix:
                return m.get("video")
    return meeting_video or work_session_video


def _grouped_by_meeting(
    items: list[AgendaItem], period_meetings: list[dict] | None
) -> list[tuple[str, list[AgendaItem]]]:
    """Split a list into (meeting label, items), in meeting order.

    One group with an empty label when this is not a period recap, so the
    caller renders a flat list exactly as it always did.
    """
    if not period_meetings:
        return [("", items)]
    groups: list[tuple[str, list[AgendaItem]]] = []
    for m in period_meetings:
        prefix = m["numberdate"][4:]
        got = [i for i in items if i.number.partition("-")[0] == prefix]
        if got:
            groups.append((m["label"], got))
    placed = {id(i) for _, g in groups for i in g}
    if (rest := [i for i in items if id(i) not in placed]):
        groups.append(("", rest))
    return groups


def _remaining_by_meeting(
    actions: list[AgendaItem],
    consent: list[AgendaItem],
    period_meetings: list[dict] | None,
) -> list[tuple[str, list[AgendaItem], list[AgendaItem]]]:
    """(meeting label, its other action items, its consent agenda), in order.

    Seeded from `period_meetings` so the meetings stay in date order. Building
    the dict from the two passes instead put a meeting whose action items were
    all highlighted — it reaches the dict only on the consent pass — after every
    meeting that still had action items to show.
    """
    by_label: dict[str, tuple[list, list]] = {
        m["label"][:1].upper() + m["label"][1:]: ([], [])
        for m in (period_meetings or [])
    }
    for n, items in ((0, actions), (1, consent)):
        for where, group in _grouped_by_meeting(items, period_meetings):
            label = where[:1].upper() + where[1:] if where else ""
            by_label.setdefault(label, ([], []))[n].extend(group)
    return [(label, a, c) for label, (a, c) in by_label.items() if a or c]


def _remainder_summary(actions: list[AgendaItem], consent: list[AgendaItem]) -> str:
    """One line for a meeting's routine business: how much, and what split.

    These used to be two full bullet lists — 20 titles for August, most of them
    "Approved 6-0" — which buried the two or three that a reader would actually
    stop on. What survives the collapse is the part that carries information:
    the counts, whether everything passed, and the name of anything that did
    NOT pass unanimously. All counted, none judged.
    """
    counts = []
    if actions:
        counts.append(f"{len(actions)} action item"
                      f"{'s' if len(actions) != 1 else ''}")
    if consent:
        counts.append(f"a {len(consent)}-item consent agenda")
    voted = [i for i in actions + consent if i.vote]
    unvoted = len(actions) + len(consent) - len(voted)

    line = " and ".join(counts)
    failed = [i for i in voted if not i.vote.passed]
    if voted and not failed:
        line += ", all approved"
    if unvoted:
        line += f" ({unvoted} with no recorded vote)"
    line += "."

    # Anything that did not pass, or did not pass unanimously, is named: those
    # are the one or two lines of this section a reader might stop on, and a
    # bare count would hide them.
    def named(items: list[AgendaItem]) -> str:
        return "; ".join(f"{i.title} (**{vote_summary(i)}**)" for i in items)

    if failed:
        line += f" Did not pass: {named(failed)}."
    split = [i for i in voted if i.vote.contested and i.vote.passed]
    if split:
        how_many = "One was" if len(split) == 1 else f"{len(split)} were"
        line += f" {how_many} not unanimous: {named(split)}."
    return line


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
        clause = f"{n} {'person' if n == 1 else 'people'} spoke"
        top = public_comment.top_item
        if top and top.count > 1:
            clause += f", {top.count} of them on {_short_title(top.label)}"
        parts.append(clause)

    return "**By the numbers:** " + " · ".join(parts) if parts else None


def _clock(seconds: float) -> str:
    """Seconds from the recording start as m:ss / h:mm:ss.

    This is the model's own start_seconds for the speaker, so it can differ by
    a few seconds from the heading in `out/transcript-meeting-<date>.md`, which
    anchors on the enclosing transcript snippet (13:17 here against the
    transcript's 13:03). Both land inside the same turn at the podium."""
    total = int(seconds)
    h, rem = divmod(total, 3600)
    m, sec = divmod(rem, 60)
    return f"{h}:{m:02d}:{sec:02d}" if h else f"{m}:{sec:02d}"


def _quote_block(
    quote: dict,
    period_meetings: list[dict] | None,
    meeting_video: str | None,
    work_session_video: str | None,
) -> list[str]:
    """A maintainer-chosen quote from the meeting video, with its timestamp.

    The third and last kind of content a highlight can carry, after the agenda's
    own words and the counted tallies — and the only one a person picks rather
    than a parser produces. See `quotes-<period>.json` for the rules the picker
    works to; nothing here is written, ranked or selected by a model.

    The text is the video's AUTOMATIC CAPTIONS, so the attribution says so and
    links the timestamp: the reader verifies by listening, which is the same
    bargain the public-comment anchors make. A quote with no locatable video
    still renders — it is the maintainer's own reading of the recording — but
    without a link to offer.
    """
    text = " ".join((quote.get("text") or "").split())
    if not text:
        return []
    video = meeting_video or work_session_video
    if (numberdate := quote.get("meeting")) and period_meetings:
        for m in period_meetings:
            if m["numberdate"] == numberdate:
                video = m.get("video")
                where = m["label"]
                break
        else:
            where = ""
    else:
        where = ""

    start = quote.get("start_seconds")
    stamp = _clock(start) if start is not None else ""
    if (url := _watch_url(video, start)) and stamp:
        stamp = f"[{stamp}]({url})"
    credit = ", ".join(p for p in (quote.get("speaker"), where) if p)
    tail = " · ".join(p for p in (stamp, "from the automatic captions") if p)

    lines = [f"> {text}"]
    if credit or tail:
        lines += [">", f"> — {credit}" + (f" ({tail})" if tail else "")]
    return lines + [""]


def _speaker_excerpt(anchor, speaker_quotes: dict[str, dict] | None) -> str:
    """The maintainer's short excerpt for one public-comment speaker, if any.

    Keyed by meeting and by the anchor's own start_seconds — the same number
    the timestamp link is built from — so an excerpt cannot drift onto the
    speaker beside it. Missing is the normal case and renders nothing.
    """
    if not speaker_quotes or not anchor.meeting:
        return ""
    return (speaker_quotes.get(anchor.meeting) or {}).get(
        str(int(anchor.start_seconds)), ""
    )


def _speaker_video(
    anchor, period_meetings: list[dict] | None,
    meeting_video: str | None, work_session_video: str | None,
) -> str | None:
    """The video a public-comment speaker's timestamp indexes into."""
    if anchor.meeting and period_meetings:
        for m in period_meetings:
            if m["numberdate"] == anchor.meeting:
                return m.get("video")
    return meeting_video or work_session_video


def _highlight_speakers(
    topic,
    period_meetings: list[dict] | None,
    meeting_video: str | None,
    work_session_video: str | None,
    speaker_quotes: dict[str, dict] | None = None,
) -> list[str]:
    """The speaker tally for ONE highlighted item, inline under that highlight.

    Same rule as the section below: one link per speaker, so the number of
    links IS the count and a reader who doubts it can check in five seconds.
    """
    links = []
    for a in topic.anchors:
        url = _watch_url(
            _speaker_video(a, period_meetings, meeting_video, work_session_video),
            a.start_seconds,
        )
        if not url:
            continue
        row = f"- [{_clock(a.start_seconds)}]({url})"
        if (excerpt := _speaker_excerpt(a, speaker_quotes)):
            row += f" — “{excerpt}”"
        links.append(row)
    n = topic.count
    line = f"**Public comment:** {n} {'person' if n == 1 else 'people'} spoke on this item"
    if not links:
        return [line + ".", ""]
    # One timestamp per line, under a lead-in that says what a timestamp IS.
    # Run together on one row they read as a reference code rather than as an
    # invitation to click, and six of them wrapped mid-link.
    return ([line + ". Each timestamp opens the video at that speaker's turn:", ""]
            + links + [""])


def _public_comment_section(
    public_comment: PublicCommentTally,
    period_meetings: list[dict] | None = None,
    *,
    meeting_video: str | None = None,
    work_session_video: str | None = None,
    moved: set[str] | None = None,
    speaker_quotes: dict[str, dict] | None = None,
) -> list[str]:
    """The speaker tally — a count per topic, ranked, each speaker linked.

    No prose, no names, no paraphrase. Every counted speaker gets a timestamp
    link into the video, so THE NUMBER OF LINKS IS THE COUNT: the bullet proves
    itself, and a reader who doubts a tally can check it in five seconds.

    On-agenda and off-agenda topics are ONE list, not two sections. Splitting
    them separated speakers who were plainly at the meeting together — on Aug 18
    the resident who spoke against the Hispanic Heritage Month resolution and
    the one who spoke against heritage months generally were consecutive at the
    podium, and landed in different halves of the recap because only one of them
    named the agenda item. Ordering by count, then by when the first speaker
    rose, puts them back next to each other without the newsletter deciding they
    were "the same topic" — a judgment it has no way to check.
    """
    def first_start(t) -> float:
        return t.anchors[0].start_seconds if t.anchors else float("inf")

    moved = moved or set()
    topics = sorted(
        [t for t in public_comment.by_item if t.item_number not in moved]
        + list(public_comment.off_agenda),
        key=lambda t: (-t.count, first_start(t), t.label.casefold()),
    )
    if not topics:
        return []

    # The count above the list has to be the count OF the list — the whole
    # point of linking every speaker is that the bullet proves itself, and a
    # month total sitting over a shorter list would break exactly that. So say
    # the total, then say how many of them are up with the highlights.
    lines = ["## More Public Comment", ""]
    n = public_comment.total_speakers
    listed = sum(t.count for t in topics)
    head = f"{n} {'person' if n == 1 else 'people'} spoke during public comment."
    if listed < n:
        head += f" {n - listed} are counted with the highlights above."
    # Same lead-in as under a highlight, and it goes LAST so it sits directly
    # above the list it explains: a bare timestamp does not announce that it is
    # a link into the video at that speaker's turn.
    if any(t.anchors for t in topics):
        head += " Each timestamp opens the video at that speaker's turn."
    lines += [head, ""]

    def video_for(anchor) -> str | None:
        return _speaker_video(
            anchor, period_meetings, meeting_video, work_session_video
        )

    for t in topics:
        # Agenda titles arrive capitalized; off-agenda labels come back in the
        # model's lowercase phrasing, so lift the first letter to match.
        label = t.label if t.item_number else t.label[:1].upper() + t.label[1:]
        tag = ""
        if t.item_number:
            # Item numbers are namespaced by meeting in a period recap; show the
            # number the agenda actually printed, and say which agenda that was.
            number = display_number(t.item_number)
            where = _meeting_label(t.item_number, period_meetings)
            tag = f" (Item {number}, {where})" if where else f" (Item {number})"
        n = t.count
        head = f"- **{label}**{tag} — {n} {'person' if n == 1 else 'people'}"
        links = []
        for a in t.anchors:
            if not (url := _watch_url(video_for(a), a.start_seconds)):
                continue
            row = f"  - [{_clock(a.start_seconds)}]({url})"
            if (excerpt := _speaker_excerpt(a, speaker_quotes)):
                row += f" — “{excerpt}”"
            links.append(row)
        lines.append(head + (":" if links else ""))
        lines += links
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
    period_label: str | None = None,
    period_meetings: list[dict] | None = None,
    packet_paths: dict[str, str] | None = None,
    attachment_paths: dict[str, str] | None = None,
    quotes: dict[str, dict] | None = None,
    speaker_quotes: dict[str, dict] | None = None,
) -> str:
    """Render the post-meeting recap: top highlights + vote-count lists.

    Every word here is either quoted from the agenda, quoted from the video by
    a human (see `quotes`), or counted from parsed data — no model writes any
    of it. Descriptions are the agenda's own
    BACKGROUND text, block-quoted and attributed so a reader can tell the
    district's wording from the newsletter's. Vote tallies come from the
    deterministically parsed `item.vote`, so they are exact against the
    finalized agenda. Per-item "watch" deep-links are built from the
    segmentation start timestamps, which a reader can verify by seeking.
    """
    date_str = _format_date(logistics.date)
    lines: list[str] = []

    # A period recap covers several meetings, so it is titled by the period and
    # names the meetings it read; a single-meeting recap keeps its own date.
    title = "Recap: WJCC School Board"
    if period_label:
        title += f", {period_label}"
    elif date_str:
        title += f", {date_str}"
    lines += [f"# {title}", ""]

    # The meetings this covers, each with its own recording behind "(Watch)".
    # This replaces the footer's single "Watch the full meeting" line, which
    # could only ever name one video of the two.
    agenda_link = _agenda_url(meeting_unique)
    if period_meetings:
        names = []
        for m in period_meetings:
            # Each meeting has its own recording AND its own agenda packet, so
            # both links hang off the meeting they belong to. `agenda` is
            # supplied per meeting (see make_newsletter's --agenda-url); the
            # district publishes no derivable per-meeting URL, so a meeting
            # without one shows Watch alone rather than a link to a landing
            # page that describes a different meeting.
            links = []
            if (url := _watch_url(m.get("video"), None)):
                links.append(f"[Watch]({url})")
            if (url := m.get("agenda")):
                links.append(f"[Agenda]({url})")
            label = m["label"]
            if links:
                label += " (" + " / ".join(links) + ")"
            names.append(label)
        lines += ["*" + " · ".join(names) + "*", ""]
    elif (url := _watch_url(meeting_video or work_session_video, None)):
        # One meeting, so there is no label to hang "(Watch)" off — name the
        # link instead. This is also the line that replaced the footer's old
        # "Watch the full meeting:" for a single-meeting recap.
        lines += [f"*[Watch the full meeting]({url})*", ""]
    lines += [f"[Full agenda]({agenda_link})", ""]

    # --- Highlights (agenda title + the agenda's own description) ---
    lines += ["## Highlights", ""]
    top_numbers = {s.item.number for s in top_items}
    # Public comment on a highlighted item belongs to that highlight, not to a
    # list at the bottom of the page — the speakers and the item are the same
    # story. `_public_comment_section` is told which topics moved so its own
    # count stays honest.
    pc_by_item = {
        t.item_number: t
        for t in (public_comment.by_item if public_comment else [])
        if t.item_number in top_numbers
    }
    for s in top_items:
        lines += [f"### {s.item.title.strip()}", ""]

        cost = _cost_line(s.item)
        signals = _signal_line(
            s,
            dollars=not cost,
            meeting_video=meeting_video,
            work_session_video=work_session_video,
            attachment_url=_attachment_url(s.item, attachment_paths),
        )
        if signals:
            lines += [signals, ""]

        # The maintainer's quote sits under the signals, so the reader learns
        # why the item is here and then hears someone at the meeting say it —
        # before the packet's own wording, which is the driest thing on the page.
        if (quote := (quotes or {}).get(s.item.number)):
            lines += _quote_block(
                quote, period_meetings, meeting_video, work_session_video
            )

        description, trimmed = _item_description(s.item)
        label = "**From the agenda item (excerpt):**" if trimmed else "**From the agenda item:**"
        lines += [label, ""]
        lines += _block_quote(description)
        # Only when something was actually cut: an excerpt the reader has in
        # full needs no "read more", and the link's whole value is that it
        # shows them what the newsletter left out.
        if trimmed and (url := _read_more(s.item.number, packet_paths)):
            lines.append(f"> [… read more]({url})")
        lines.append("")

        if cost:
            lines += [cost, ""]

        # A BoardDocs agenda links its attachments directly, and those URLs are
        # the district's own — worth naming one per line. A Diligent packet
        # embeds them instead, and there the signal line's "N attachments" link
        # to the packet slice is the whole answer, so nothing is listed here.
        if any(a.url for a in s.item.attachments):
            lines += ["**Documents:**", ""]
            lines += [
                f"- [{_attachment_label(a.name)}]({a.url})"
                for a in s.item.attachments if a.url
            ]
            lines.append("")

        tally = _vote_tally_line(
            s.item,
            _meeting_video(
                s.item.number, period_meetings,
                meeting_video=meeting_video, work_session_video=work_session_video,
            ),
        )
        if tally:
            lines += [tally, ""]

        if (topic := pc_by_item.get(s.item.number)):
            lines += _highlight_speakers(
                topic, period_meetings, meeting_video, work_session_video,
                speaker_quotes,
            )

    # --- Public comment on everything the highlights did not cover ---
    if public_comment and public_comment.total_speakers:
        lines += _public_comment_section(
            public_comment, period_meetings,
            meeting_video=meeting_video, work_session_video=work_session_video,
            moved=set(pc_by_item),
            speaker_quotes=speaker_quotes,
        )

    # --- Everything else, counted rather than listed ---
    other_actions = [i for i in action_items if i.number not in top_numbers]
    consent = [i for i in consent_items if i.number not in top_numbers]
    if other_actions or consent:
        lines += ["## Other Agenda Items", ""]
        for where, actions, batch in _remaining_by_meeting(
            other_actions, consent, period_meetings
        ):
            summary = _remainder_summary(actions, batch)
            lines += [f"- **{where}** — {summary}" if where else f"- {summary}"]
        # No "see the full agenda for the rest" line. `_agenda_url` falls back
        # to the district's meetings LANDING page whenever the packet carries
        # no BoardDocs id — which is every Diligent meeting — and that page
        # shows the next meeting, not this month's. The header link is honest
        # about being a starting point; a sentence promising these items are
        # there would not be. (A tally is not in the agenda either way: a
        # Diligent packet is published before the meeting and records no votes.)
        lines.append("")

    # --- Footer ---
    lines += ["## Upcoming Meetings", ""]
    for e in logistics.upcoming_meetings:
        lines.append(f"- {_schedule_line(e)}")
    if logistics.upcoming_meetings:
        lines.append("")
    if logistics.upcoming_events:
        lines += ["**Upcoming events**", ""]
        for e in logistics.upcoming_events:
            lines.append(f"- {_schedule_line(e)}")
        lines.append("")
    lines += [f"**How to participate:** {_participation_note(logistics)}", ""]

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
        # Absent from score JSON saved before the split; evidence_line falls
        # back to discussion_minutes, so an archived run still re-renders.
        s.signals.meeting_minutes = sig.get("meeting_minutes", 0.0)
        s.signals.work_session_minutes = sig.get("work_session_minutes", 0.0)
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
