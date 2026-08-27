#!/usr/bin/env python3
"""Render the recap (or the preview newsletter) into paste-ready Markdown.

Deterministic, no LLM, no network. Lays out parsed agenda data, the signal data
from score.py, and the public-comment tally from pubcomment.py.

The recap contains no model-written prose at all. A highlight carries the
maintainer's chosen quote from the meeting video (quotes-<period>.json), a
link to where the item comes up in each recording, the agenda item's own title
and verbatim BACKGROUND text as an attributed block quote, the documents filed
with it by name, the parsed vote tally, and the counted public-comment
speakers as bare timestamps. No line claims how long anything took. The
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


def _who(names: list[str] | None, count: int, state: str) -> str:
    """'Michael Hosang absent', or '1 absent' when nobody was named.

    Names come from the maintainer's `quotes-<period>.json`, never from the
    roll call itself: `votes.py` reads that off an auto-caption track that
    spells one member Kvassos, Kabaso and Vasquez in a single evening. A person
    who has checked the captions against the board's published roster can write
    the name down, with their reasoning, and this prints it. The count is the
    fallback and is always true.

    A supplied list that does not match what the roll call counted is ignored —
    the tally is the record, and a name list disagreeing with it means the file
    is stale, not that the count is wrong.
    """
    if names and len(names) == count:
        return f"{', '.join(names)} {state}"
    return f"{count} {state}"


def _vote_tally_line(
    item: AgendaItem,
    video_url: str | None = None,
    vote_notes: dict[str, dict] | None = None,
) -> str | None:
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
        named = (vote_notes or {}).get(item.number) or {}
        detail = []
        if v.abstain:
            detail.append(_who(named.get("abstain"), len(v.abstain), "abstaining"))
        if v.absent:
            detail.append(_who(named.get("absent"), len(v.absent), "absent"))
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
    """Drop a trailing file-size annotation like ' (905 KB)' and the extension.

    What is left is the district's own filename, which is what the document is
    called and the only description of it anyone has — staff name these things
    ("Redistricting Update - August 4 2026 SB Presentation"), and a name the
    reader can see beats a count they cannot.
    """
    cleaned = re.sub(r"\s*\(\d[\d.,]*\s*[KMG]B\)\s*$", "", name).strip()
    cleaned = re.sub(r"(?i)\.(pdf|docx?|xlsx?|pptx?)$", "", cleaned).strip()
    return cleaned or name


def _attachments_block(item: AgendaItem, slice_url: str | None) -> list[str]:
    """Name every document filed with this item, each linked where one can be.

    A BoardDocs agenda links its attachments directly and those URLs are the
    district's own. A Diligent packet embeds them instead, so the linkable copy
    is the slice `pdfslice.py` cut — one PDF holding that packet page's
    documents, which is why several names can share one link. A name with
    neither is still printed: the reader learns the document exists and can ask
    for it, which "1 attachment" never told them.
    """
    if not item.attachments:
        return []
    rows = []
    for a in item.attachments:
        label = _attachment_label(a.name)
        url = a.url or slice_url
        rows.append(f"- [{label}]({url})" if url else f"- {label}")
    if len(rows) == 1:
        return [f"**Attachment(s):** {rows[0][2:]}", ""]
    return ["**Attachment(s):**", ""] + rows + [""]


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


# The district's own page for the board, which lists every member with their
# contact information. Deliberately the roster page and not a mailbox: the
# board has seven members and no single address reaches them as a body. The
# sentence promises a page of contacts because that is what the link opens.
_CONTACT_NOTE = (
    "**Contact the board:** The school boardmember's contact information is "
    "listed on the division's [School Board page]"
    "(https://wjccschools.org/about-wjcc/leadership/school-board/)."
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


def _schedule_lines(entry) -> list[str]:
    """Footer bullets for one upcoming meeting/event: name, then its facts.

    Date, time and location on their own labelled sub-bullets rather than run
    together after the name. The board's own time strings are a sentence long
    ("Call to Order & Closed Session at 4:00 p.m.; Open Session at 4:30 p.m.")
    and a comma-joined line buried the location behind them. A fact the packet
    did not give us gets no bullet — an empty "When:" would read as a missing
    meeting rather than a missing field.
    """
    rows = [f"- **{entry.name}**"]
    for label, value in (("Date", entry.date), ("Time", entry.time),
                         ("Location", entry.location)):
        if value:
            rows.append(f"  - **{label}:** {value}")
    return rows


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


def _watch_seconds(value) -> float | None:
    """A maintainer's watch override as seconds: 1717, "28:37" or "1:04:53"."""
    if isinstance(value, (int, float)):
        return float(value)
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parts = [int(p) for p in value.strip().split(":")]
    except ValueError:
        return None
    seconds = 0.0
    for part in parts:
        seconds = seconds * 60 + part
    return seconds


def _watch_line(
    s: ScoredItem,
    period_meetings: list[dict] | None,
    meeting_video: str | None,
    work_session_video: str | None,
    watch_starts: dict[str, dict] | None = None,
) -> str | None:
    """Where in the recording this item comes up.

    Default form says WHERE, not how long and not what happened there: an item
    can be a staff presentation, a board discussion, a vote, or all three in
    one evening, and a label the pipeline picked for any of them would be
    wrong for the others — so it renders "from 28:37" and lets the reader
    seek. "from 28:37" also keeps it distinct from the quote's credit
    timestamp directly above, which points at one sentence, not a segment.

    A maintainer who has watched the video can override the anchor per meeting
    in `quotes-<period>.json`'s `watch` block (see `_watch_seconds`):

      "0804-6.01": {"20260804": "28:37"}          # move the single anchor
      "0804-6.01": {"20260804": {"Presentation": "7:20",
                                 "Discussion": "14:18"}}

    The second form renders one "[Watch <label>]" link per entry, in file
    order, with no "Watch this agenda item:" prefix. The label there is a
    person's word for what that timestamp points at — the same bargain the
    `quotes` and `votes` blocks make: a human watched and wrote it down, so
    the recap may name it. When any meeting for an item uses the dict form,
    the item is in explicit-only mode: a meeting with no entry (or an entry
    set to null, meaning "a person checked and there is nothing here") emits
    no link.
    """
    overrides = (watch_starts or {}).get(s.item.number) or {}
    labeled_mode = any(isinstance(v, dict) for v in overrides.values())

    def label(video: str | None, fallback: str) -> tuple[str, str | None]:
        """This meeting's own label from the period list, and its numberdate."""
        for m in (period_meetings or []):
            if video and m.get("video") == video:
                return m["label"], m["numberdate"]
        return fallback, None

    links: list[str] = []
    saw_plain = False
    for video, fallback_label, fallback_start in (
        (work_session_video, "the work session", s.signals.work_session_start_seconds),
        (meeting_video, "the regular meeting", s.signals.meeting_start_seconds),
    ):
        name, numberdate = label(video, fallback_label)
        has_entry = numberdate is not None and numberdate in overrides
        ov = overrides.get(numberdate) if numberdate is not None else None
        if isinstance(ov, dict):
            for seg_label, raw in ov.items():
                at = _watch_seconds(raw)
                if at is not None and (url := _watch_url(video, at)):
                    links.append(f"[Watch {seg_label}]({url})")
            continue
        if labeled_mode and (not has_entry or ov is None):
            continue
        at = _watch_seconds(ov) if ov is not None else fallback_start
        if at is None:
            continue
        if url := _watch_url(video, at):
            links.append(f"[{name}, from {_clock(at)}]({url})")
            saw_plain = True
    if not links:
        return None
    if saw_plain:
        return "**Watch this agenda item:** " + " · ".join(links)
    return " · ".join(links)


def _signal_line(s: ScoredItem, *, dollars: bool) -> str | None:
    """What is known about this item beyond its own text: prior comment, cost.

    A recap-local twin of `score.evidence_line`, deliberately NOT that function:
    `evidence_line` also feeds the forecast product and the checkpoint printout,
    and drops or keeps different clauses than a reader-facing recap wants.

    Dropped relative to `evidence_line`: the vote outcome (the full tally gets
    its own line below), the rubric score (an internal ranking input a reader
    has no way to check), and the attachment count (the documents are named on
    their own line — see `_attachments_block`). Clauses whose data is absent are
    dropped, so the line degrades rather than lying.

    Dropped from this line too, and deliberately: the measured minutes. They
    opened every highlight with an arithmetic claim the reader could not check
    without watching the whole segment, and the number was wrong in a way that
    read as precision — the segmenter is asked for where the BOARD discussed an
    item, so a staff presentation on it counts as zero (August's budget item
    measured 0.5 min because only the vote was captured). The video links moved
    to `_watch_line`, which points at where the item comes up without claiming
    how long it ran. Minutes still rank items; they no longer make a claim on
    the page.
    """
    parts: list[str] = []

    if s.signals.carry_forward_note:
        parts.append(f"prior comment: {s.signals.carry_forward_note}")
    if dollars and s.signals.dollar_raw:
        parts.append(s.signals.dollar_raw)
    if s.signals.is_routine:
        parts.append("routine")

    if not parts:
        return None
    return " · ".join(parts)


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


def _remaining_by_meeting(
    actions: list[AgendaItem],
    consent: list[AgendaItem],
    period_meetings: list[dict] | None,
) -> list[tuple[str, str | None, list[AgendaItem], list[AgendaItem]]]:
    """(label, agenda URL, its other action items, its consent agenda), in order.

    Walked in `period_meetings` order so the meetings stay in date order, and
    matched on the item-number PREFIX rather than on the label: the prefix is
    the key `merge.py` namespaced each number with, and matching prose would
    fail silently into the wrong meeting the day a label is reworded.

    A meeting whose items were all highlighted drops out; anything whose number
    matches no meeting is grouped last under an empty label, so nothing is lost.
    """
    if not period_meetings:
        return [("", None, actions, consent)]
    groups: list[tuple[str, str | None, list, list]] = []
    placed: set[int] = set()
    for m in period_meetings:
        prefix = m["numberdate"][4:]
        mine = tuple(
            [i for i in group if i.number.partition("-")[0] == prefix]
            for group in (actions, consent)
        )
        placed.update(id(i) for g in mine for i in g)
        if any(mine):
            label = m["label"][:1].upper() + m["label"][1:]
            groups.append((label, m.get("agenda"), *mine))
    rest = tuple([i for i in group if id(i) not in placed]
                 for group in (actions, consent))
    if any(rest):
        groups.append(("", None, *rest))
    return groups


def _action_line(item: AgendaItem, vote_notes: dict[str, dict] | None = None) -> str:
    """One routine action item: its own title, and how the board voted.

    A split vote names the members who voted no when — and only when — a
    person has written them into `quotes-<period>.json`'s `votes` block after
    checking the caption roll call against the board's roster (see `_who`);
    otherwise the count stands alone, as it does for absentees.
    """
    tally = vote_summary(item)
    title = " ".join(item.title.split())
    if not item.vote:
        return f"  - {title}"
    line = f"  - {title} — **{tally}**"
    v = item.vote
    if v.source == "transcript" and v.nay:
        named = (vote_notes or {}).get(item.number) or {}
        line += f" ({_who(named.get('nay'), len(v.nay), 'voted no')})"
    return line


def _consent_summary(consent: list[AgendaItem]) -> str:
    """The consent agenda as one line: how many, and anything not unanimous.

    A consent agenda is voted as a block — printing its titles is printing the
    same tally eight times — so it stays a total. What does not collapse is an
    item that failed or split, and the block vote makes that the whole block:
    named, not counted away.
    """
    n = len(consent)
    line = f"Consent agenda: {n} item{'s' if n != 1 else ''}"
    voted = [i for i in consent if i.vote]
    failed = [i for i in voted if not i.vote.passed]
    if voted and not failed:
        line += ", all approved"
    if (unvoted := n - len(voted)):
        line += f" ({unvoted} with no recorded vote)"
    line += "."

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


# Tacked onto the end of the sentence that introduces a list of speaker
# timestamps, not set on its own line: a bare timestamp does not announce that
# it is a link into the recording at that speaker's turn, but the note is a
# gloss on the count it follows, not a heading over the list.
_TIMESTAMP_NOTE = "Timestamps open the meeting recording at each speaker's turn."


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
) -> list[str]:
    """The speaker tally for ONE highlighted item, inline under that highlight.

    Same rule as the section below: one link per speaker, so the number of
    links IS the count and a reader who doubts it can check in five seconds.
    What each of them said is behind their own timestamp, not summarized here.
    """
    def stamp(a) -> str | None:
        url = _watch_url(
            _speaker_video(a, period_meetings, meeting_video, work_session_video),
            a.start_seconds,
        )
        return f"[{_clock(a.start_seconds)}]({url})" if url else None

    n = topic.count
    head = (f"**Public comment:** {n} public comment speaker"
            f"{'s' if n != 1 else ''} on this item.")
    if any(stamp(a) for a in topic.anchors):
        head += f" {_TIMESTAMP_NOTE}"
    lines = [head, ""]

    # NOT grouped by what each speaker asked for. `pubcomment.group_subtopics`
    # still computes those groups and the tally JSON still carries them, but
    # the recap prints bare timestamps: the labels were the model's words, they
    # were the last model-written text on the page, and a label is a
    # description of two minutes of speech that a reader can only check by
    # listening to the two minutes — at which point the label has done nothing
    # the timestamp beside it did not already do.
    links = [f"- {t}" for a in topic.anchors if (t := stamp(a))]
    if not links:
        return lines
    # One timestamp per line, under the count that glosses what a timestamp IS.
    # Run together on one row they read as a reference code rather than as an
    # invitation to click, and six of them wrapped mid-link.
    return lines + links + [""]


def _public_comment_section(
    public_comment: PublicCommentTally,
    period_meetings: list[dict] | None = None,
    *,
    meeting_video: str | None = None,
    work_session_video: str | None = None,
    moved: set[str] | None = None,
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
    # Same gloss as under a highlight, on the same line as the count it
    # explains: a bare timestamp does not announce that it is a link into the
    # video at that speaker's turn.
    if any(t.anchors for t in topics):
        head += f" {_TIMESTAMP_NOTE}"
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
            links.append(f"  - [{_clock(a.start_seconds)}]({url})")
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
    # The maintainer's per-speaker excerpts from `quotes-<period>.json`. Kept
    # in the file and still loaded, but nothing renders them: a short quote
    # beside every timestamp under "More Public Comment" crowded a list whose
    # job is to be countable. Accepted here so the file and its callers stay
    # intact if they are wanted back.
    speaker_quotes: dict[str, dict] | None = None,
    vote_notes: dict[str, dict] | None = None,
    watch_starts: dict[str, dict] | None = None,
    # Maintainer's replacement wording for an off-agenda public-comment topic,
    # keyed by the classifier's own label, casefolded. The one place a person
    # can overrule the model's topic phrasing — reviewed the same way the
    # label itself is; see `quotes-<period>.json`'s `topics` block.
    topic_labels: dict[str, str] | None = None,
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

    # A person's wording for an off-agenda topic replaces the classifier's,
    # before anything downstream reads it. Applied here so every use — the
    # tally list, the lead paragraph — sees the same label.
    if public_comment and topic_labels:
        for t in public_comment.off_agenda:
            if (ov := topic_labels.get(t.label.casefold())):
                t.label = ov

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
    linked_agendas = False
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
                linked_agendas = True
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
    # Only when the line above did not already link an agenda per meeting. Two
    # meetings each carrying their own [Agenda] link, under a third link
    # labelled "Full agenda" that can only point at one of them (or at the
    # district's landing page), is one link too many and the vaguest one wins
    # the reader's attention.
    if not linked_agendas:
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

        # A highlight opens with somebody at the meeting saying something. The
        # measured minutes used to sit here and no longer appear at all (see
        # `_signal_line`): a reader met the item as a number they could not
        # check before they met it as a person talking.
        if (quote := (quotes or {}).get(s.item.number)):
            lines += _quote_block(
                quote, period_meetings, meeting_video, work_session_video
            )

        # Then where to watch the item itself — after the quote, because the
        # quote's own timestamp points at one sentence and this points at the
        # whole segment, and the reader should meet them in that order.
        if (watch := _watch_line(
            s, period_meetings, meeting_video, work_session_video, watch_starts,
        )):
            lines += [watch, ""]

        signals = _signal_line(s, dollars=not cost)
        if signals:
            lines += [signals, ""]

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

        lines += _attachments_block(
            s.item, _attachment_url(s.item, attachment_paths)
        )

        tally = _vote_tally_line(
            s.item,
            _meeting_video(
                s.item.number, period_meetings,
                meeting_video=meeting_video, work_session_video=work_session_video,
            ),
            vote_notes,
        )
        if tally:
            lines += [tally, ""]

        if (topic := pc_by_item.get(s.item.number)):
            lines += _highlight_speakers(
                topic, period_meetings, meeting_video, work_session_video,
            )

    # --- Public comment on everything the highlights did not cover ---
    if public_comment and public_comment.total_speakers:
        lines += _public_comment_section(
            public_comment, period_meetings,
            meeting_video=meeting_video, work_session_video=work_session_video,
            moved=set(pc_by_item),
        )

    # --- Everything else, counted rather than listed ---
    other_actions = [i for i in action_items if i.number not in top_numbers]
    consent = [i for i in consent_items if i.number not in top_numbers]
    if other_actions or consent:
        lines += ["## Other Agenda Items", ""]
        for where, agenda, actions, batch in _remaining_by_meeting(
            other_actions, consent, period_meetings
        ):
            # The meeting's own agenda, so a reader who wants the detail behind
            # one of these lines has somewhere to go. Only the URL the caller
            # supplied for THIS meeting: `_agenda_url`'s fallback is the
            # district's meetings landing page, which shows the next meeting
            # rather than this one, and a link that lands somewhere else is
            # worse here than no link at all.
            rows = [_action_line(i, vote_notes) for i in actions]
            if batch:
                rows.append(f"  - {_consent_summary(batch)}")
            if where or agenda:
                head = f"- **{where}**" if where else "-"
                if agenda:
                    head += f" ([Full agenda]({agenda}))"
                lines.append(head + ":")
                lines += rows
            else:
                # A single-meeting recap has no meeting to head the list with —
                # the whole recap is that meeting — so the items are the list.
                lines += [row.lstrip() for row in rows]
        lines.append("")

    # --- Footer ---
    lines += ["## Upcoming Meetings", ""]
    for e in logistics.upcoming_meetings:
        lines += _schedule_lines(e)
    if logistics.upcoming_meetings:
        lines.append("")
    if logistics.upcoming_events:
        lines += ["**Upcoming events**", ""]
        for e in logistics.upcoming_events:
            lines += _schedule_lines(e)
        lines.append("")
    lines += [f"**How to participate:** {_participation_note(logistics)}", ""]
    lines += [_CONTACT_NOTE, ""]

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
