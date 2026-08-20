#!/usr/bin/env python3
"""Parse a Diligent work-session agenda PDF into `AgendaItem` records.

Deterministic, no LLM, no network — the PDF counterpart to `parse.py`, which
does the same job for BoardDocs HTML. Everything downstream of `AgendaItem`
(triage, score, carryforward, render) is untouched.

WJCC moved from BoardDocs to Diligent Community on 2026-07-01. The new portal
publishes the *work session* packet as a single PDF: a `-layout` outline of
item numbers, titles, attachment names and packet page numbers, followed by one
"Agenda Item Details" page per item written to the same labeled template staff
have always used (TOPIC / BACKGROUND / COST BUDGETED / ...). So the fields the
pipeline depends on all survived the migration; only the container changed.

Why this matters for `forecast.py`: the work session packet IS the forecast's
agenda source. Its section 5, "Proposed Agenda Items", holds the items the
board reviews now and votes on at the regular meeting two weeks later — with
full bodies, two weeks ahead. The regular meeting's own pre-meeting document is
a title-only outline; this is strictly better and earlier.

    python pdfagenda.py "wjcc-fixtures/Work Session ... Aug 04 2026 ....pdf"
    python pdfagenda.py <pdf> --json
"""

import argparse
import dataclasses
import json
import pathlib
import re
import subprocess
import sys

from parse import BODY_LABELS, AgendaItem, Attachment, _DOLLAR_RE, _dedupe

# Sections whose items are reviewed now and voted on at the *next* meeting.
# 5 = "Proposed Agenda Items", 6 = "Information/Discussion Items". Sections 7
# (Consent Agenda) and 8 (Action Items) are disposed of at the work session
# itself, so they are not part of a forecast of the next meeting.
FORECAST_SECTIONS = (5, 6)

# A recap takes every section that carries substance, chosen by the section's
# own TITLE. Numbers cannot do this job: "Consent Agenda" is section 6 on a
# regular packet and section 7 on a work session packet, and section 8 is
# "Board Matters" on one and "Action Items" on the other. A fixed
# `RECAP_SECTIONS = (6, 7)` therefore read the Aug 18 packet correctly and
# silently dropped all seven of the Aug 4 work session's action items.
_RECAP_SECTION_KEYWORDS = (
    "consent",
    "action",
    "information",
    "discussion",
    "polic",
    "proposed agenda",
)

# Item Type strings, keyed off the section's own title. `triage.group_of` sorts
# on this string ("consent" -> consent agenda, "action" -> action items), and
# `triage.DISCUSSION_TYPES` lists the rest; both are matched case-insensitively
# as substrings, so these values must keep those words in them.
#
# Deriving the type from the section TITLE rather than its NUMBER is the whole
# point: section 6 is "Information/Discussion Items" on a work session packet
# and "Consent Agenda" on a regular one. A number-based rule files every
# consent item as discussion, which empties `result.consent_agenda` and
# `result.action_items` without raising anything — the recap would simply lose
# its vote-count lists and nothing would say why.
_TYPE_BY_SECTION_KEYWORD = (
    ("consent", "Action (Consent)"),
    ("action", "Action"),
    ("information", "Information/Discussion"),
    ("discussion", "Information/Discussion"),
    ("polic", "Policies"),
    ("proposed agenda", "Proposed Agenda Item"),
)

# How many pages after an item's own page to scan for the rest of its details
# block. Two is enough for every item observed; the pages beyond that are the
# item's attachments, which can run to hundreds of pages.
DETAIL_SPAN = 2

_SECTION_RE = re.compile(r"^\s{0,6}(\d{1,2})\.\s+(\S.*?)\s*$")
# NOTE: the item number may sit at column 0. `-layout` computes indentation per
# page, so an outline page whose widest line is an item row (rather than a
# section heading) comes out fully dedented — that is what happens on page 3 of
# the Aug 18 regular packet, and an `^\s{3,}` anchor silently dropped items 7.05
# through 7.09 from it, including the two largest contracts on the agenda.
# Requiring only the "N.NN" shape plus a gap is enough: `_SECTION_RE` is tried
# after this and cannot match a two-part number.
_ITEM_RE = re.compile(r"^\s*(\d{1,2}\.\d{2})\s{2,}(\S.*?)\s*$")
_FOOTER_RE = re.compile(r"^\s*Page \d+ of \d+\s*$")
_TIME_RE = re.compile(r"\d{1,2}:\d{2}\s*[ap]\.?m\.?", re.IGNORECASE)
_PAGE_SPLIT_RE = re.compile(r"^\s*Page \d+ of \d+\s*$", re.MULTILINE)
_STOP_RE = re.compile(r"^\s*(Goals:|(?:Consent )?Recommended Action:)", re.IGNORECASE)
_TRAILING_PAGE_RE = re.compile(r"\s{2,}(\d+)\s*$")
_FILE_RE = re.compile(r"\.(pdf|docx?|xlsx?|pptx?)\s*$", re.IGNORECASE)
_DETAILS_RE = re.compile(r"^\s*Agenda Item Details\s*$", re.MULTILINE)

# Staff state an item's destination in its own words, in SUPERINTENDENT'S
# RECOMMENDATION: "be prepared to vote at the August 18, 2026 meeting" versus
# "be prepared to vote later in the same evening". That sentence is the
# authoritative routing signal — it is why 5.01 (Amendment to FY27 Operating
# Fund Budget) must be excluded from an Aug 18 forecast even though it sits in
# section 5: it was reviewed and voted the same night, as its own duplicate at
# 8.07 confirms. Guessing from section number alone would have ranked a settled
# item first, on the largest dollar figure on the agenda.
# NOTE: match across whitespace, not literal spaces. pdftotext preserves the
# PDF's line breaks, and this sentence really does wrap mid-phrase ("...to vote
# at\nthe August 18, 2026 meeting" on item 5.03). A literal-space pattern
# silently dropped that item from the forecast.
_AT_MEETING_RE = re.compile(
    r"at\s+the\s+(?P<date>[A-Z][a-z]+\s+\d{1,2},?\s+\d{4})\s+(?:board\s+)?meeting",
    re.IGNORECASE,
)
_SAME_EVENING_RE = re.compile(
    r"later\s+in\s+the\s+same\s+(?:evening|meeting)", re.IGNORECASE
)

# A details page always carries at least one of the template's labels; an
# attachment page carries none. That is what separates them when an item's
# attachments start within DETAIL_SPAN pages of its details block.
_ANY_LABEL_RE = re.compile(
    r"^[ \t]*(?:" + "|".join(
        re.escape(lbl).replace(r"\'", ".") for lbl in BODY_LABELS
    ) + r")\s*:",
    re.IGNORECASE | re.MULTILINE,
)

_MONTHS = {
    m: i + 1 for i, m in enumerate(
        "january february march april may june july august september october "
        "november december".split()
    )
}


# --- Data model ------------------------------------------------------------

@dataclasses.dataclass
class OutlineEntry:
    """One row of the packet's front-matter outline."""
    number: str
    title: str
    section: str            # e.g. "5. Proposed Agenda Items"
    section_num: int
    page: int | None        # 1-based page of this item's details block
    attachments: list[str] = dataclasses.field(default_factory=list)


# --- pdftotext -------------------------------------------------------------

def extract_text(pdf: pathlib.Path, first: int, last: int, *, layout: bool) -> str:
    """Run `pdftotext` over a page range. Raises if the binary is missing."""
    cmd = ["pdftotext", "-f", str(first), "-l", str(last)]
    if layout:
        cmd.append("-layout")
    cmd += [str(pdf), "-"]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
    except FileNotFoundError:
        raise SystemExit("pdftotext not found — install poppler-utils.")
    return proc.stdout


def _strip_footers(text: str) -> str:
    """Drop the 'Page N of M' running footer, which interrupts wrapped text."""
    return "\n".join(
        line for line in text.splitlines() if not _FOOTER_RE.match(line)
    )


# --- Outline ---------------------------------------------------------------

def _split_title_and_files(
    lines: list[str], title_end: int, margin: int
) -> tuple[list[str], list[str]]:
    """Split an item's continuation lines into title-wrap and attachment names.

    Titles wrap before attachments are listed, and the first attachment usually
    carries a file extension, so a line bearing one closes the title. Later
    attachment names may themselves wrap onto an extension-less line (e.g.
    "... Easement_Covenant Retaining" / "Wall_REVISED.pdf"), which is why names
    are accumulated until an extension is seen rather than taken per-line.

    The extension test alone is not enough: item 7.09 on the Aug 18 packet
    attaches "Clean Copy Acquisition Easement - DEQ Toano MS - James City County
    - (CPM RL) 08.04.2026 (JRE).pdf", whose FIRST line carries no extension and
    so was swallowed into the item's title. The second test is the typesetting
    itself — **a line continues the title only if the line before it was full**,
    meaning this line's first word could not have fit there. Both columns are
    the same width and the same indent, so this is the only structural
    difference left; nothing about the words themselves distinguishes an
    easement title from an easement deed's filename.

    `title_end` is the column where the item line's own title text stops, and
    `margin` the widest such column in the outline — its right-hand text edge.
    """
    title_parts: list[str] = []
    idx = 0
    while idx < len(lines):
        line = lines[idx]
        if _FILE_RE.search(line):
            break
        words = line.split()
        if not words:
            break
        if title_end + 1 + len(words[0]) <= margin:
            # The previous line had room for this line's first word, so the
            # previous line ended a sentence rather than running out of space.
            break
        title_parts.append(line.strip())
        title_end = len(line.rstrip())
        idx += 1

    files: list[str] = []
    buffer: list[str] = []
    for line in lines[idx:]:
        buffer.append(line.strip())
        if _FILE_RE.search(line):
            files.append(" ".join(buffer))
            buffer = []
    if buffer:
        files.append(" ".join(buffer))
    return title_parts, files


def _title_end(line: str, m: "re.Match[str]") -> int:
    """Column where an item line's title text stops, ignoring its page number."""
    rest = m.group(2)
    if (pm := _TRAILING_PAGE_RE.search(rest)):
        rest = rest[: pm.start()].rstrip()
    return line.index(m.group(2)) + len(rest)


def parse_outline(text: str) -> list[OutlineEntry]:
    """Parse the packet's front-matter outline into one entry per agenda item."""
    # The outline page count is an over-estimate, so the tail of `text` may run
    # into the first item's details block. Cut there: past that point every
    # line belongs to an item body, not to the outline.
    if (head := _DETAILS_RE.search(text)):
        text = text[: head.start()]
    # Split on "\n" rather than with `splitlines()`, which treats the form feed
    # as a line break of its own and so eats the page markers this needs.
    raw_lines = [ln for ln in text.split("\n") if not _FOOTER_RE.match(ln)]
    entries: list[OutlineEntry] = []
    section, section_num = "", 0

    # Page each line belongs to. `pdftotext` marks a page break with a form feed
    # on the first line of the new page; it is blanked afterwards so it does not
    # shift that line's columns.
    page_of: list[int] = []
    lines: list[str] = []
    page = 0
    for raw in raw_lines:
        if "\f" in raw:
            page += 1
            raw = raw.replace("\f", " ")
        lines.append(raw)
        page_of.append(page)

    # The outline's right-hand text edge, measured rather than assumed: the
    # widest column an item's title text reaches ON ITS OWN PAGE.
    # `_split_title_and_files` needs it to tell a wrapped title from an
    # attachment name. Two things make it per-page rather than global:
    # `-layout` recomputes indentation per page, and the title column is bounded
    # by the page-number column, which widens with the page count (p.31 vs
    # p.1599). Measured globally it comes out 90 — the widest line on the last
    # outline page — and every line on page 2 then looks like it had room to
    # spare, which turns the test off.
    #
    # Only lines carrying a page number count: a title column is bounded on the
    # right by that number, but the procedural rows have none and run past it.
    margins: dict[int, int] = {}
    for idx, line in enumerate(lines):
        if (m := _ITEM_RE.match(line)) and _TRAILING_PAGE_RE.search(m.group(2)):
            pg = page_of[idx]
            margins[pg] = max(margins.get(pg, 0), _title_end(line, m))

    pending: OutlineEntry | None = None
    pending_end = 0
    pending_margin = 0
    buffer: list[str] = []

    def flush() -> None:
        nonlocal pending, buffer
        if pending is None:
            return
        extra, files = _split_title_and_files(buffer, pending_end, pending_margin)
        if extra:
            pending.title = " ".join([pending.title, *extra])
        pending.attachments = files
        entries.append(pending)
        pending, buffer = None, []

    for idx, line in enumerate(lines):
        if not line.strip():
            continue
        if (m := _ITEM_RE.match(line)):
            flush()
            rest = m.group(2)
            packet_page = None
            if (pm := _TRAILING_PAGE_RE.search(rest)):
                packet_page = int(pm.group(1))
                rest = rest[: pm.start()].rstrip()
            pending_end = _title_end(line, m)
            pending_margin = margins.get(page_of[idx], 0)
            pending = OutlineEntry(
                number=m.group(1), title=rest,
                section=section, section_num=section_num, page=packet_page,
            )
            continue
        if (m := _SECTION_RE.match(line)):
            flush()
            section_num = int(m.group(1))
            section = f"{section_num}. {m.group(2)}"
            continue
        if _STOP_RE.match(line):
            # "Goals:" / "Recommended Action:" end the title+attachment block.
            # Everything after belongs to neither, so stop collecting.
            if pending is not None:
                flush()
            continue
        if pending is not None:
            # Kept unstripped: the column a line ends at is the signal
            # `_split_title_and_files` reads.
            buffer.append(line.rstrip())

    flush()
    return entries


# --- Item details ----------------------------------------------------------

def parse_detail(text: str) -> str:
    """Return one item's details block from the raw text of its pages.

    Two boundaries have to hold, and only using both keeps attachment text out
    of the body:

    - The next item's "Agenda Item Details" header ends this item's block.
    - An item's own attachments follow immediately and carry no such header, so
      trailing pages are kept only while they still contain template labels.

    The second rule is not cosmetic. Items 5.09 and 5.10 are $10.00 easement
    conveyances whose attached deeds recite property values above $7,000,000.
    Without the page filter those figures land in `dollar_figures`, and
    `dollar_magnitude` is 0.20 of the composite — the two items jump to the top
    of the forecast on a number the board is not voting on.
    """
    pages = _PAGE_SPLIT_RE.split(text)
    kept = [pages[0]]
    for page in pages[1:]:
        if not _ANY_LABEL_RE.search(page):
            break
        kept.append(page)

    body = _strip_footers("\n".join(kept))
    heads = list(_DETAILS_RE.finditer(body))
    if not heads:
        return ""
    start = heads[0].end()
    stop = heads[1].start() if len(heads) > 1 else len(body)
    return body[start:stop].strip()


def _iso(date_text: str) -> str:
    """'August 18, 2026' -> '20260818'. Returns '' if unrecognized.

    Whitespace is collapsed first: the phrase wraps across a line break often
    enough ("August 4,\\n2026") that matching literal spaces loses real items.
    """
    text = re.sub(r"\s+", " ", date_text).strip()
    m = re.match(r"([A-Za-z]+) (\d{1,2}),? (\d{4})", text)
    if not m or m.group(1).lower() not in _MONTHS:
        return ""
    return f"{m.group(3)}{_MONTHS[m.group(1).lower()]:02d}{int(m.group(2)):02d}"


def routed_to(body: str) -> tuple[str, str, str]:
    """Where this item's vote goes, read from the staff's own wording.

    Returns `(status, numberdate, verbatim)` with status one of:

    - `same_evening` — voted at the work session itself, so it will not appear
      on the next meeting's agenda.
    - `named` — an explicit "at the <date> meeting"; `numberdate` is that date.
    - `unstated` — no routing sentence at all.

    `unstated` must NOT be read as "not carried". Checked against the July 7
    packet, whose section 5 items all reappeared on August 4: two of the four
    named the meeting and two said nothing, yet all four carried. Section 5 is
    titled "Proposed Agenda Items", so carrying forward is its default and the
    sentence only confirms or overrides it.
    """
    if (m := _SAME_EVENING_RE.search(body)):
        return "same_evening", "", re.sub(r"\s+", " ", m.group(0))
    if (m := _AT_MEETING_RE.search(body)):
        return "named", _iso(m.group("date")), re.sub(r"\s+", " ", m.group(0))
    return "unstated", "", ""


# --- Assembly --------------------------------------------------------------

def recap_sections(entries: list[OutlineEntry]) -> tuple[int, ...]:
    """Section numbers a recap should read, picked by section title.

    "Board Matters/Requests" and "Meeting Schedule" carry no items with bodies;
    everything a board votes on or is briefed on lives under one of the
    keywords. Returned as numbers because that is what `build_items` filters on.
    """
    return tuple(sorted({
        e.section_num for e in entries
        if any(k in e.section.lower() for k in _RECAP_SECTION_KEYWORDS)
    }))


def _item_type(section: str) -> str:
    """Map a section title ("6. Consent Agenda") to an Item Type string.

    The values match what `parse.py` reads out of BoardDocs, so `triage` and
    `score` sort a Diligent item exactly as they sort an HTML one. Falls back to
    the work session's own wording, which is what an unrecognized section on
    that packet has always been.
    """
    text = section.lower()
    for keyword, item_type in _TYPE_BY_SECTION_KEYWORD:
        if keyword in text:
            return item_type
    return "Proposed Agenda Item"


def build_items(
    pdf: pathlib.Path,
    *,
    sections: tuple[int, ...] | None = FORECAST_SECTIONS,
    outline_pages: int = 8,
    target: str | None = None,
    verbose: bool = True,
) -> tuple[list[AgendaItem], str]:
    """Parse `pdf` into AgendaItems for the given sections.

    When `target` is a numberdate, items are kept only if the packet routes
    them to that meeting — the exclusion is printed with its verbatim reason so
    the decision is inspectable rather than implicit. Information/discussion
    items name no meeting and are kept regardless, since they carry the work
    session's discussion time, which is the forecast's heaviest signal.

    Returns `(items, outline_text, decisions)`, where `decisions` records the
    keep/skip call and its verbatim reason for every candidate item.
    """
    outline_text = extract_text(pdf, 1, outline_pages, layout=True)
    entries = parse_outline(outline_text)
    # `sections=None` means "every substantive section, whatever it is numbered
    # in this packet" — the recap's rule. See `recap_sections`.
    chosen = recap_sections(entries) if sections is None else sections
    wanted = [e for e in entries if e.section_num in chosen and e.page]

    items: list[AgendaItem] = []
    decisions: list[dict] = []
    for entry in wanted:
        raw = extract_text(pdf, entry.page, entry.page + DETAIL_SPAN, layout=False)
        body = parse_detail(raw)
        status, destination, reason = routed_to(body)

        item_type = _item_type(entry.section)
        is_discussion = item_type == "Information/Discussion"
        # Information items name no meeting because they carry no vote. They are
        # kept anyway: they hold the work session's discussion time, the
        # forecast's heaviest signal, and June's recap #1 was exactly such an
        # item. Whether one reappears on the regular agenda is genuinely
        # unknown, so it is labelled rather than quietly ranked.
        if not target:
            # No routing question to answer: this is the meeting's own packet,
            # so every item on it belongs to this meeting by definition.
            note = f"on this meeting's own agenda ({entry.section})"
        elif is_discussion:
            note = ("information/discussion item — carried for its work-session "
                    "time; not confirmed on the regular meeting agenda")
        if target and not is_discussion:
            if status == "same_evening" or (status == "named" and destination != target):
                if verbose:
                    print(f"  skip [{entry.number}] {entry.title[:50]} — \"{reason}\"")
                decisions.append({
                    "number": entry.number, "title": entry.title,
                    "kept": False, "reason": reason,
                })
                continue
            note = f'"{reason}"' if status == "named" else "ASSUMED (no routing sentence)"
        decisions.append({
            "number": entry.number, "title": entry.title,
            "kept": True, "reason": note,
        })

        items.append(AgendaItem(
            number=entry.number,
            title=entry.title,
            section=entry.section,
            section_num=entry.section_num,
            # Read off the section's title, never its number — see
            # `_TYPE_BY_SECTION_KEYWORD`. On a work session packet every value
            # lands in triage.DISCUSSION_TYPES, which is what these items are
            # there: reviewed, not yet voted, so is_vote and off_consent read
            # False for all of them and both signals go uniform — a no-op once
            # finalize() min-max normalizes. See the brief's "What this can't
            # see". On a regular packet the same code yields real Action and
            # Action (Consent) types, and both signals measure again.
            item_type=item_type,
            recommended_action=None,
            body=body,
            dollar_figures=_dedupe(_DOLLAR_RE.findall(body)),
            # The packet embeds attachments rather than linking them, so there
            # is no URL to give. The page number is the usable affordance: it
            # is seekable in the PDF the reader already has.
            attachments=[
                Attachment(name=name, url="", unique=None)
                for name in entry.attachments
            ],
            vote=None,
            source_page=entry.page,
        ))
        if verbose:
            print(f"  keep [{entry.number}] p.{entry.page:<5} "
                  f"{entry.title[:46]:<48} {note}")

    return items, outline_text, decisions


# --- Fixtures --------------------------------------------------------------

# The table runs from its heading to the next numbered section or item — or,
# on a regular packet where "Upcoming Meetings" is the last thing in the front
# matter, to the first item's "Agenda Item Details" page.
_SCHEDULE_RE_TMPL = (
    r"{}(.*?)(?=^\s{{0,6}}\d{{1,2}}\.\s+\S"
    r"|^\s{{0,8}}\d{{1,2}}\.\d{{2}}\s{{2,}}\S"
    r"|^\s*Agenda Item Details\s*$)"
)
_GAP_RE = re.compile(r"\S\s{3,}\S")
_HEADER_ROW_RE = re.compile(r"(?:Meeting|Event)\s*/\s*Date")
_DATE_RE = re.compile(r"\b\d{1,2}/\d{1,2}/\d{2,4}\b")


def _right_column(block: list[str]) -> int:
    """Column where the right cell starts in one row-block, or 0 if unclear.

    A row's right cell wraps onto lines of its own, indented past everything in
    the left cell — so the shallowest such indent is the column boundary. When a
    block has no wrapped right line (a one-line row), fall back to the widest
    intra-line gap.
    """
    indents = [len(ln) - len(ln.lstrip()) for ln in block]
    base = min(indents)
    deeper = [i for i in indents if i > base]
    if deeper:
        return min(deeper)
    for line in block:
        if (g := _GAP_RE.search(line)):
            return g.end() - 1
    return 0


def _schedule_rows(outline_text: str, heading: str = "Upcoming Meetings") -> list[tuple[str, list[str]]]:
    """Parse one of the packet's two-column schedule tables into rows.

    `-layout` preserves the columns as whitespace, and a BLANK LINE separates
    one row from the next — which is the only reliable row boundary, because
    both cells wrap and a wrapped left line looks exactly like a new row. Each
    block is then split at its own column boundary, since `-layout` recomputes
    indentation per page and the table straddles a page break.

    Returns `(left_cell, right_cell_lines)` per row, or **[] for the whole
    table** if any block cannot be split cleanly. The failure this guards
    against is real and unfixable here: in the Aug 18 packet's Upcoming Events
    table the left cell "Advisory Committee" runs to within one space of the
    right cell, so there is no column to cut at and a positional split lands
    mid-word ("School Bo" / "ard and Central Office"). Publishing a mangled
    meeting time is worse than publishing none, so the table is dropped whole
    rather than emitted damaged.
    """
    m = re.search(
        _SCHEDULE_RE_TMPL.format(re.escape(heading)), outline_text, re.S | re.M
    )
    if not m:
        return []

    blocks: list[list[str]] = []
    current: list[str] = []
    for line in m.group(1).splitlines():
        if not line.strip() or _FOOTER_RE.match(line):
            if current:
                blocks.append(current)
                current = []
            continue
        if _HEADER_ROW_RE.search(line):
            continue
        current.append(line.replace("\f", " "))
    if current:
        blocks.append(current)

    rows: list[tuple[str, list[str]]] = []
    for block in blocks:
        right_col = _right_column(block)
        if not right_col:
            return []
        left_parts: list[str] = []
        right_parts: list[str] = []
        for line in block:
            padded = line.ljust(right_col + 1)
            # Both sides of the cut must be whitespace-adjacent; otherwise the
            # boundary is inside a word and this table is not recoverable.
            if padded[right_col - 1].strip() and padded[right_col].strip():
                return []
            if (left := line[:right_col].strip()):
                left_parts.append(left)
            if (right := line[right_col:].strip()):
                right_parts.append(right)
        if left_parts or right_parts:
            rows.append((" ".join(left_parts), right_parts))
    return rows


def _join_wrapped(detail: list[str]) -> list[str]:
    """Rejoin a right-column fact split across lines.

    The right column wraps too, so a fragment that opens with a digit or a
    lowercase word continues the previous line ("... in Room" + "200") rather
    than starting a new fact.
    """
    parts: list[str] = []
    for fragment in detail:
        if parts and (fragment[:1].isdigit() or fragment[:1].islower()):
            parts[-1] = f"{parts[-1]} {fragment}"
        else:
            parts.append(fragment)
    return parts


def schedule_entries(outline_text: str, heading: str) -> list[dict]:
    """One schedule table as `triage.ScheduleEntry` field dicts.

    A BoardDocs agenda carries these as pipe tables inside a Meeting Schedule
    agenda item, which `triage._parse_schedule_table` reads. A Diligent packet
    puts them in its front matter instead, outside any item — so they are parsed
    here and travel in the meeting-meta fixture, where `extract_logistics` picks
    them up. Every field stays verbatim; nothing is reformatted or inferred.

    Fail-closed, like `_schedule_rows`: every row must yield both a name and a
    date, or the whole table is dropped. A footer entry is only worth printing
    if a reader can put it in their calendar.
    """
    entries: list[dict] = []
    for label, detail in _schedule_rows(outline_text, heading):
        date = m.group(0) if (m := _DATE_RE.search(label)) else ""
        name = _DATE_RE.sub("", label).strip(" ,")
        if not (name and date):
            return []
        parts = _join_wrapped(detail)
        # The first fragment carrying a clock time is the time; everything else
        # is where. Neither is split further — a packet row like "Call to Order
        # & Closed Session at 4:00 p.m." is one fact, and cutting it would lose
        # which session the time belongs to.
        time_parts = [p for p in parts if _TIME_RE.search(p)]
        where = [p for p in parts if not _TIME_RE.search(p)]
        entries.append({
            "name": name,
            "date": date,
            "time": "; ".join(time_parts),
            "location": "; ".join(where),
        })
    return entries


def _header_name(outline_text: str) -> str:
    """The packet's own cover lines, as a BoardDocs-style meeting `name`.

    A regular packet opens with "Regular Meeting - Agenda / Tuesday, August 18,
    2026 / <school, address> / Call to Order & Closed Session beginning at 6:00
    p.m. ... followed by the Open Session at 6:30 p.m. ...". Those last lines
    are exactly what `triage._parse_sessions` reads out of a BoardDocs meeting
    name, so they are joined verbatim rather than rewritten.
    """
    lines: list[str] = []
    for raw in outline_text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if _SECTION_RE.match(raw) or line == "Page":
            break
        lines.append(line)
    session_lines = [ln for ln in lines if _TIME_RE.search(ln)]
    return " ".join(session_lines or lines[:1])


def packet_kind(outline_text: str) -> str:
    """"worksession" or "regular", from the packet's own cover line.

    The cover reads "Work Session / Action Items - Agenda" or "Regular Meeting -
    Agenda". It decides which video signal the meeting's transcript feeds, and
    it has to come from the document rather than the date: the board holds both
    kinds on a Tuesday.
    """
    for raw in outline_text.splitlines():
        line = raw.strip().lower()
        if not line:
            continue
        return "worksession" if "work session" in line else "regular"
    return "regular"


def meeting_meta(outline_text: str, numberdate: str) -> dict:
    """Build a `meeting-meta-<date>.json` payload for the meeting being forecast.

    The work session packet lists the next regular meeting's time and place in
    its "Upcoming Meetings" table, so these facts come from the source document
    rather than being typed in — the same rule the rest of the pipeline follows
    for dates, times and locations.
    """
    target = f"{int(numberdate[4:6])}/{int(numberdate[6:8])}/{numberdate[:4]}"
    meetings = schedule_entries(outline_text, "Upcoming Meetings")
    name = ""
    for entry in meetings:
        if entry["date"] == target:
            detail = "; ".join(p for p in (entry["time"], entry["location"]) if p)
            name = f"{entry['name']}: {detail}" if detail else entry["name"]
            break
    return {
        "numberdate": numberdate,
        "name": name,
        "source": f"diligent-{packet_kind(outline_text)}-pdf",
        # A Diligent packet keeps its schedule tables in the front matter rather
        # than in an agenda item, so they ride along here for extract_logistics.
        "upcoming_meetings": meetings,
        "upcoming_events": schedule_entries(outline_text, "Upcoming Events"),
    }


def write_fixtures(
    pdf: pathlib.Path,
    numberdate: str,
    fixtures_dir: pathlib.Path,
    *,
    recap: bool = False,
    suffix: str = "",
) -> tuple[pathlib.Path, pathlib.Path]:
    """Write the agenda + meta fixtures the pipeline loads.

    `recap=True` reads the meeting's OWN packet: its consent agenda and action
    items, with no routing filter (the meeting has happened; nothing is being
    projected forward).

    `suffix` goes into both filenames, because one meeting can have two agenda
    documents. Aug 18 2026 has the Aug 4 work-session packet's projection of it
    — the forecast's input, saved unsuffixed as `agenda-20260818.json` — and its
    own regular-meeting packet, the recap's input. Writing the second over the
    first would destroy the evidence the forecast was scored from.
    """
    if recap:
        items, outline_text, decisions = build_items(
            pdf, sections=None, target=None
        )
    else:
        items, outline_text, decisions = build_items(pdf, target=numberdate)

    agenda_path = fixtures_dir / f"agenda-{numberdate}{suffix}.json"
    agenda_path.write_text(json.dumps({
        "numberdate": numberdate,
        "source_pdf": pdf.name,
        "preamble": outline_text,
        "decisions": decisions,
        "items": [dataclasses.asdict(i) for i in items],
    }, indent=2))

    meta = meeting_meta(outline_text, numberdate)
    # A work session packet is read two ways: as the FORECAST's view of the next
    # regular meeting, and as the RECAP of the work session itself. Both would
    # otherwise look identical to `--period`, which would then count the same
    # packet twice under two different dates.
    meta["role"] = "recap" if recap else "forecast"
    if recap:
        # The regular packet's "Upcoming Meetings" table lists the meetings
        # AFTER this one, so it never names the meeting itself — `meeting_meta`
        # correctly finds nothing. The logistics for the meeting being recapped
        # are printed in the packet's own header instead.
        meta["name"] = meta["name"] or _header_name(outline_text)
        # The public-comment item is boilerplate and carries no details page, so
        # it never becomes an AgendaItem — but its title IS the speaking rule
        # the recap footer prints, so lift it from the outline.
        meta["public_comment_rule"] = next(
            (e.title for e in parse_outline(outline_text)
             if "public comment" in e.title.lower()),
            "",
        )
    meta_path = fixtures_dir / f"meeting-meta-{numberdate}{suffix}.json"
    meta_path.write_text(json.dumps(meta, indent=2))
    return agenda_path, meta_path


# --- CLI -------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("pdf", type=pathlib.Path)
    parser.add_argument("--target", help="numberdate of the meeting being forecast")
    parser.add_argument("--json", action="store_true", help="dump AgendaItems as JSON")
    parser.add_argument("--outline", action="store_true", help="dump the parsed outline")
    parser.add_argument(
        "--fixtures", action="store_true",
        help="write wjcc-fixtures/agenda-<target>.json and meeting-meta-<target>.json",
    )
    parser.add_argument(
        "--recap", action="store_true",
        help="this is the meeting's OWN regular packet: read its consent agenda "
        "and action items, with no routing filter",
    )
    parser.add_argument(
        "--suffix", default="",
        help="appended to both fixture filenames, so a meeting's regular packet "
        "does not overwrite the work-session packet's projection of it "
        "(e.g. --suffix -regular)",
    )
    args = parser.parse_args()

    if not args.pdf.is_file():
        sys.exit(f"No such file: {args.pdf}")

    if args.outline:
        for e in parse_outline(extract_text(args.pdf, 1, 8, layout=True)):
            files = f"  [{len(e.attachments)} file(s)]" if e.attachments else ""
            print(f"{e.section_num:>3} {e.number:>6}  p.{str(e.page or '-'):<5} "
                  f"{e.title[:70]}{files}")
        return

    if args.fixtures:
        if not args.target:
            sys.exit("--fixtures requires --target (the meeting this packet is for)")
        agenda_path, meta_path = write_fixtures(
            args.pdf, args.target,
            pathlib.Path(__file__).resolve().parent / "wjcc-fixtures",
            recap=args.recap, suffix=args.suffix,
        )
        print(f"\nWrote {agenda_path}\nWrote {meta_path}")
        return

    items, _, _ = build_items(
        args.pdf,
        sections=None if args.recap else FORECAST_SECTIONS,
        target=None if args.recap else args.target,
        verbose=not args.json,
    )
    if args.json:
        print(json.dumps([dataclasses.asdict(i) for i in items], indent=2))
        return

    print(f"\n{len(items)} items parsed from {args.pdf.name}")
    for i in items:
        marks = []
        if i.dollar_figures:
            marks.append(f"$: {', '.join(i.dollar_figures[:4])}")
        if i.attachments:
            marks.append(f"{len(i.attachments)} file(s)")
        print(f"  {i.number:>6}  {i.title[:56]:<58}{'; '.join(marks)}")


if __name__ == "__main__":
    main()
