#!/usr/bin/env python3
"""Parse a BoardDocs agenda HTML file into structured `AgendaItem` records.

Deterministic, no LLM, no network. The agenda HTML saved by `pull_agenda.py`
is cleanly structured (see the Phase 1 log): each item is a
`div.container.item.agendaorder` holding labeled `dl.row` field pairs, an
optional rich-text `div.itembody`, and `div.public-file` attachment links.

Usage:
    python parse.py wjcc-fixtures/agenda-20260519.html         # summary
    python parse.py wjcc-fixtures/agenda-20260519.html --json  # full dump
"""

import argparse
import dataclasses
import json
import pathlib
import re
import sys
from dataclasses import dataclass, field

import html2text
from bs4 import BeautifulSoup

# BoardDocs serves attachment links as host-relative paths; this is the host
# half of the `BASE` constant in pull_agenda.py.
SITE_ROOT = "https://go.boarddocs.com"

_NUMBER_RE = re.compile(r"^(\d+\.\d+)\s+(.*)$", re.DOTALL)
_SECTION_NUM_RE = re.compile(r"^\s*(\d+)")
# Require the figure to end on a digit so trailing punctuation ("$138,766,")
# isn't swallowed into the captured amount.
_DOLLAR_RE = re.compile(r"\$[\d,]*\d(?:\.\d{2})?")
# Post-meeting agendas add a "Motion by X, second by Y." line to each motion.
_MOTION_BY_RE = re.compile(
    r"Motion by\s+(?P<mover>.+?)"
    r"(?:,\s*second(?:ed)?\s+by\s+(?P<seconder>.+?))?\.?\s*$",
    re.IGNORECASE,
)

# WJCC staff write every substantive item body to the same labeled template.
# Matching an explicit label list (rather than a generic ALL-CAPS-colon pattern)
# keeps stray shouty text in the prose from being mistaken for a section head.
BODY_LABELS = (
    "TOPIC",
    "POLICY ALIGNMENT",
    "BACKGROUND",
    "RATIONALE",
    "SPECIAL CIRCUMSTANCE(S)",
    "COST BUDGETED",
    "ALTERNATIVES",
    "SUPERINTENDENT'S RECOMMENDATION",
    "DATA SOURCE",
)
# The label's emphasis markup is inconsistent across items — `_**TOPIC:**_`,
# `**_TOPIC:_**`, `** _BACKGROUND:_**`, `**_RATIONALE:_ **` all occur — and the
# apostrophe in SUPERINTENDENT'S is curly in the source, hence the `.` for it.
# NOTE: do not put `\b` after the `[\s*_]*` prefix. `_` is a word character, so
# `\b` never fires on `**_TOPIC:` and every such item silently falls through.
_BODY_LABEL_RE = re.compile(
    r"(?:(?<=\n)|\A)[\s*_]*("
    + "|".join(re.escape(lbl).replace(r"'", ".") for lbl in BODY_LABELS)
    + r")\s*:[\s*_]*\n*",
    re.IGNORECASE,
)


# --- Data model ------------------------------------------------------------

@dataclass
class Attachment:
    name: str
    url: str
    unique: str | None = None
    # Packet page this document sits on, when it differs from its item's own
    # `source_page`. Only `merge.py` sets it, for an attachment a vote inherited
    # from its work-session review — that document lives on the REVIEW's page,
    # hundreds of pages away from the vote's. None everywhere else.
    page: int | None = None


@dataclass
class Vote:
    """A recorded board vote, present only on finalized (post-meeting) agendas.

    Parsed verbatim from the BoardDocs `Motion & Voting` block. All name lists
    are exactly as printed; tallies are just their lengths, so the outcome is
    deterministic and checkable against the source agenda.
    """
    result: str                 # verbatim, e.g. "Motion Carries" / "Motion Fails"
    motion_text: str            # the motion wording (a verbatim quote source)
    mover: str | None = None
    seconder: str | None = None
    aye: list[str] = field(default_factory=list)
    nay: list[str] = field(default_factory=list)
    abstain: list[str] = field(default_factory=list)
    absent: list[str] = field(default_factory=list)
    # Where this vote came from. "agenda" is the district's own Motion & Voting
    # block — authoritative, and its names are printed. "transcript" is
    # `votes.py` reading the roll call off an auto-captioned video, which is the
    # only source a Diligent packet leaves: its COUNTS are solid (a misheard
    # name is still one voice) but its SPELLINGS are not, so the recap prints
    # the tally and the timestamp and withholds the names.
    source: str = "agenda"
    start_seconds: float | None = None   # roll call location, transcript votes

    @property
    def passed(self) -> bool:
        return "carries" in self.result.lower() or "pass" in self.result.lower()

    @property
    def contested(self) -> bool:
        """True if the vote was not a clean unanimous pass."""
        return bool(self.nay or self.abstain) or not self.passed

    @property
    def tally(self) -> str:
        """Compact for/against tally, e.g. '7-0' or '5-2'."""
        return f"{len(self.aye)}-{len(self.nay)}"


@dataclass
class AgendaItem:
    number: str          # e.g. "7.01" ("" if the subject has no leading number)
    title: str           # the subject text minus the leading number
    section: str         # the Category value, e.g. "7. Consent Agenda"
    section_num: int     # leading integer of the Category (0 if absent)
    item_type: str       # the Type value, e.g. "Action (Consent)"
    recommended_action: str | None
    body: str            # itembody rendered to Markdown ("" if no itembody)
    # Every "$" amount found in the body, verbatim — NOT a single authoritative
    # cost. Includes thresholds, per-unit rates, and even revenue (e.g. a lease
    # the division collects). The headline figure lives in `recommended_action`
    # or the body's labeled COST section; downstream code decides relevance.
    dollar_figures: list[str] = field(default_factory=list)
    attachments: list[Attachment] = field(default_factory=list)
    # Present only on finalized (post-meeting) agendas; None on previews.
    vote: "Vote | None" = None
    # 1-based page of this item's details block in the source packet PDF. Set
    # only by pdfagenda.py — BoardDocs agendas are HTML and have no paging.
    # It is the only usable pointer back to the source for a Diligent packet,
    # whose attachments are embedded rather than linked.
    source_page: int | None = None


# --- Parsing ---------------------------------------------------------------

def _markdown_converter() -> html2text.HTML2Text:
    """An html2text converter configured like pull_agenda.py's (no hard-wrap)."""
    converter = html2text.HTML2Text()
    converter.body_width = 0
    return converter


def _fields(item) -> dict[str, str]:
    """Map each `dt.leftcol` field name to its `dd.rightcol` value text."""
    out: dict[str, str] = {}
    for row in item.select("dl.row"):
        dt = row.select_one("dt.leftcol")
        dd = row.select_one("dd.rightcol")
        if dt and dd:
            out[dt.get_text(strip=True)] = dd.get_text(" ", strip=True)
    return out


def _attachments(item) -> list[Attachment]:
    found: list[Attachment] = []
    for div in item.select("div.public-file"):
        link = div.select_one("a")
        if not link:
            continue
        href = link.get("href", "")
        url = SITE_ROOT + href if href.startswith("/") else href
        found.append(Attachment(
            name=link.get_text(" ", strip=True),
            url=url,
            unique=div.get("unique"),
        ))
    return found


def _names(value: str) -> list[str]:
    """Split a comma-separated voter list into trimmed names."""
    return [n.strip() for n in value.split(",") if n.strip()]


def _parse_vote(item) -> "Vote | None":
    """Parse the BoardDocs `Motion & Voting` block, if this agenda has one.

    Finalized (post-meeting) agendas attach one or more `div.motion` blocks per
    acted-on item. The `finalresolution` block holds the outcome; we take the
    last one (an amended motion supersedes earlier ones). Preview agendas have
    no such block, so this returns None and the item's `vote` stays unset.
    """
    blocks = item.select("div.motion.finalresolution") or item.select("div.motion")
    if not blocks:
        return None
    block = blocks[-1]

    divs = block.find_all("div", recursive=False)
    if not divs:
        return None

    motion_text = divs[0].get_text(" ", strip=True)
    vote = Vote(result="", motion_text=motion_text)

    for div in divs[1:]:
        text = div.get_text(" ", strip=True)
        low = text.lower()
        if low.startswith("motion by"):
            if (m := _MOTION_BY_RE.search(text)):
                vote.mover = (m.group("mover") or "").strip() or None
                vote.seconder = (m.group("seconder") or "").strip() or None
        elif low.startswith("final resolution:"):
            vote.result = text.split(":", 1)[1].strip()
        elif low.startswith("aye:"):
            vote.aye = _names(text.split(":", 1)[1])
        elif low.startswith("nay:"):
            vote.nay = _names(text.split(":", 1)[1])
        elif low.startswith("abstain"):
            vote.abstain = _names(text.split(":", 1)[1])
        elif low.startswith("absent"):
            vote.absent = _names(text.split(":", 1)[1])

    return vote


def _dedupe(values: list[str]) -> list[str]:
    """Order-preserving de-duplication."""
    seen: set[str] = set()
    out: list[str] = []
    for v in values:
        if v not in seen:
            seen.add(v)
            out.append(v)
    return out


def parse_agenda(html: str) -> list[AgendaItem]:
    """Parse agenda HTML into a list of `AgendaItem` records, in agenda order."""
    soup = BeautifulSoup(html, "html.parser")
    converter = _markdown_converter()
    items: list[AgendaItem] = []

    for el in soup.select("div.container.item.agendaorder"):
        f = _fields(el)

        subject = f.get("Subject", "").strip()
        m = _NUMBER_RE.match(subject)
        number, title = (m.group(1), m.group(2).strip()) if m else ("", subject)

        category = f.get("Category", "").strip()
        sec_m = _SECTION_NUM_RE.match(category)
        section_num = int(sec_m.group(1)) if sec_m else 0

        body_el = el.select_one("div.itembody")
        body = converter.handle(str(body_el)).strip() if body_el else ""

        items.append(AgendaItem(
            number=number,
            title=title,
            section=category,
            section_num=section_num,
            item_type=f.get("Type", "").strip(),
            recommended_action=f.get("Recommended Action") or None,
            body=body,
            dollar_figures=_dedupe(_DOLLAR_RE.findall(body)),
            attachments=_attachments(el),
            vote=_parse_vote(el),
        ))

    return items


def body_sections(body: str) -> tuple[dict[str, str], str]:
    """Split an item body into its labeled sections.

    Returns `(sections, lead)` where `sections` maps an upper-cased label from
    BODY_LABELS to its verbatim text, and `lead` is any prose appearing before
    the first label (a handful of older items have no labels at all).

    Text is returned exactly as the staff wrote it — this is what lets the
    newsletter quote the agenda instead of paraphrasing it.
    """
    hits = [
        (m.start(), m.end(), re.sub(r"\s+", " ", m.group(1)).upper())
        for m in _BODY_LABEL_RE.finditer(body)
    ]
    sections: dict[str, str] = {}
    for i, (_, end, label) in enumerate(hits):
        stop = hits[i + 1][0] if i + 1 < len(hits) else len(body)
        # setdefault: if a label somehow repeats, the first occurrence wins.
        sections.setdefault(label, body[end:stop].strip())
    lead = (body[: hits[0][0]] if hits else body).strip()
    return sections, lead


def agenda_preamble(html: str) -> str:
    """Return the text that appears before the first agenda item.

    On a BoardDocs agenda this preamble carries the meeting date, the
    closed/regular session line, and the "watch ... live" livestream notice —
    triage.py mines it for logistics.
    """
    soup = BeautifulSoup(html, "html.parser")
    first = soup.select_one("div.container.item.agendaorder")
    if first is None:
        return soup.get_text("\n", strip=True)
    lines = [s.strip() for s in first.find_all_previous(string=True)]
    return "\n".join(line for line in reversed(lines) if line)


def to_dict(obj) -> object:
    """JSON-friendly conversion shared across the pipeline scripts."""
    return dataclasses.asdict(obj) if dataclasses.is_dataclass(obj) else obj


# --- CLI -------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("html", type=pathlib.Path, help="agenda HTML file")
    parser.add_argument("--json", action="store_true",
                        help="dump the full parsed list as JSON")
    args = parser.parse_args()

    if not args.html.is_file():
        sys.exit(f"No such file: {args.html}")

    items = parse_agenda(args.html.read_text())

    if args.json:
        print(json.dumps([to_dict(i) for i in items], indent=2))
        return

    print(f"{len(items)} agenda items parsed from {args.html.name}")
    for i in items:
        marks = []
        if i.vote:
            marks.append(f"vote {i.vote.tally} ({i.vote.result})")
        if i.attachments:
            marks.append(f"{len(i.attachments)} file(s)")
        if i.dollar_figures:
            marks.append(f"$: {', '.join(i.dollar_figures)}")
        suffix = f"  [{'; '.join(marks)}]" if marks else ""
        print(f"  {i.number or '   -':>6}  {i.item_type:<26}  {i.title[:60]}{suffix}")


if __name__ == "__main__":
    main()
