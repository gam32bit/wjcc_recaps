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


# --- Data model ------------------------------------------------------------

@dataclass
class Attachment:
    name: str
    url: str
    unique: str | None = None


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
