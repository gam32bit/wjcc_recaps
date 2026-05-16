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
_COST_RE = re.compile(r"\$[\d,]*\d(?:\.\d{2})?")


# --- Data model ------------------------------------------------------------

@dataclass
class Attachment:
    name: str
    url: str
    unique: str | None = None


@dataclass
class AgendaItem:
    number: str          # e.g. "7.01" ("" if the subject has no leading number)
    title: str           # the subject text minus the leading number
    section: str         # the Category value, e.g. "7. Consent Agenda"
    section_num: int     # leading integer of the Category (0 if absent)
    item_type: str       # the Type value, e.g. "Action (Consent)"
    recommended_action: str | None
    body: str            # itembody rendered to Markdown ("" if no itembody)
    costs: list[str] = field(default_factory=list)        # verbatim "$" figures
    attachments: list[Attachment] = field(default_factory=list)


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
            costs=_dedupe(_COST_RE.findall(body)),
            attachments=_attachments(el),
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
        if i.attachments:
            marks.append(f"{len(i.attachments)} file(s)")
        if i.costs:
            marks.append(f"costs: {', '.join(i.costs)}")
        suffix = f"  [{'; '.join(marks)}]" if marks else ""
        print(f"  {i.number or '   -':>6}  {i.item_type:<26}  {i.title[:60]}{suffix}")


if __name__ == "__main__":
    main()
