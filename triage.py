#!/usr/bin/env python3
"""Triage a parsed agenda: drop boilerplate, group items, extract logistics.

Deterministic, no LLM, no network. This is the step that decides what the
later Claude call will (and won't) see:

  * boilerplate items (call to order, roll, pledge, adjournment, public
    comment, board housekeeping) are dropped by section + title rules;
  * the survivors are grouped by item Type — robust to Work Sessions, which
    have no dedicated "Action Items" section;
  * meeting logistics (date, session times/locations, livestream URL, the
    public-comment rule) are pulled out into a `Logistics` record that is
    NEVER sent to the LLM, so those facts can't be hallucinated.

Usage:
    python triage.py wjcc-fixtures/agenda-20260519.html \\
                      wjcc-fixtures/meeting-meta-20260519.json
    python triage.py <agenda.html> <meeting-meta.json> --json
"""

import argparse
import datetime as dt
import json
import pathlib
import re
import sys
from dataclasses import dataclass, field

from parse import AgendaItem, agenda_preamble, parse_agenda, to_dict

# --- Triage rules ----------------------------------------------------------

# Whole sections that never carry newsletter-worthy substance for a meeting
# *preview* — matched as a case-insensitive substring against an item's Category
# text. Recognitions, announcements, and spotlights are real meeting content,
# but they have nothing to preview (no decision, no figures, no item body), so
# they are dropped here rather than surfaced as empty one-liners.
BOILERPLATE_SECTIONS = (
    "meeting opening",
    "closed session",
    "citizens' comment",
    "public comment",
    "board matters",
    "meeting schedule",
    "set agenda",
    "adjournment",
    "recognition",
    "announcement",
    "superintendent's report",
    "spotlight",
)

# Individual procedural items, matched against the item title. These catch
# procedural rows that live inside an otherwise-substantive section (e.g. the
# May 19 "3. Regular Meeting" section).
BOILERPLATE_TITLES = (
    "call to order",
    "the roll",
    "pledge of allegiance",
    "minute of silence",
    "reconvenes",
    "certifies closed",
    "approval of the agenda",
    "approval of minutes",
    "public comment",
    "adjourn",
)

# Item Type values that legitimately belong in the discussion group; anything
# else that falls through to discussion is flagged as an unmatched type.
DISCUSSION_TYPES = (
    "information/discussion",
    "proposed agenda item",
    "policies",
)

_URL_RE = re.compile(r"(?:https?://|www\.)[^\s,;)]+", re.IGNORECASE)
_TIME_RE = re.compile(r"\d{1,2}:\d{2}\s*[ap]\.?m\.?", re.IGNORECASE)


# --- Data model ------------------------------------------------------------

@dataclass
class Session:
    name: str               # e.g. "Closed Session", "Regular Meeting"
    time: str | None        # e.g. "6:00 p.m."
    location: str | None    # e.g. "Room 200 at Berkeley Middle School, ..."


@dataclass
class Logistics:
    date: str | None                 # ISO date, e.g. "2026-05-19"
    meeting_name: str                # the raw BoardDocs meeting `name`
    sessions: list[Session] = field(default_factory=list)
    livestream: str | None = None
    public_comment_rule: str | None = None
    next_meeting: str | None = None  # wired up in a later step


@dataclass
class TriageResult:
    logistics: Logistics
    action_items: list[AgendaItem] = field(default_factory=list)
    consent_agenda: list[AgendaItem] = field(default_factory=list)
    discussion: list[AgendaItem] = field(default_factory=list)
    report: dict = field(default_factory=dict)


# --- Filtering & grouping --------------------------------------------------

def is_boilerplate(item: AgendaItem) -> bool:
    """True if the item is procedural and should never reach the newsletter."""
    section = item.section.lower()
    if any(kw in section for kw in BOILERPLATE_SECTIONS):
        return True
    title = item.title.lower()
    return any(kw in title for kw in BOILERPLATE_TITLES)


def group_of(item: AgendaItem) -> str:
    """Return the triage group name for a surviving (non-boilerplate) item."""
    item_type = item.item_type.lower()
    if "consent" in item_type:
        return "consent_agenda"
    if "action" in item_type:
        return "action_items"
    return "discussion"


# --- Logistics extraction --------------------------------------------------

def _parse_sessions(meeting_name: str) -> list[Session]:
    """Split a BoardDocs meeting `name` into its closed/regular sessions.

    The name reads like "Closed Session beginning at 6:00 p.m. in Room 200 ...
    and the Regular Meeting at 6:30 p.m. in the ... Auditorium ...".
    """
    sessions: list[Session] = []
    for segment in re.split(r"\s+and the\s+", meeting_name):
        segment = segment.strip()
        if not segment:
            continue
        time_m = _TIME_RE.search(segment)
        time = time_m.group(0).strip() if time_m else None

        if time_m:
            head = segment[: time_m.start()]
            tail = segment[time_m.end():]
        else:
            head, tail = segment, ""
        name = re.sub(r"\s*(?:beginning\s+)?at\s*$", "", head, flags=re.IGNORECASE).strip()
        location = re.sub(r"^\s*(?:in|at)\s+", "", tail.strip(), flags=re.IGNORECASE).strip(" .")
        sessions.append(Session(name=name or segment, time=time, location=location or None))
    return sessions


def extract_logistics(items: list[AgendaItem], meta: dict, preamble: str) -> Logistics:
    """Pull never-hallucinate-able meeting facts out of the metadata + agenda."""
    numberdate = meta.get("numberdate", "")
    try:
        iso_date = dt.datetime.strptime(numberdate, "%Y%m%d").date().isoformat()
    except ValueError:
        iso_date = None

    meeting_name = meta.get("name", "")

    # Livestream URL: take it from the preamble's "watch ... live" notice.
    livestream = None
    for line in preamble.splitlines():
        if "watch" in line.lower() or "live" in line.lower():
            if (m := _URL_RE.search(line)):
                livestream = m.group(0).rstrip(".")
                break

    # Public-comment rule: the title of the public-comment agenda item.
    public_comment_rule = None
    for item in items:
        if "public comment" in item.title.lower() or "citizens' comment" in item.section.lower():
            public_comment_rule = item.title
            break

    return Logistics(
        date=iso_date,
        meeting_name=meeting_name,
        sessions=_parse_sessions(meeting_name),
        livestream=livestream,
        public_comment_rule=public_comment_rule,
        next_meeting=None,  # needs the live meetings list — wired up later
    )


# --- Triage ----------------------------------------------------------------

def triage(items: list[AgendaItem], meta: dict, preamble: str = "") -> TriageResult:
    """Filter, group, and extract logistics from a parsed agenda."""
    result = TriageResult(logistics=extract_logistics(items, meta, preamble))

    dropped: list[str] = []
    unmatched_types: list[str] = []
    groups = {
        "action_items": result.action_items,
        "consent_agenda": result.consent_agenda,
        "discussion": result.discussion,
    }

    for item in items:
        if is_boilerplate(item):
            dropped.append(f"{item.number or '-'} {item.title}".strip())
            continue
        group = group_of(item)
        groups[group].append(item)
        # A discussion-group item whose type isn't a known discussion type
        # was caught only by the fallthrough — surface it so the rules can be
        # tightened rather than letting it pass silently.
        if group == "discussion" and item.item_type.lower() not in DISCUSSION_TYPES:
            unmatched_types.append(f"{item.number or '-'} {item.title} (type: {item.item_type!r})")

    notes: list[str] = []
    if result.logistics.next_meeting is None:
        notes.append("next_meeting not yet wired up (needs the live meetings list)")
    if result.logistics.livestream is None:
        notes.append("no livestream URL found in the agenda preamble")
    if result.logistics.public_comment_rule is None:
        notes.append("no public-comment item found")

    result.report = {
        "total_items": len(items),
        "kept": sum(len(g) for g in groups.values()),
        "dropped_count": len(dropped),
        "dropped": dropped,
        "group_sizes": {name: len(g) for name, g in groups.items()},
        "unmatched_types": unmatched_types,
        "notes": notes,
    }
    return result


def kept_items(result: TriageResult) -> list[AgendaItem]:
    """Every surviving (non-boilerplate) item, in group order.

    The order matches `print_report`'s group listing — action items, consent
    agenda, then discussion — so downstream code can present a single flat list
    without re-deriving the grouping.
    """
    return [
        *result.action_items,
        *result.consent_agenda,
        *result.discussion,
    ]


# --- CLI -------------------------------------------------------------------

def print_report(result: TriageResult) -> None:
    """Human-readable triage summary for the dry-run / standalone CLI."""
    r = result.report
    log = result.logistics
    print(f"Triage: {r['total_items']} items -> {r['kept']} kept, "
          f"{r['dropped_count']} dropped")
    for name, size in r["group_sizes"].items():
        print(f"  {name:<16} {size}")
    print("Logistics:")
    print(f"  date         {log.date}")
    for s in log.sessions:
        print(f"  session      {s.name} @ {s.time or '?'} - {s.location or '?'}")
    print(f"  livestream   {log.livestream}")
    print(f"  comment rule {log.public_comment_rule}")
    print(f"  next meeting {log.next_meeting}")
    if r["dropped"]:
        print("Dropped as boilerplate:")
        for d in r["dropped"]:
            print(f"  - {d}")
    if r["unmatched_types"]:
        print("Unmatched item types (landed in discussion):")
        for u in r["unmatched_types"]:
            print(f"  ! {u}")
    if r["notes"]:
        print("Notes:")
        for n in r["notes"]:
            print(f"  * {n}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("html", type=pathlib.Path, help="agenda HTML file")
    parser.add_argument("meta", type=pathlib.Path, help="meeting-meta JSON file")
    parser.add_argument("--json", action="store_true",
                        help="dump the full TriageResult as JSON")
    args = parser.parse_args()

    for path in (args.html, args.meta):
        if not path.is_file():
            sys.exit(f"No such file: {path}")

    html = args.html.read_text()
    items = parse_agenda(html)
    meta = json.loads(args.meta.read_text())
    result = triage(items, meta, agenda_preamble(html))

    if args.json:
        print(json.dumps(to_dict(result), indent=2))
    else:
        print_report(result)


if __name__ == "__main__":
    main()
