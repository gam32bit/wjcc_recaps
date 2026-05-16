#!/usr/bin/env python3
"""Pull a WJCC School Board agenda from BoardDocs.

Reproduces the working `requests`-based approach from the Phase 0 log:
drop the llama-index reader, hit the two BoardDocs POST endpoints directly
with browser-style headers, and save the agenda as a fixture.

Usage:
    python pull_agenda.py                 # next upcoming meeting
    python pull_agenda.py --date 20260519 # a specific meeting by numberdate
    python pull_agenda.py --list          # just list available meetings
"""

import argparse
import datetime as dt
import json
import pathlib
import random
import sys

import html2text
import requests

# --- Constants discovered during Phase 0 -----------------------------------

BASE = "https://go.boarddocs.com/vsba/wjcc/Board.nsf"
COMMITTEE_ID = "A9HEJH3AA9D7"  # WJCC School Board

PROJECT_DIR = pathlib.Path(__file__).resolve().parent
FIXTURES_DIR = PROJECT_DIR / "wjcc-fixtures"
HEADERS_FILE = FIXTURES_DIR / "working-headers.json"


# --- BoardDocs API ---------------------------------------------------------

def _cachebust(endpoint: str) -> str:
    """BoardDocs endpoints expect `?open&<random>` — jQuery appends Math.random()."""
    return f"{BASE}/{endpoint}?open&{random.random()}"


def new_session() -> requests.Session:
    """A session with the reverse-engineered browser headers, warmed up.

    Stale UAs get blocked by the CloudFront WAF (403), so the headers from
    `working-headers.json` matter. The initial GET of /Public mirrors a real
    browser visit before the XHR calls.
    """
    session = requests.Session()
    session.headers.update(json.loads(HEADERS_FILE.read_text()))
    session.get(f"{BASE}/Public").raise_for_status()
    return session


def get_meetings(session: requests.Session) -> list[dict]:
    """Return the list of meetings for the WJCC School Board committee.

    The site's jQuery `ajaxPrefilter` attaches `current_committee_id` to every
    POST; without it BD-GetMeetingsList returns an empty 200.
    """
    resp = session.post(
        _cachebust("BD-GetMeetingsList"),
        data={"current_committee_id": COMMITTEE_ID},
    )
    resp.raise_for_status()
    meetings = resp.json()
    # Phase 0: the list contains one empty `{}` record — filter on `unique`.
    return [m for m in meetings if m.get("unique")]


def get_agenda_html(session: requests.Session, meeting_id: str) -> str:
    """Return the detailed agenda HTML for a meeting `unique` id."""
    resp = session.post(
        _cachebust("PRINT-AgendaDetailed"),
        data={"id": meeting_id, "current_committee_id": COMMITTEE_ID},
    )
    resp.raise_for_status()
    return resp.text


# --- Meeting selection -----------------------------------------------------

def meeting_date(meeting: dict) -> dt.date | None:
    """Parse a meeting's `numberdate` (YYYYMMDD) into a date."""
    raw = meeting.get("numberdate", "")
    try:
        return dt.datetime.strptime(raw, "%Y%m%d").date()
    except ValueError:
        return None


def pick_meeting(meetings: list[dict], date: str | None) -> dict:
    """Pick a meeting by numberdate, or the next upcoming one.

    Phase 0: the `current=1` flag does NOT mean "upcoming" — use a date
    comparison against today instead.
    """
    if date:
        for m in meetings:
            if m.get("numberdate") == date:
                return m
        raise SystemExit(f"No meeting found with numberdate {date}.")

    today = dt.date.today()
    upcoming = sorted(
        (m for m in meetings if (d := meeting_date(m)) and d >= today),
        key=meeting_date,
    )
    if not upcoming:
        raise SystemExit("No upcoming meetings found.")
    return upcoming[0]


# --- Output ----------------------------------------------------------------

def save_fixtures(meeting: dict, html: str) -> None:
    """Write the agenda HTML, Markdown, and meeting meta as date-stamped fixtures."""
    stamp = meeting.get("numberdate", "unknown")
    FIXTURES_DIR.mkdir(exist_ok=True)

    converter = html2text.HTML2Text()
    converter.body_width = 0  # don't hard-wrap; keep lines intact
    markdown = converter.handle(html)

    targets = {
        FIXTURES_DIR / f"agenda-{stamp}.html": html,
        FIXTURES_DIR / f"agenda-{stamp}.md": markdown,
        FIXTURES_DIR / f"meeting-meta-{stamp}.json": json.dumps(meeting, indent=2),
    }
    for path, content in targets.items():
        path.write_text(content)
        print(f"  wrote {path.relative_to(PROJECT_DIR)} ({len(content):,} chars)")


def describe(meeting: dict) -> str:
    d = meeting_date(meeting)
    label = d.isoformat() if d else meeting.get("numberdate", "?")
    return f"{label}  {meeting.get('unique', '?')}  {meeting.get('name', '')[:70]}"


# --- Entry point -----------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", help="meeting numberdate, e.g. 20260519")
    parser.add_argument("--list", action="store_true", help="list meetings and exit")
    args = parser.parse_args()

    session = new_session()

    print("Fetching meetings list...")
    meetings = get_meetings(session)
    print(f"  {len(meetings)} meetings found.")

    if args.list:
        for m in sorted(meetings, key=lambda m: m.get("numberdate", "")):
            print(f"  {describe(m)}")
        return

    meeting = pick_meeting(meetings, args.date)
    print(f"Selected meeting:\n  {describe(meeting)}")

    print("Fetching agenda HTML...")
    html = get_agenda_html(session, meeting["unique"])
    if len(html) < 500:
        print("  WARNING: agenda HTML looks too short — agenda may not be posted yet.")
        print(f"  response: {html!r}")
        sys.exit(1)
    print(f"  {len(html):,} chars of HTML.")

    save_fixtures(meeting, html)
    print("Done.")


if __name__ == "__main__":
    main()
