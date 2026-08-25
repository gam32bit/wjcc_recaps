#!/usr/bin/env python3
"""Rubric calibration tool for tuning rubric.md against historical data.

Fetches agendas for past meetings, scores them (optionally using work-session
transcripts from past-meeting-urls), parses local news archives, and produces
a side-by-side calibration report showing rubric scores vs. what news outlets
covered. Run once (or occasionally) to identify where rubric.md diverges from
community impact. The core pipeline (make_newsletter.py, score.py) is untouched.

Input files (not committed; add to .gitignore):
  news-archives/          one .md file per outlet, freeform headlines + dates
  past-meeting-urls       YYYYMMDD <youtube-url> for each regular meeting
                          (the URL should be the WORK SESSION for that meeting)

Usage:
    python calibrate.py --list              # list available meetings in range
    python calibrate.py --dry-run           # deterministic signals only
    python calibrate.py                     # full run, fall 2025 to today
    python calibrate.py --from 20260101     # custom start date
    python calibrate.py --no-cache          # re-score even if cached
    python calibrate.py --suggest-edits     # also propose rubric.md edits
"""

import argparse
import dataclasses
import datetime as dt
import json
import os
import pathlib
import re
import sys

import anthropic

import llmcache

from parse import agenda_preamble, parse_agenda
from pull_agenda import (
    describe,
    get_agenda_html,
    get_meetings,
    meeting_date,
    new_session,
    save_fixtures,
)
from score import (
    MODEL,
    ScoredItem,
    _print_table,
    compute_deterministic,
    finalize,
    score_rubric,
    segment_transcript,
)
from transcript import fetch_transcript
from triage import kept_items, triage

PROJECT_DIR = pathlib.Path(__file__).resolve().parent
FIXTURES_DIR = PROJECT_DIR / "wjcc-fixtures"
NEWS_DIR = PROJECT_DIR / "news-archives"
CACHE_DIR = PROJECT_DIR / ".cache"
OUT_DIR = PROJECT_DIR / "out"

DEFAULT_FROM = "20250901"
# Articles within [meeting_date - BEFORE, meeting_date + AFTER] are shown.
NEWS_BEFORE_DAYS = 14
NEWS_AFTER_DAYS = 7


# --- Env loading --------------------------------------------------------------

def _load_dotenv() -> None:
    env_path = PROJECT_DIR / ".env"
    if not env_path.is_file():
        return
    for raw in env_path.read_text().splitlines():
        line = raw.strip().removeprefix("export ").strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


# --- News parsing -------------------------------------------------------------

_MONTH_NUM: dict[str, int] = {
    "January": 1, "February": 2, "March": 3, "April": 4,
    "May": 5, "June": 6, "July": 7, "August": 8,
    "September": 9, "October": 10, "November": 11, "December": 12,
}

_DATE_RE = re.compile(
    r"\b(January|February|March|April|May|June|July|August"
    r"|September|October|November|December)"
    r"\s+(\d{1,2}),\s+(\d{4})"
)
_BY_RE = re.compile(r"^By\s+[A-Z]")
_CATEGORY_RE = re.compile(
    r"^(Education|Latest News|Business|Alerts|Brews and Bites|Commentary"
    r"|Latest\s+News)\s*$",
    re.IGNORECASE,
)


def _is_caption(line: str) -> bool:
    return bool(re.search(
        r"/The\s+Virginia\s+Gazette|/For\s+The|Courtesy|Maggie Root"
        r"|James W\. Robinson/|Courtesy of",
        line,
    ))


def _parse_news_file(path: pathlib.Path) -> list[dict]:
    """Extract {date, headline, outlet} entries from a freeform news archive file."""
    outlet = path.stem
    lines = path.read_text().splitlines()
    articles: list[dict] = []

    for i, line in enumerate(lines):
        m = _DATE_RE.search(line)
        if not m:
            continue
        try:
            date = dt.date(int(m.group(3)), _MONTH_NUM[m.group(1)], int(m.group(2)))
        except (KeyError, ValueError):
            continue

        # Walk back up to 3 lines to find the headline.
        headline = None
        for j in range(i - 1, max(i - 4, -1), -1):
            candidate = lines[j].strip()
            if (candidate
                    and not _BY_RE.match(candidate)
                    and not _CATEGORY_RE.match(candidate)
                    and not _is_caption(candidate)
                    and len(candidate) > 10):
                headline = candidate
                break

        if headline:
            articles.append({"date": date, "headline": headline, "outlet": outlet})

    return articles


def load_news() -> list[dict]:
    """Load all articles from news-archives/ sorted by date."""
    if not NEWS_DIR.is_dir():
        return []
    articles: list[dict] = []
    for p in sorted(NEWS_DIR.glob("*.md")):
        parsed = _parse_news_file(p)
        articles.extend(parsed)
    articles.sort(key=lambda a: a["date"])
    return articles


def _articles_for_meeting(
    articles: list[dict], date: dt.date
) -> list[dict]:
    lo = date - dt.timedelta(days=NEWS_BEFORE_DAYS)
    hi = date + dt.timedelta(days=NEWS_AFTER_DAYS)
    return [a for a in articles if lo <= a["date"] <= hi]


# --- Transcript URL loading ---------------------------------------------------

# Matches M/D/YY, M/D/YYYY, MM/DD/YY, MM/DD/YYYY at the start of a line.
_FLEX_DATE_RE = re.compile(r"^(\d{1,2})/(\d{1,2})/(\d{2,4})")
# Also matches YYYYMMDD.
_STAMP_DATE_RE = re.compile(r"^(\d{8})\s")


def _parse_flex_date(s: str) -> dt.date | None:
    """Parse M/D/YY, M/D/YYYY, or YYYYMMDD into a date. Returns None on failure."""
    m = _FLEX_DATE_RE.match(s)
    if m:
        month, day, year = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if year < 100:
            year += 2000
        try:
            return dt.date(year, month, day)
        except ValueError:
            return None
    m = _STAMP_DATE_RE.match(s)
    if m:
        try:
            return dt.datetime.strptime(m.group(1), "%Y%m%d").date()
        except ValueError:
            return None
    return None


def load_urls(path: pathlib.Path) -> dict[dt.date, str]:
    """Parse a markdown URLs file, returning work-session {date: url} entries.

    Understands a sectioned markdown format (## work sessions, ## joint
    meetings, ## Regular meetings). Only entries under '## work sessions'
    sections are returned — those are the transcripts used for scoring.
    Dates may be M/D/YY, M/D/YYYY, or YYYYMMDD.
    """
    if not path.is_file():
        return {}
    result: dict[dt.date, str] = {}
    in_work_sessions = False
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if line.startswith("##"):
            in_work_sessions = re.search(r"work\s+session", line, re.IGNORECASE) is not None
            continue
        if not in_work_sessions or not line or line.startswith("#"):
            continue
        parts = line.split(None, 1)
        if len(parts) != 2:
            continue
        date = _parse_flex_date(parts[0])
        if date:
            result[date] = parts[1]
    return result


def _find_work_session_url(
    work_sessions: dict[dt.date, str],
    meeting_dt: dt.date,
    max_days: int = 35,
) -> tuple[dt.date | None, str | None]:
    """Return (work_session_date, url) for the latest work session ≤ max_days
    before the given regular meeting date. Returns (None, None) if not found."""
    best_date: dt.date | None = None
    best_url: str | None = None
    for ws_date, url in work_sessions.items():
        days_before = (meeting_dt - ws_date).days
        if 0 < days_before <= max_days:
            if best_date is None or ws_date > best_date:
                best_date = ws_date
                best_url = url
    return best_date, best_url


# --- Fixture helpers ----------------------------------------------------------

def _load_or_fetch_fixture(
    meeting: dict, session
) -> tuple[str, str] | None:
    """Return (html, meta_json) for a meeting, fetching if not already cached."""
    stamp = meeting.get("numberdate", "")
    html_path = FIXTURES_DIR / f"agenda-{stamp}.html"
    meta_path = FIXTURES_DIR / f"meeting-meta-{stamp}.json"

    if html_path.is_file() and meta_path.is_file():
        return html_path.read_text(), meta_path.read_text()

    if session is None:
        print(f"  {stamp}: no cached fixture and no session — skipping.")
        return None

    print(f"  Fetching {stamp} from BoardDocs...", end=" ", flush=True)
    try:
        html = get_agenda_html(session, meeting["unique"])
    except Exception as exc:
        print(f"FAILED ({exc})")
        return None

    if len(html) < 500:
        print(f"WARNING: response too short ({len(html)} chars), agenda may not be posted.")
        return None

    save_fixtures(meeting, html)
    print(f"ok ({len(html):,} chars)")
    return html, json.dumps(meeting, indent=2)


# --- Score caching ------------------------------------------------------------

def _score_cache_path(stamp: str) -> pathlib.Path:
    return CACHE_DIR / f"calibration-scored-{stamp}.json"


def _save_score_cache(stamp: str, scored: list[ScoredItem]) -> None:
    CACHE_DIR.mkdir(exist_ok=True)
    _score_cache_path(stamp).write_text(
        json.dumps([dataclasses.asdict(s) for s in scored], indent=2)
    )


def _load_score_cache(stamp: str) -> list[dict] | None:
    p = _score_cache_path(stamp)
    if p.is_file():
        return json.loads(p.read_text())
    return None


# --- Calibration report -------------------------------------------------------

def _format_report(
    results: list[tuple[str, list[dict]]],
    articles: list[dict],
) -> str:
    sections: list[str] = [
        "# WJCC Newsletter — Rubric Calibration Report",
        f"Generated: {dt.date.today().isoformat()}",
        "",
        "For each meeting: ranked items with signals, then news articles from",
        f"the surrounding window (±{NEWS_BEFORE_DAYS}/{NEWS_AFTER_DAYS} days).",
        "Use discrepancies to tune rubric.md.",
        "",
    ]

    for stamp, scored_dicts in results:
        date_str = f"{stamp[:4]}-{stamp[4:6]}-{stamp[6:]}"
        sections.append("---")
        sections.append("")
        sections.append(f"## {date_str}")
        sections.append("")

        # Scored items table
        sections.append("### Ranked items")
        sections.append("")
        sections.append(
            f"{'Rk':>3}  {'#':>6}  {'Score':>6}  {'Disc':>5}  {'Dollar':<12}"
            f"  {'V':>1}  {'Rub':>3}  Title"
        )
        sections.append("-" * 72)
        for d in scored_dicts:
            item = d["item"]
            sig = d["signals"]
            rub = f"{sig['rubric_score']:.0f}" if sig["rubric_score"] >= 0 else " -"
            v = "Y" if sig["is_vote"] else " "
            sections.append(
                f"{d['rank']:>3}  {item['number'] or '-':>6}  {d['composite']:>6.3f}"
                f"  {sig['discussion_minutes']:>5.1f}"
                f"  {sig['dollar_raw'] or '':<12}"
                f"  {v}  {rub:>3}"
                f"  {item['title'][:45]}"
            )
            if sig.get("rubric_justification"):
                sections.append(
                    f"             Rubric: {sig['rubric_justification'][:80]}"
                )
        sections.append("")

        # News articles
        try:
            date_obj = dt.date(int(stamp[:4]), int(stamp[4:6]), int(stamp[6:]))
        except ValueError:
            continue
        nearby = _articles_for_meeting(articles, date_obj)

        if nearby:
            sections.append("### News coverage")
            sections.append("")
            by_outlet: dict[str, list[dict]] = {}
            for a in nearby:
                by_outlet.setdefault(a["outlet"], []).append(a)
            for outlet, arts in sorted(by_outlet.items()):
                sections.append(f"**{outlet}:**")
                for a in sorted(arts, key=lambda x: x["date"]):
                    sections.append(f"- {a['date'].strftime('%b %d')} — \"{a['headline']}\"")
                sections.append("")
        else:
            sections.append(
                "*No news articles found in news-archives/ for this date range.*"
            )
            sections.append("")

    return "\n".join(sections)


# --- Rubric suggestion --------------------------------------------------------

_SUGGEST_SYSTEM = """\
You are given a rubric for scoring school board agenda items 0–5, along with a
calibration report showing rubric scores alongside what local news outlets
covered around each meeting.

Your task: identify discrepancies (items that scored high but got no news
coverage; items that scored low but were covered) and propose specific, targeted
edits to the rubric criteria that would fix them.

Rules:
- Propose edits as concrete text changes: quote the current criterion and show
  the proposed replacement.
- Justify each edit with the specific discrepancy from the calibration data.
- Do not invent criteria that aren't grounded in the evidence.
- Keep the human-owned structure and scoring bands intact.
- Patterns that repeat across multiple meetings deserve more weight.
"""


def suggest_rubric_edits(
    report: str, rubric_text: str, client: anthropic.Anthropic
) -> str:
    return llmcache.text_call(
        client,
        action="rubric edit suggestions",
        # Shares score.py's MODEL rather than pinning its own string —
        # this call was the one the last model bump nearly missed.
        model=MODEL,
        # Thinking is left at Sonnet 5's default (adaptive): unlike the
        # extraction calls this one is open-ended analysis of where the
        # rubric diverges from reality, which is what thinking is for.
        # The budget is raised so thinking can't crowd out the suggestions.
        max_tokens=16000,
        system=_SUGGEST_SYSTEM,
        messages=[{
            "role": "user",
            "content": (
                f"CURRENT RUBRIC:\n{rubric_text}\n\n"
                f"CALIBRATION REPORT:\n{report}\n\n"
                "Based on the discrepancies above, propose specific edits to rubric.md."
            ),
        }],
    )


# --- Main ---------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--from", dest="from_date", default=DEFAULT_FROM, metavar="YYYYMMDD",
        help=f"start of date range (default: {DEFAULT_FROM})",
    )
    parser.add_argument(
        "--to", dest="to_date", default=None, metavar="YYYYMMDD",
        help="end of date range (default: today)",
    )
    parser.add_argument(
        "--list", action="store_true",
        help="list meetings in range and exit (no scoring)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="deterministic signals only, no API calls, no cache writes",
    )
    parser.add_argument(
        "--urls", metavar="FILE", default="past-meeting-urls.md",
        help="markdown file with work-session URLs (default: past-meeting-urls.md)",
    )
    parser.add_argument(
        "--no-cache", action="store_true",
        help="ignore cached scored results and re-score every meeting",
    )
    parser.add_argument(
        "--suggest-edits", action="store_true",
        help="call Claude to propose specific rubric.md edits (writes out/rubric-suggestions.md)",
    )
    parser.add_argument(
        "--no-llm-cache", action="store_true",
        help="Re-run every Claude call instead of reusing .cache/llm/ "
             "answers (the cache is still refreshed).",
    )
    args = parser.parse_args()
    llmcache.set_enabled(not args.no_llm_cache)

    try:
        from_date = dt.datetime.strptime(args.from_date, "%Y%m%d").date()
        to_date = (
            dt.datetime.strptime(args.to_date, "%Y%m%d").date()
            if args.to_date
            else dt.date.today()
        )
    except ValueError as exc:
        sys.exit(f"Bad date format: {exc}. Use YYYYMMDD.")

    # --- Discover meetings in range ---
    print("Connecting to BoardDocs...", flush=True)
    try:
        session = new_session()
        meetings = get_meetings(session)
    except Exception as exc:
        sys.exit(f"Failed to connect to BoardDocs: {exc}")

    all_in_range = [
        m for m in meetings if (d := meeting_date(m)) and from_date <= d <= to_date
    ]
    in_range = sorted(
        [m for m in all_in_range if "regular" in m.get("name", "").lower()],
        key=meeting_date,
    )
    skipped = len(all_in_range) - len(in_range)

    if not in_range:
        sys.exit(f"No regular meetings found between {from_date} and {to_date}.")

    print(f"\nRegular meetings in range ({from_date} to {to_date})"
          + (f"  [{skipped} non-regular skipped]" if skipped else "") + ":")
    for m in in_range:
        cached = _score_cache_path(m.get("numberdate", "")).is_file()
        tag = " [cached]" if cached else ""
        print(f"  {describe(m)}{tag}")

    if args.list:
        return

    # --- API setup ---
    if not args.dry_run:
        _load_dotenv()
        if not os.environ.get("ANTHROPIC_API_KEY"):
            sys.exit("ANTHROPIC_API_KEY is not set; use --dry-run or set the key.")
        client: anthropic.Anthropic | None = anthropic.Anthropic()
    else:
        client = None

    # --- Load transcript URLs ---
    urls_path = PROJECT_DIR / args.urls
    work_session_urls = load_urls(urls_path)
    if work_session_urls:
        print(f"\nLoaded {len(work_session_urls)} work-session URL(s) from {urls_path.name}.")
    elif urls_path.is_file():
        print(f"\n  Note: {urls_path.name} exists but no work-session entries found.")
    else:
        print(f"\n  Note: {urls_path.name} not found — discussion_minutes will be 0.")

    # --- Load news archives ---
    articles = load_news()
    if articles:
        print(f"Loaded {len(articles)} news articles from {NEWS_DIR.name}/.")
    else:
        print(f"  Note: no news articles parsed from {NEWS_DIR}/.")

    # --- Score each meeting ---
    OUT_DIR.mkdir(exist_ok=True)
    results: list[tuple[str, list[dict]]] = []

    for meeting in in_range:
        stamp = meeting["numberdate"]
        print(f"\n{'─'*55}")
        print(f"  {stamp}", flush=True)

        # Use cached scored results if available and not forcing re-score.
        if not args.dry_run and not args.no_cache:
            cached = _load_score_cache(stamp)
            if cached is not None:
                print(f"  Using cached scored results ({len(cached)} items).")
                results.append((stamp, cached))
                continue

        # Load or fetch the fixture.
        fixture = _load_or_fetch_fixture(meeting, session)
        if fixture is None:
            print(f"  Skipping {stamp}.")
            continue
        html, meta_json = fixture
        meta = json.loads(meta_json)

        items = parse_agenda(html)
        triage_result = triage(items, meta, agenda_preamble(html))
        all_items = kept_items(triage_result)
        if not all_items:
            print(f"  No substantive items after triage.")
            continue

        scored = compute_deterministic(all_items, triage_result)

        if not args.dry_run:
            try:
                meeting_d = dt.datetime.strptime(stamp, "%Y%m%d").date()
            except ValueError:
                meeting_d = None
            ws_date, yt_url = (
                _find_work_session_url(work_session_urls, meeting_d)
                if meeting_d else (None, None)
            )
            if yt_url:
                print(f"  Work session: {ws_date} → {yt_url}")
                snippets = fetch_transcript(yt_url, cache_dir=CACHE_DIR)
                segment_transcript(scored, snippets, client)
            else:
                print(f"  No work-session URL found for {stamp} (within 35 days).")

            rubric_path = PROJECT_DIR / "rubric.md"
            if rubric_path.is_file():
                score_rubric(scored, rubric_path.read_text(), client)
            else:
                print("  WARNING: rubric.md not found — rubric_score will be 0.")

        finalize(scored)
        _print_table(scored, dry_run=args.dry_run)

        scored_dicts = [dataclasses.asdict(s) for s in scored]

        if not args.dry_run:
            _save_score_cache(stamp, scored)
            print(f"  Cached to {_score_cache_path(stamp).name}.")

        results.append((stamp, scored_dicts))

    if not results:
        sys.exit("No meetings were scored.")

    # --- Build and write calibration report ---
    print(f"\n{'='*55}")
    print("Building calibration report...")
    report = _format_report(results, articles)

    report_path = OUT_DIR / "calibration-report.md"
    report_path.write_text(report)
    print(f"Wrote {report_path.relative_to(PROJECT_DIR)}")

    # --- Optional: suggest rubric edits ---
    if args.suggest_edits:
        if args.dry_run:
            print("  --suggest-edits skipped in --dry-run mode.")
        else:
            rubric_path = PROJECT_DIR / "rubric.md"
            if not rubric_path.is_file():
                print("  Cannot suggest edits: rubric.md not found.")
            else:
                suggestions = suggest_rubric_edits(report, rubric_path.read_text(), client)
                sug_path = OUT_DIR / "rubric-suggestions.md"
                sug_path.write_text(suggestions)
                print(f"Wrote {sug_path.relative_to(PROJECT_DIR)}")


if __name__ == "__main__":
    main()
