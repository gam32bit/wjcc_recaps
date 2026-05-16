#!/usr/bin/env python3
"""Orchestrate the WJCC agenda -> newsletter pipeline.

Pipeline (Phase 1 design):

    pull_agenda.py -> parse.py -> triage.py -> write.py -> render.py
       (done)        HTML->data   filter/group   Claude     template

This step wires up the no-API front half. `--dry-run` runs parse + triage
against a saved fixture, dumps the triaged JSON to `out/`, and stops before
the (not-yet-built) Claude call — so it needs no API key and costs nothing.

Usage:
    python make_newsletter.py --dry-run                 # newest fixture
    python make_newsletter.py --date 20260519 --dry-run # a specific meeting
"""

import argparse
import json
import pathlib

from parse import agenda_preamble, parse_agenda, to_dict
from triage import print_report, triage

PROJECT_DIR = pathlib.Path(__file__).resolve().parent
FIXTURES_DIR = PROJECT_DIR / "wjcc-fixtures"
OUT_DIR = PROJECT_DIR / "out"


def discover_fixtures() -> dict[str, tuple[pathlib.Path, pathlib.Path]]:
    """Map each fixture's numberdate -> (agenda HTML, meeting-meta JSON).

    Fixture pairs are named two ways — `meeting-meta-<date>.json` next to
    `agenda-<date>.html`, and the Phase 0 `sample-meeting-meta.json` next to
    `sample-agenda.html` — so the HTML name is derived from the meta name.
    """
    fixtures: dict[str, tuple[pathlib.Path, pathlib.Path]] = {}
    for meta_path in FIXTURES_DIR.glob("*meta*.json"):
        html_path = meta_path.with_name(
            meta_path.name.replace("meeting-meta", "agenda").replace(".json", ".html")
        )
        if not html_path.is_file():
            continue
        try:
            numberdate = json.loads(meta_path.read_text()).get("numberdate", "")
        except json.JSONDecodeError:
            continue
        if numberdate:
            fixtures[numberdate] = (html_path, meta_path)
    return fixtures


def select_fixture(date: str | None) -> tuple[str, pathlib.Path, pathlib.Path]:
    """Resolve --date (or the newest fixture) to a (date, html, meta) triple."""
    fixtures = discover_fixtures()
    if not fixtures:
        raise SystemExit(f"No agenda fixtures found in {FIXTURES_DIR}.")
    if date:
        if date not in fixtures:
            available = ", ".join(sorted(fixtures)) or "none"
            raise SystemExit(f"No fixture for {date}. Available: {available}")
        chosen = date
    else:
        chosen = max(fixtures)
    html_path, meta_path = fixtures[chosen]
    return chosen, html_path, meta_path


def dry_run(date: str, html_path: pathlib.Path, meta_path: pathlib.Path) -> None:
    """Parse + triage a fixture, print the report, and dump triaged JSON."""
    html = html_path.read_text()
    items = parse_agenda(html)
    meta = json.loads(meta_path.read_text())
    result = triage(items, meta, agenda_preamble(html))

    print(f"Dry run for {date}  ({html_path.name})\n")
    print_report(result)

    OUT_DIR.mkdir(exist_ok=True)
    out_path = OUT_DIR / f"triage-{date}.json"
    out_path.write_text(json.dumps(to_dict(result), indent=2))
    print(f"\nWrote {out_path.relative_to(PROJECT_DIR)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--date", help="meeting numberdate, e.g. 20260519 "
                                       "(default: newest fixture)")
    parser.add_argument("--dry-run", action="store_true",
                        help="run parse + triage only; no API call, no cost")
    args = parser.parse_args()

    date, html_path, meta_path = select_fixture(args.date)

    if args.dry_run:
        dry_run(date, html_path, meta_path)
        return

    print("write.py not built yet — run with --dry-run for now.")


if __name__ == "__main__":
    main()
