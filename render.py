#!/usr/bin/env python3
"""Render a newsletter draft into "The Rundown" Markdown.

Deterministic, no LLM, no network. Takes the structured Draft from write.py and
the TriageResult from triage.py and lays them out as a paste-ready Markdown
post. Meeting logistics come straight from the (never-sent-to-Claude) Logistics
record, so the date, session times, locations, and livestream URL are exact.
Item titles in the "Also on the agenda" list are resolved from the deterministic
parse by agenda number, not from Claude — only judgment and prose come from the
model.

Usage:
    # re-runs parse + triage (free, deterministic) and renders a saved draft
    python render.py wjcc-fixtures/agenda-20260519.html \\
                      wjcc-fixtures/meeting-meta-20260519.json \\
                      out/draft-20260519.json
"""

import argparse
import datetime as dt
import json
import pathlib
import sys

from parse import agenda_preamble, parse_agenda
from triage import TriageResult, kept_items, triage
from write import Draft, draft_from_dict


def _format_date(iso: str | None) -> str | None:
    """Turn an ISO date into 'Tuesday, May 19, 2026' (no zero-padded day)."""
    if not iso:
        return None
    try:
        d = dt.date.fromisoformat(iso)
    except ValueError:
        return iso
    return f"{d:%A, %B} {d.day}, {d.year}"


def render_newsletter(result: TriageResult, draft: Draft) -> str:
    """Render a TriageResult + Draft into 'The Rundown' Markdown."""
    log = result.logistics
    titles = {item.number: item.title for item in kept_items(result) if item.number}

    date_str = _format_date(log.date)
    lines: list[str] = []

    title = "The Rundown: WJCC School Board"
    if date_str:
        title += f" — {date_str}"
    lines += [f"# {title}", "", draft.intro.strip(), ""]

    lines += ["## What to watch", ""]
    for i, w in enumerate(draft.watch_items, 1):
        tag = f" (Item {w.number})" if w.number else ""
        lines += [f"### {i}. {w.headline.strip()}{tag}", "", w.summary.strip(), ""]

    if draft.also_on_agenda:
        lines += ["## Also on the agenda", ""]
        for a in draft.also_on_agenda:
            note = a.note.strip()
            label = titles.get(a.number) or (f"Item {a.number}" if a.number else "")
            lines.append(f"- **{label}** — {note}" if label else f"- {note}")
        lines.append("")

    lines += ["## How to weigh in", ""]
    if date_str:
        lines += [f"The WJCC School Board meets **{date_str}**.", ""]
    for s in log.sessions:
        detail = ", ".join(bit for bit in (s.time, s.location) if bit)
        lines.append(f"- **{s.name}**" + (f" — {detail}" if detail else ""))
    if log.sessions:
        lines.append("")
    if log.public_comment_rule:
        lines += [f"**To speak:** {log.public_comment_rule}", ""]
    if log.livestream:
        lines += [f"**Watch live:** {log.livestream}", ""]
    if log.next_meeting:
        lines += [f"**Next meeting:** {log.next_meeting}", ""]

    return "\n".join(lines).rstrip() + "\n"


# --- CLI -------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("html", type=pathlib.Path, help="agenda HTML file")
    parser.add_argument("meta", type=pathlib.Path, help="meeting-meta JSON file")
    parser.add_argument("draft", type=pathlib.Path,
                        help="draft JSON file from write.py / make_newsletter.py")
    args = parser.parse_args()

    for path in (args.html, args.meta, args.draft):
        if not path.is_file():
            sys.exit(f"No such file: {path}")

    html = args.html.read_text()
    result = triage(parse_agenda(html), json.loads(args.meta.read_text()),
                    agenda_preamble(html))
    draft = draft_from_dict(json.loads(args.draft.read_text()))

    print(render_newsletter(result, draft), end="")


if __name__ == "__main__":
    main()
