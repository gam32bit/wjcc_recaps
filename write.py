#!/usr/bin/env python3
"""Draft a newsletter from a triaged agenda with one Claude call.

This is the only step in the pipeline that touches the API. It takes the
grouped, boilerplate-free items from triage.py, sends them to Claude, and gets
back a structured draft: an intro, a ranked list of "watch" items, and one-line
notes for everything else. Claude supplies judgment (which 3-5 items matter) and
words (plain-English summaries); render.py owns all layout.

The meeting logistics in the TriageResult (date, times, locations, livestream,
public-comment rule) are deterministic and are NEVER sent to Claude — render.py
adds them straight from the Logistics record, so they can't be hallucinated.

Claude is constrained to JSON matching a fixed schema via structured outputs
(`output_config.format`), so the draft is always shaped the same way and
render.py never has to cope with prose.

Caching note: this call runs once per meeting and has no large shared prefix
(the bulk of the prompt is the per-meeting agenda), so prompt caching would only
add the cache-write premium with nothing to read back — it is deliberately not
used here.

Usage:
    python write.py wjcc-fixtures/agenda-20260519.html \\
                     wjcc-fixtures/meeting-meta-20260519.json
    python write.py <agenda.html> <meeting-meta.json> --json   # raw draft JSON

Requires ANTHROPIC_API_KEY in the environment.
"""

import argparse
import dataclasses
import json
import os
import pathlib
import sys
from dataclasses import dataclass, field

import anthropic

from parse import AgendaItem, agenda_preamble, parse_agenda
from triage import TriageResult, kept_items, triage

# Phase 1 design decision: Sonnet handles this judgement-and-summarize task well
# at modest cost. Swap to another model ID here if that changes.
MODEL = "claude-sonnet-4-6"


def _load_dotenv() -> None:
    """Load `KEY=value` pairs from a gitignored `.env` file next to this script.

    Dependency-free so the pipeline needs nothing beyond `anthropic`. A real
    environment variable always wins — the file only fills in what's unset.
    """
    env_path = pathlib.Path(__file__).resolve().parent / ".env"
    if not env_path.is_file():
        return
    for raw in env_path.read_text().splitlines():
        line = raw.strip().removeprefix("export ").strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


# --- Data model ------------------------------------------------------------

@dataclass
class WatchItem:
    number: str    # agenda item number it maps to, e.g. "8.01" ("" if none)
    headline: str  # Claude's own plain-English headline (not the agenda title)
    summary: str   # 2-4 sentences: what it is and why it matters


@dataclass
class AlsoItem:
    number: str    # agenda item number ("" if none)
    note: str      # one neutral sentence


@dataclass
class Draft:
    intro: str
    watch_items: list[WatchItem] = field(default_factory=list)
    also_on_agenda: list[AlsoItem] = field(default_factory=list)


def draft_from_dict(data: dict) -> Draft:
    """Rebuild a Draft from its JSON form (the structured-output payload)."""
    return Draft(
        intro=data["intro"],
        watch_items=[WatchItem(**w) for w in data["watch_items"]],
        also_on_agenda=[AlsoItem(**a) for a in data["also_on_agenda"]],
    )


# --- Claude contract -------------------------------------------------------

# The schema Claude's response is constrained to. Every object needs
# `additionalProperties: false` and a full `required` list for structured
# outputs; count limits ("3-5 watch items") are enforced by the prompt, not the
# schema, since JSON-schema array constraints aren't supported.
DRAFT_SCHEMA: dict = {
    "type": "object",
    "additionalProperties": False,
    "required": ["intro", "watch_items", "also_on_agenda"],
    "properties": {
        "intro": {"type": "string"},
        "watch_items": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["number", "headline", "summary"],
                "properties": {
                    "number": {"type": "string"},
                    "headline": {"type": "string"},
                    "summary": {"type": "string"},
                },
            },
        },
        "also_on_agenda": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["number", "note"],
                "properties": {
                    "number": {"type": "string"},
                    "note": {"type": "string"},
                },
            },
        },
    },
}

SYSTEM_PROMPT = """\
You are the editor of "The Rundown," a free community newsletter that previews \
upcoming Williamsburg-James City County (WJCC) School Board meetings for local \
parents, residents, and taxpayers.

Your job: read the substantive agenda items for the next meeting and produce a \
preview draft. Write in a plain, neutral, concise voice. Explain why each item \
matters to families and taxpayers — what it would change, who it affects, what \
is at stake — without taking sides, editorializing, or speculating. Use only \
facts present in the agenda text you are given. Never invent figures, dates, \
names, or outcomes. Dollar amounts must match the agenda verbatim. The board \
has not yet voted on any of these items; describe them as proposed or up for a \
vote, never as decided.

Produce three things:

1. intro — one short paragraph (2-4 sentences) that orients the reader: what \
kind of meeting this is and the one or two threads worth paying attention to. \
Do not state meeting times or locations; those are added separately.

2. watch_items — the 3 to 5 agenda items that most deserve a resident's \
attention, ordered most to least significant. Prioritize items with real \
impact on students, families, school staff, or taxpayer money: budgets, policy \
changes, programs, personnel costs, contracts. For each, give a short \
plain-English headline in your own clear phrasing (not the agenda's official \
title) and a 2-4 sentence summary of what it is and why it matters. Put the \
agenda item number in the `number` field, or "" if the item has none.

3. also_on_agenda — every other substantive item, one neutral sentence each, \
with its agenda number in the `number` field. Keep these brief.

Choose no fewer than 3 and no more than 5 watch items. Every substantive item \
that is not a watch item must appear in also_on_agenda."""


# --- Prompt assembly -------------------------------------------------------

_GROUP_LABELS = {
    "action_items": "ACTION ITEM",
    "discussion": "DISCUSSION ITEM",
    "consent_agenda": "CONSENT AGENDA ITEM",
}


def _format_item(item: AgendaItem, label: str) -> str:
    """Render one agenda item as a plain-text block for the prompt."""
    head = f"{item.number} — {item.title}" if item.number else item.title
    lines = [f"[{label}] {head}", f"Type: {item.item_type}"]
    if item.recommended_action:
        lines.append(f"Recommended action: {item.recommended_action}")
    if item.dollar_figures:
        lines.append(f"Dollar figures in the text: {', '.join(item.dollar_figures)}")
    if item.attachments:
        lines.append(f"Attachments: {'; '.join(a.name for a in item.attachments)}")
    if item.body:
        lines.append("Background:")
        lines.append(item.body)
    return "\n".join(lines)


def build_prompt(result: TriageResult) -> str:
    """Assemble the user message from the triaged, grouped agenda items."""
    groups = (
        ("action_items", result.action_items),
        ("discussion", result.discussion),
        ("consent_agenda", result.consent_agenda),
    )
    blocks = [
        _format_item(item, _GROUP_LABELS[name])
        for name, items in groups
        for item in items
    ]
    return (
        "Here are the substantive agenda items for the next WJCC School Board "
        "meeting, already grouped and with procedural boilerplate removed. "
        "Write the preview draft.\n\n" + "\n\n---\n\n".join(blocks)
    )


# --- The Claude call -------------------------------------------------------

def write_draft(result: TriageResult, *, model: str = MODEL,
                client: anthropic.Anthropic | None = None) -> Draft:
    """Send the triaged agenda to Claude and return a structured Draft."""
    if not kept_items(result):
        raise SystemExit("Triage kept no substantive items — nothing to draft.")

    if client is None:
        _load_dotenv()
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise SystemExit("ANTHROPIC_API_KEY is not set (env var or .env file); "
                             "cannot call the API.")
        client = anthropic.Anthropic()

    try:
        response = client.messages.create(
            model=model,
            max_tokens=8000,
            thinking={"type": "adaptive"},
            output_config={
                "effort": "medium",
                "format": {"type": "json_schema", "schema": DRAFT_SCHEMA},
            },
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": build_prompt(result)}],
        )
    except anthropic.APIError as e:
        raise SystemExit(f"Claude API call failed: {e}")

    if response.stop_reason == "refusal":
        raise SystemExit("Claude refused to generate the draft.")
    if response.stop_reason == "max_tokens":
        raise SystemExit("Claude's response hit the token limit; raise max_tokens.")

    text = next((b.text for b in response.content if b.type == "text"), None)
    if text is None:
        raise SystemExit("Claude returned no text content.")
    return draft_from_dict(json.loads(text))


def validate_draft(draft: Draft, result: TriageResult) -> list[str]:
    """Return human-readable notes about anything off in the returned draft."""
    notes: list[str] = []
    count = len(draft.watch_items)
    if not 3 <= count <= 5:
        notes.append(f"expected 3-5 watch items, got {count}")

    known = {item.number for item in kept_items(result) if item.number}
    for w in draft.watch_items:
        if w.number and w.number not in known:
            notes.append(f"watch item references unknown agenda number {w.number!r}")
    for a in draft.also_on_agenda:
        if a.number and a.number not in known:
            notes.append(f"also-on-agenda note references unknown number {a.number!r}")

    # Every kept item must land somewhere in the draft — flag any that Claude
    # left out of both lists.
    placed = ({w.number for w in draft.watch_items}
              | {a.number for a in draft.also_on_agenda})
    for item in kept_items(result):
        if item.number and item.number not in placed:
            notes.append(f"agenda item {item.number} ({item.title}) is in "
                         "neither watch_items nor also_on_agenda")
    return notes


# --- CLI -------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("html", type=pathlib.Path, help="agenda HTML file")
    parser.add_argument("meta", type=pathlib.Path, help="meeting-meta JSON file")
    parser.add_argument("--json", action="store_true",
                        help="print the raw draft JSON instead of a summary")
    args = parser.parse_args()

    for path in (args.html, args.meta):
        if not path.is_file():
            sys.exit(f"No such file: {path}")

    html = args.html.read_text()
    result = triage(parse_agenda(html), json.loads(args.meta.read_text()),
                    agenda_preamble(html))
    draft = write_draft(result)

    if args.json:
        print(json.dumps(dataclasses.asdict(draft), indent=2))
        return

    print(f"Intro:\n  {draft.intro}\n")
    print(f"Watch items ({len(draft.watch_items)}):")
    for i, w in enumerate(draft.watch_items, 1):
        print(f"  {i}. [{w.number or '-'}] {w.headline}")
        print(f"     {w.summary}")
    print(f"\nAlso on the agenda ({len(draft.also_on_agenda)}):")
    for a in draft.also_on_agenda:
        print(f"  [{a.number or '-'}] {a.note}")
    for note in validate_draft(draft, result):
        print(f"  ! {note}")


if __name__ == "__main__":
    main()
