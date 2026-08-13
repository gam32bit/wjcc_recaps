#!/usr/bin/env python3
"""Draft newsletter prose for the top-ranked agenda items.

This module is prose-only — it does NOT select or rank items. Selection is
score.py's job; write.py receives the already-ranked top items and writes
plain-English copy grounded in verbatim quotes from the source documents.

Each item gets a headline, a 'what it is' paragraph, and one or more verbatim
quotes attributed to named sources (staff report, recommended action, attached
PDF title, etc.). Logistics (dates, times, locations) are never sent to Claude;
render.py adds them straight from the deterministic Logistics record.

Usage:
    python write.py wjcc-fixtures/agenda-20260519.html \\
                    wjcc-fixtures/meeting-meta-20260519.json

Requires ANTHROPIC_API_KEY in the environment or a .env file.
"""

import argparse
import base64
import dataclasses
import json
import os
import pathlib
import sys
from dataclasses import dataclass, field

import anthropic

from parse import agenda_preamble, parse_agenda
from score import ScoredItem, compute_deterministic, finalize
from triage import kept_items, triage

MODEL = "claude-sonnet-4-6"
FIXTURES_DIR = pathlib.Path(__file__).resolve().parent / "wjcc-fixtures"


def _load_dotenv() -> None:
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
class Quote:
    source: str   # named source: "staff report", "recommended action", "FY2027 budget document"
    quote: str    # verbatim excerpt from the source


@dataclass
class DraftItem:
    number: str         # agenda item number, e.g. "8.01"
    headline: str       # short plain-English headline (not the agenda's official title)
    what_it_is: str     # 2-4 sentences: what it is and what it would change
    quotes: list[Quote] = field(default_factory=list)


@dataclass
class Draft:
    intro: str
    items: list[DraftItem] = field(default_factory=list)


@dataclass
class PublicCommentSummary:
    summary: str            # 2-4 sentences on the main topics residents raised
    quote: str = ""         # one verbatim line from a speaker (may be empty)
    speaker: str = ""       # speaker's name if clearly stated (may be empty)


def public_comment_from_dict(data: dict) -> PublicCommentSummary:
    return PublicCommentSummary(
        summary=data.get("summary", ""),
        quote=data.get("quote", ""),
        speaker=data.get("speaker", ""),
    )


def draft_from_dict(data: dict) -> Draft:
    return Draft(
        intro=data["intro"],
        items=[
            DraftItem(
                number=it["number"],
                headline=it["headline"],
                what_it_is=it["what_it_is"],
                quotes=[Quote(**q) for q in it.get("quotes", [])],
            )
            for it in data.get("items", [])
        ],
    )


# --- Claude contract -------------------------------------------------------

_DRAFT_SCHEMA: dict = {
    "type": "object",
    "additionalProperties": False,
    "required": ["intro", "items"],
    "properties": {
        "intro": {"type": "string"},
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["number", "headline", "what_it_is", "quotes"],
                "properties": {
                    "number": {"type": "string"},
                    "headline": {"type": "string"},
                    "what_it_is": {"type": "string"},
                    "quotes": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": ["source", "quote"],
                            "properties": {
                                "source": {"type": "string"},
                                "quote": {"type": "string"},
                            },
                        },
                    },
                },
            },
        },
    },
}

_SYSTEM_PROMPT = """\
You are the editor of "The Rundown," a free community newsletter that previews \
upcoming Williamsburg-James City County (WJCC) School Board meetings for local \
parents, residents, and taxpayers.

You will be given a ranked list of the top agenda items for the next meeting — \
the ranking is already done. Your only job is to write clear, accurate prose.

For each item, write:
- number: the agenda item number exactly as shown in the item header, e.g. \
  "8.01" or "7.04". Copy it verbatim; do not renumber.
- headline: a short plain-English headline in your own phrasing (not the \
  official agenda title). Under 10 words.
- what_it_is: 2-4 sentences. What is this item, what would it change, and \
  who does it affect? Focus on practical impact on families and taxpayers.
- quotes: 1-3 verbatim quotes from the source documents you are given, each \
  with a named source (e.g. "staff report", "recommended action", the document \
  title). Quotes must be exact — copy and paste, do not paraphrase. If no \
  document is provided for an item, omit quotes rather than fabricate them.

Also write:
- intro: one paragraph (2-4 sentences) orienting the reader to what kind of \
  meeting this is and the one or two threads worth paying attention to. \
  Do not state meeting times, locations, or dates.

Rules:
- Use only facts present in the text you are given. Never invent figures, \
  names, or outcomes.
- Dollar amounts must match the source verbatim.
- The board has not yet voted; describe proposals as proposed or up for a vote.
- Write in a plain, neutral, concise voice. Do not editorialize or speculate.
- Never mention ranking signals (discussion minutes, dollar scores, etc.).\
"""

_RECAP_SYSTEM_PROMPT = """\
You are the editor of "The Rundown," a free community newsletter for local \
parents, residents, and taxpayers. This edition is a RECAP of a Williamsburg-\
James City County (WJCC) School Board meeting that has already happened.

You will be given a ranked list of the meeting's top agenda items — the ranking \
is already done. Your only job is to write clear, accurate prose, in PAST TENSE.

For each item, write:
- number: the agenda item number exactly as shown in the item header, e.g. \
  "8.01" or "7.04". Copy it verbatim; do not renumber.
- headline: a short plain-English headline in your own phrasing (not the \
  official agenda title). Under 10 words.
- what_it_is: 2-4 sentences. What was this item, what did the decision change, \
  and who does it affect? Focus on practical impact on families and taxpayers.
- quotes: 1-3 verbatim quotes from the source documents you are given, each \
  with a named source (e.g. "staff report", "recommended action", "board \
  motion", the document title). Quotes must be exact — copy and paste, do not \
  paraphrase. If no document is provided for an item, omit quotes.

Also write:
- intro: one paragraph (2-4 sentences) orienting the reader to what kind of \
  meeting this was and the one or two threads worth knowing about. Do not state \
  meeting times, locations, or dates.

Rules:
- Use only facts present in the text you are given. Never invent figures, \
  names, or outcomes.
- Dollar amounts must match the source verbatim.
- Describe what the board DID (approved, declined, deferred) — but DO NOT state \
  the vote tally or name how individual members voted. The newsletter adds the \
  official tally separately; if you guess it you may contradict the record.
- Write in a plain, neutral, concise voice. Do not editorialize or speculate.
- Never mention ranking signals (discussion minutes, public comment, etc.).\
"""


def _format_item_block(s: ScoredItem, mode: str = "preview") -> str:
    item = s.item
    head = f"Item {item.number} — {item.title}" if item.number else item.title
    lines = [f"[{head}]", f"Type: {item.item_type}"]
    if item.recommended_action:
        lines.append(f"Recommended action: {item.recommended_action}")
    if item.dollar_figures:
        lines.append(f"Dollar figures: {', '.join(item.dollar_figures)}")
    if item.attachments:
        lines.append(f"Attachments: {'; '.join(a.name for a in item.attachments)}")
    # In recap mode the verbatim board motion is a citable quote source. The
    # vote tally / member names are NOT included — render.py injects the
    # official outcome deterministically.
    if mode == "recap" and item.vote and item.vote.motion_text:
        lines.append(f"Board motion (verbatim): {item.vote.motion_text}")
    if item.body:
        lines.append("Background:")
        lines.append(item.body[:3000])  # cap at 3k chars per item body
    return "\n".join(lines)


def draft(
    ranked_top: list[ScoredItem],
    attachment_paths: dict[str, pathlib.Path] | None = None,
    feedback: str | None = None,
    *,
    mode: str = "preview",
    model: str = MODEL,
    client: anthropic.Anthropic | None = None,
) -> Draft:
    """Call Claude to write newsletter prose for the already-ranked top items.

    mode "preview" (upcoming meeting, future tense) or "recap" (past meeting,
    past tense, board motion offered as a quote source).
    attachment_paths maps attachment URL -> cached PDF path (optional).
    feedback is optional human notes from the checkpoint review.
    """
    if not ranked_top:
        raise SystemExit("No ranked items to draft — nothing to write.")

    if client is None:
        _load_dotenv()
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise SystemExit(
                "ANTHROPIC_API_KEY is not set (env var or .env file)."
            )
        client = anthropic.Anthropic()

    content: list[dict] = []

    # Attach PDFs as document blocks (before the text prompt)
    if attachment_paths:
        for s in ranked_top:
            for att in s.item.attachments:
                path = (attachment_paths or {}).get(att.url)
                if path and path.is_file():
                    pdf_bytes = path.read_bytes()
                    content.append({
                        "type": "document",
                        "source": {
                            "type": "base64",
                            "media_type": "application/pdf",
                            "data": base64.standard_b64encode(pdf_bytes).decode(),
                        },
                        "title": att.name,
                    })

    items_text = "\n\n---\n\n".join(_format_item_block(s, mode) for s in ranked_top)
    intro_line = (
        "Here are the top agenda items from the WJCC School Board meeting that "
        "just took place, ranked by importance. Write the recap draft.\n\n"
        if mode == "recap"
        else "Here are the top agenda items for the next WJCC School Board meeting, "
        "ranked by importance. Write the newsletter draft.\n\n"
    )
    prompt_parts = [intro_line + items_text]
    if feedback:
        prompt_parts.append(f"\nReviewer notes:\n{feedback}")

    content.append({"type": "text", "text": "\n\n".join(prompt_parts)})

    system_prompt = _RECAP_SYSTEM_PROMPT if mode == "recap" else _SYSTEM_PROMPT
    # Stream the draft: adaptive thinking and the JSON output share the
    # max_tokens budget, so we need generous headroom. The SDK refuses
    # large non-streaming requests (HTTP-timeout risk), hence .stream().
    try:
        with client.messages.stream(
            model=model,
            max_tokens=32000,
            thinking={"type": "adaptive"},
            output_config={
                "effort": "medium",
                "format": {"type": "json_schema", "schema": _DRAFT_SCHEMA},
            },
            system=system_prompt,
            messages=[{"role": "user", "content": content}],
        ) as stream:
            response = stream.get_final_message()
    except anthropic.APIError as exc:
        raise SystemExit(f"Claude API call failed: {exc}")

    if response.stop_reason == "refusal":
        raise SystemExit("Claude refused to generate the draft.")
    if response.stop_reason == "max_tokens":
        raise SystemExit("Claude's response hit the token limit; raise max_tokens.")

    text = next((b.text for b in response.content if b.type == "text"), None)
    if text is None:
        raise SystemExit("Claude returned no text content.")

    return draft_from_dict(json.loads(text))


_PUBLIC_COMMENT_SCHEMA: dict = {
    "type": "object",
    "additionalProperties": False,
    "required": ["summary", "quote", "speaker"],
    "properties": {
        "summary": {"type": "string"},
        "quote": {"type": "string"},
        "speaker": {"type": "string"},
    },
}

_PUBLIC_COMMENT_SYSTEM = """\
You are given an auto-captioned transcript of the PUBLIC COMMENT portion of a \
Williamsburg-James City County (WJCC) School Board meeting, where residents \
addressed the board. Summarize it for a community newsletter recap.

Write:
- summary: 2-4 plain, neutral sentences describing the main topics speakers \
  raised and roughly how the comment broke down (e.g. which topic drew the most \
  speakers). Do not editorialize or take sides.
- quote: ONE short verbatim line, copied exactly from the transcript, from a \
  speaker addressing the MOST-discussed topic. Copy and paste; do not paraphrase \
  or clean up grammar. If the transcript is too garbled to quote faithfully, \
  return an empty string.
- speaker: the quoted speaker's name ONLY if it is clearly stated in the \
  transcript; otherwise an empty string. Never guess a name.

Rules:
- Use only what is in the transcript. Never invent topics, names, or positions.
- This is an auto-generated caption track and may be imperfect; when unsure, \
  prefer a shorter summary and an empty quote over anything you are not certain \
  the transcript supports.\
"""


def summarize_public_comment(
    transcript_text: str,
    *,
    top_topic: str | None = None,
    model: str = MODEL,
    client: anthropic.Anthropic | None = None,
) -> PublicCommentSummary | None:
    """Summarize the public-comment portion of a meeting (recap only).

    `transcript_text` is the public-comment slice of the meeting transcript.
    `top_topic` is an optional hint (the agenda item that drew the most public
    comment time, from segmentation) to steer which topic the quote comes from.
    Returns None when there is nothing to summarize.
    """
    if not transcript_text.strip():
        return None

    if client is None:
        _load_dotenv()
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise SystemExit("ANTHROPIC_API_KEY is not set (env var or .env file).")
        client = anthropic.Anthropic()

    hint = (
        f"\n\nBy meeting timing, the topic that drew the most public comment was: "
        f"{top_topic}. Prefer a quote from a speaker on that topic.\n"
        if top_topic
        else ""
    )
    user_msg = (
        "PUBLIC COMMENT TRANSCRIPT (auto-captioned):\n"
        + transcript_text
        + hint
        + "\n\nSummarize the public comment."
    )

    # Stream: the transcript can be long input, so streaming guards against
    # request timeouts even though the summary output itself is small.
    try:
        with client.messages.stream(
            model=model,
            max_tokens=4000,
            output_config={
                "effort": "medium",
                "format": {"type": "json_schema", "schema": _PUBLIC_COMMENT_SCHEMA},
            },
            system=_PUBLIC_COMMENT_SYSTEM,
            messages=[{"role": "user", "content": user_msg}],
        ) as stream:
            resp = stream.get_final_message()
    except anthropic.APIError as exc:
        raise SystemExit(f"Public-comment summary API call failed: {exc}")

    if resp.stop_reason == "refusal":
        print("  ! Claude declined to summarize public comment; skipping section.")
        return None

    text = next((b.text for b in resp.content if b.type == "text"), None)
    if not text:
        return None
    return public_comment_from_dict(json.loads(text))


def validate_draft(the_draft: Draft, ranked_top: list[ScoredItem]) -> list[str]:
    """Return human-readable notes about anything off in the draft."""
    notes: list[str] = []
    known = {s.item.number for s in ranked_top if s.item.number}
    for di in the_draft.items:
        if di.number and di.number not in known:
            notes.append(f"draft item references unknown agenda number {di.number!r}")
        if not di.quotes:
            notes.append(f"item {di.number or di.headline!r} has no quotes")
    return notes


# --- CLI -------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("html", type=pathlib.Path, help="agenda HTML file")
    parser.add_argument("meta", type=pathlib.Path, help="meeting-meta JSON file")
    parser.add_argument("--top", type=int, default=5, help="top-N items to draft (default 5)")
    parser.add_argument("--json", action="store_true", help="print raw draft JSON")
    args = parser.parse_args()

    for p in (args.html, args.meta):
        if not p.is_file():
            sys.exit(f"No such file: {p}")

    html = args.html.read_text()
    result = triage(parse_agenda(html), json.loads(args.meta.read_text()), agenda_preamble(html))
    scored = finalize(compute_deterministic(kept_items(result), result))
    top = scored[: args.top]

    print(f"Drafting prose for top {len(top)} items...")
    the_draft = draft(top)

    for note in validate_draft(the_draft, top):
        print(f"  ! {note}")

    if args.json:
        print(json.dumps(dataclasses.asdict(the_draft), indent=2))
        return

    print(f"\nIntro:\n  {the_draft.intro}\n")
    for di in the_draft.items:
        print(f"[{di.number or '-'}] {di.headline}")
        print(f"  {di.what_it_is}")
        for q in di.quotes:
            print(f"  > ({q.source}) \"{q.quote}\"")
        print()


if __name__ == "__main__":
    main()
