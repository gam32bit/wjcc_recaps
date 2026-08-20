#!/usr/bin/env python3
"""Recover each item's recorded vote from the meeting's own video transcript.

The district moved to Diligent Community on 2026-07-01. A Diligent agenda packet
is published BEFORE the meeting and is never amended afterwards, so unlike the
BoardDocs agendas the pipeline grew up on, it carries no "Motion & Voting" block
— only the recommended action. The outcomes are not lost, though: the board
takes every vote by roll call, aloud, on a livestream that is captioned.

Claude's role here is the same narrow measuring-instrument role it plays in
`score.segment_transcript` and `pubcomment.classify_speakers`: it reports what
was said in each roll call — which item, who moved, who seconded, and how each
member answered. It does no arithmetic and reaches no conclusion. Python counts
the lists, decides carried/failed, and checks the totals.

**Why this cannot be a regex.** The affirmative is transcribed as "I", "Hi",
"I.", and "Aye" interchangeably, and "aye" matches inside *Lafayette* and
*Warhill* — a plain search of the Aug 18 transcript returns 16 hits, every one
of them a school name. The members are worse: the same seven people appear as
Hodgees/Hodes/Hodges, Kavasos/Kvassos/Cabazos/Kasos, Hosang/Hosen/Hosing,
Riffle/Ripple/Riffel, Chen/Chin, Hunley/Huntley. No pattern survives that.

**Why the names are counted but never printed.** They come from an auto-caption
track, and the project's standing rule (see `pubcomment.py`) is that such
spellings are not published. So a transcript-sourced vote publishes its counts —
which are robust, since a miscaught spelling is still one voice — and its
`start_seconds`, so a reader can seek the roll call and hear it. The names stay
in the audit JSON, where they can be checked, and out of the newsletter. A vote
parsed from a BoardDocs agenda is unaffected: those names are printed, because
there they are the district's own record rather than a transcription of it.
"""

import dataclasses
import json

import anthropic

from parse import AgendaItem, Vote
from score import MODEL, compact_transcript


def _seated(raw: list[dict]) -> int:
    """How many members the clerk actually called, at this meeting.

    Measured, not assumed. The board has seven seats, but a seat can be vacant
    and an absent member is often simply not called — on Aug 18 2026 every roll
    call ran to six. Hardcoding seven would then have flagged all fourteen votes
    as suspect, which trains a reviewer to ignore the warning. The largest roll
    call of the night is the number seated; a SHORTER one is the thing worth
    looking at, because it means a member who was there did not answer.
    """
    return max(
        (len(e.get("aye") or []) + len(e.get("nay") or [])
         + len(e.get("abstain") or []) + len(e.get("absent") or [])
         for e in raw),
        default=0,
    )


_SCHEMA: dict = {
    "type": "object",
    "additionalProperties": False,
    "required": ["votes"],
    "properties": {
        "votes": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "item_numbers", "start_seconds", "motion_text",
                    "mover", "seconder", "aye", "nay", "abstain", "absent",
                ],
                "properties": {
                    "item_numbers": {"type": "array", "items": {"type": "string"}},
                    "start_seconds": {"type": "number"},
                    "motion_text": {"type": "string"},
                    "mover": {"type": "string"},
                    "seconder": {"type": "string"},
                    "aye": {"type": "array", "items": {"type": "string"}},
                    "nay": {"type": "array", "items": {"type": "string"}},
                    "abstain": {"type": "array", "items": {"type": "string"}},
                    "absent": {"type": "array", "items": {"type": "string"}},
                },
            },
        },
    },
}

_SYSTEM = """\
You are given a school board meeting transcript and the meeting's agenda items. \
The board disposes of items by roll call: a member moves, another seconds, and \
the clerk reads each member's name in turn for their answer. Your task is \
purely mechanical: report each roll call.

Return one entry per ROLL CALL, not per item:
- item_numbers: the agenda items this one vote disposed of, copied EXACTLY from \
  the agenda list. A consent agenda is a single roll call covering several \
  items, so list all of them. Use an empty list for a procedural vote (approving \
  the agenda, certifying a closed session) — those are not agenda items.
- start_seconds: the timestamp where the motion begins. Timestamps are seconds \
  from the recording start, shown in the transcript as [1234s].
- motion_text: the motion as moved, verbatim.
- mover / seconder: the member named, spelled as the transcript spells them. \
  Use "" if not stated.
- aye / nay / abstain / absent: the members answering each way, spelled as the \
  transcript spells them, one entry per member.

Rules:
- The affirmative is transcribed inconsistently — "Aye", "I", "Hi", "Yes" all \
  mean aye. "No" and "Nay" mean nay. "Abstain" means abstain.
- A member the clerk calls who does not answer is absent, not an abstention.
- Report every member the clerk calls, once each, in the order called.
- Do NOT total anything, do not say whether the motion carried, and do not \
  judge importance. Names, answers and timestamps only.
- Names may be misspelled by the captioner. Copy what the transcript says; do \
  not correct or normalize it.
- If an item was never voted on, return no entry for it. Do not invent one.\
"""


def extract(
    items: list[AgendaItem],
    snippets: list[dict],
    client: anthropic.Anthropic,
    *,
    model: str = MODEL,
) -> list[dict]:
    """Ask Claude to report the meeting's roll calls. Returns raw entries."""
    if not snippets:
        return []

    items_list = "\n".join(f"{i.number} — {i.title}" for i in items if i.number)
    user_msg = (
        "AGENDA ITEMS (number — title):\n"
        + items_list
        + "\n\nMEETING TRANSCRIPT (auto-captioned, seconds from recording start):\n"
        + compact_transcript(snippets)
        + "\n\nReport every roll-call vote."
    )

    print("  Calling Claude to read the roll-call votes...", flush=True)
    try:
        resp = client.messages.create(
            model=model,
            max_tokens=8192,
            # Explicitly off — see the _THINKING note in score.py.
            thinking={"type": "disabled"},
            output_config={
                "effort": "medium",
                "format": {"type": "json_schema", "schema": _SCHEMA},
            },
            system=_SYSTEM,
            messages=[{"role": "user", "content": user_msg}],
        )
    except anthropic.APIError as exc:
        raise SystemExit(f"Roll-call vote call failed: {exc}")

    text = next((b.text for b in resp.content if b.type == "text"), None)
    if not text:
        raise SystemExit("Roll-call vote call returned no text.")
    return json.loads(text).get("votes", [])


def to_votes(
    raw: list[dict], items: list[AgendaItem], *, verbose: bool = True
) -> tuple[dict[str, Vote], list[str]]:
    """Turn raw roll calls into `Vote` records, keyed by item number.

    Every count is computed here, in Python, from the name lists the model
    reported. `result` is derived the same way: more ayes than nays carries.
    Returns `(votes_by_number, warnings)` — a warning is raised rather than a
    silent correction, because a roll call whose arithmetic does not work is a
    transcription problem for a human to look at, not something to guess at.
    """
    known = {i.number for i in items if i.number}
    votes: dict[str, Vote] = {}
    warnings: list[str] = []
    seated = _seated(raw)
    if verbose and seated:
        print(f"  {seated} members answered the fullest roll call of this meeting.")

    for entry in raw:
        numbers = [n for n in entry.get("item_numbers", []) if n in known]
        unknown = [n for n in entry.get("item_numbers", []) if n not in known]
        if unknown:
            warnings.append(
                f"roll call at {entry.get('start_seconds', 0):.0f}s named "
                f"unknown item(s) {', '.join(unknown)} — dropped"
            )
        if not numbers:
            continue   # procedural vote, or nothing recognizable

        aye = list(entry.get("aye") or [])
        nay = list(entry.get("nay") or [])
        abstain = list(entry.get("abstain") or [])
        absent = list(entry.get("absent") or [])
        called = len(aye) + len(nay) + len(abstain) + len(absent)
        if called < seated:
            warnings.append(
                f"[{numbers[0]}] roll call at {entry.get('start_seconds', 0):.0f}s "
                f"accounts for {called} of the {seated} members seated "
                f"({len(aye)} aye / {len(nay)} nay / {len(abstain)} abstain / "
                f"{len(absent)} absent) — check the video"
            )

        result = "Motion Carries" if len(aye) > len(nay) else "Motion Fails"
        for number in numbers:
            votes[number] = Vote(
                result=result,
                motion_text=(entry.get("motion_text") or "").strip(),
                mover=(entry.get("mover") or "").strip() or None,
                seconder=(entry.get("seconder") or "").strip() or None,
                aye=aye, nay=nay, abstain=abstain, absent=absent,
                source="transcript",
                start_seconds=float(entry.get("start_seconds") or 0) or None,
            )
            if verbose:
                mark = " *" if votes[number].contested else ""
                print(f"  vote [{number}] {result.replace('Motion ', '')} "
                      f"{votes[number].tally}{mark}")

    return votes, warnings


def apply_to_items(votes: dict[str, Vote], items: list[AgendaItem]) -> int:
    """Attach recovered votes to their items in place. Returns how many landed.

    An item that already carries a vote keeps it: a BoardDocs agenda's own
    Motion & Voting block is the district's record, and a transcription of the
    same roll call is not an improvement on it.
    """
    attached = 0
    for item in items:
        if item.vote is None and item.number in votes:
            item.vote = votes[item.number]
            attached += 1
    return attached


def to_json(raw: list[dict], votes: dict[str, Vote], warnings: list[str]) -> str:
    """The audit record: what the model heard, what was counted, what looked off."""
    return json.dumps({
        "roll_calls": raw,
        "votes": {n: dataclasses.asdict(v) for n, v in votes.items()},
        "warnings": warnings,
    }, indent=2)
