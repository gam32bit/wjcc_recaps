#!/usr/bin/env python3
"""Count who spoke about what during the public-comment portion of a meeting.

The recap does not summarize public comment in prose — it publishes a tally:
how many residents spoke on each agenda item, ranked, plus the off-agenda
topics they raised. Prose summaries of an auto-caption track proved too costly
to fact-check, and the relative claims in them ("the majority", "a smaller
number") had no counted evidence behind them.

Claude's role here is the same narrow measuring-instrument role it plays in
score.py's segmentation: for each speaker it returns a start timestamp and
which agenda item (if any) they addressed. It never writes a sentence, never
counts, and never returns a name — the auto-caption spellings are unreliable
(the chair says "Herbist" where the speaker says "Herpst"), so no name is
published. All arithmetic below is plain Python, and every speaker is written
out with a timestamp so a reviewer can seek the video and check the call.
"""

import dataclasses
import json
from dataclasses import dataclass, field

import anthropic

from parse import AgendaItem
from score import compact_transcript

MODEL = "claude-sonnet-4-6"

# Speaker attribution needs a finer view of the transcript than segmentation's
# 30-second windows: two-minute comments would otherwise share a window.
WINDOW_SECONDS = 15.0


# --- Data model ------------------------------------------------------------

@dataclass
class Speaker:
    """One resident's turn at the podium, as located by the model."""
    start_seconds: float
    item_number: str = ""   # an agenda item number, or "" when off-agenda
    topic_label: str = ""   # short topic wording; only set when off-agenda


@dataclass
class TopicCount:
    label: str              # agenda item title, or the off-agenda topic label
    count: int
    item_number: str = ""   # "" for off-agenda topics


@dataclass
class PublicCommentTally:
    total_speakers: int = 0
    by_item: list[TopicCount] = field(default_factory=list)      # ranked
    off_agenda: list[TopicCount] = field(default_factory=list)   # ranked
    minutes_by_item: dict[str, float] = field(default_factory=dict)
    speakers_by_item: dict[str, int] = field(default_factory=dict)

    @property
    def top_item(self) -> TopicCount | None:
        return self.by_item[0] if self.by_item else None


# --- Claude contract -------------------------------------------------------

_SCHEMA: dict = {
    "type": "object",
    "additionalProperties": False,
    "required": ["speakers"],
    "properties": {
        "speakers": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["start_seconds", "item_number", "topic_label"],
                "properties": {
                    "start_seconds": {"type": "number"},
                    "item_number": {"type": "string"},
                    "topic_label": {"type": "string"},
                },
            },
        },
    },
}

_SYSTEM = """\
You are given the public-comment portion of a school board meeting transcript, \
where residents addressed the board one at a time, and the meeting's agenda \
items. Your task is purely mechanical: identify each speaker and what they \
talked about.

Return one entry per member of the PUBLIC who addressed the board:
- start_seconds: the timestamp where that speaker begins. Timestamps are \
  seconds from the recording start, shown in the transcript as [1234s].
- item_number: the agenda item their comment was about, copied exactly from \
  the agenda list. Use "" when their topic is not on the agenda.
- topic_label: leave "" when you set an item_number. When item_number is "", \
  give a short topic in 2-5 words, e.g. "classroom technology use".

Rules:
- Count speakers, not turns. A speaker interrupted by the chair, or one who \
  pauses, is still one speaker.
- Do NOT count the chair, board members, or staff — only members of the public.
- Only set an item_number when the connection to that item is clear. When in \
  doubt, use "" and a topic_label.
- REUSE THE EXACT SAME topic_label for every speaker on the same off-agenda \
  topic, character for character. Prefer the smallest number of distinct \
  labels that covers the comments — these labels are counted, and two spellings \
  of one topic split its count in half.
- Never return a speaker's name, in any field. Names are not wanted.
- Do not rank, evaluate, or comment on importance. Timestamps and topics only.
- This is an auto-generated caption track and may be imperfect. Omit a speaker \
  you are not confident was a member of the public.\
"""


def compact_slice(snippets: list[dict], pc_ranges: list[dict]) -> str:
    """The public-comment portion of a transcript, as timestamped windows.

    Keeps the `[1234s]` markers the model needs to place each speaker — and
    that a reviewer needs to seek the video and check the call.
    """
    if not pc_ranges:
        return ""
    inside = [
        s for s in snippets
        if any(r["start_seconds"] <= s["start"] <= r["end_seconds"] for r in pc_ranges)
    ]
    return compact_transcript(inside, window=WINDOW_SECONDS)


def classify_speakers(
    transcript_text: str,
    items: list[AgendaItem],
    client: anthropic.Anthropic,
    *,
    pc_ranges: list[dict] | None = None,
    model: str = MODEL,
) -> list[Speaker]:
    """Locate each public-comment speaker and the topic they addressed.

    `transcript_text` is the public-comment slice, compacted into timestamped
    windows. `pc_ranges` bounds the valid timestamps, so a speaker the model
    places outside the public-comment period is dropped rather than counted.
    """
    if not transcript_text.strip():
        return []

    items_list = "\n".join(
        f"{i.number} — {i.title}" for i in items if i.number
    )
    user_msg = (
        "AGENDA ITEMS (number — title):\n"
        + items_list
        + "\n\nPUBLIC COMMENT TRANSCRIPT (auto-captioned, seconds from "
        "recording start):\n"
        + transcript_text
        + "\n\nList each speaker and the topic they addressed."
    )

    print("  Calling Claude to identify public-comment speakers...", flush=True)
    try:
        resp = client.messages.create(
            model=model,
            max_tokens=4096,
            output_config={
                "effort": "medium",
                "format": {"type": "json_schema", "schema": _SCHEMA},
            },
            system=_SYSTEM,
            messages=[{"role": "user", "content": user_msg}],
        )
    except anthropic.APIError as exc:
        raise SystemExit(f"Public-comment speaker call failed: {exc}")

    text = next((b.text for b in resp.content if b.type == "text"), None)
    if not text:
        raise SystemExit("Public-comment speaker call returned no text.")

    return validate(json.loads(text).get("speakers", []), items, pc_ranges)


def validate(
    raw: list[dict],
    items: list[AgendaItem],
    pc_ranges: list[dict] | None = None,
) -> list[Speaker]:
    """Turn the raw response into Speakers, dropping anything unsupportable."""
    known = {i.number for i in items if i.number}
    speakers: list[Speaker] = []
    for entry in raw:
        start = float(entry.get("start_seconds", 0.0))
        if pc_ranges and not any(
            r["start_seconds"] <= start <= r["end_seconds"] for r in pc_ranges
        ):
            print(f"  ! speaker at {int(start)}s is outside public comment; dropped")
            continue
        number = (entry.get("item_number") or "").strip()
        label = (entry.get("topic_label") or "").strip()
        if number and number not in known:
            print(f"  ! speaker at {int(start)}s cites unknown item {number!r}; "
                  "counting as off-agenda")
            number = ""
        speakers.append(Speaker(
            start_seconds=start,
            item_number=number,
            topic_label="" if number else label,
        ))
    speakers.sort(key=lambda s: s.start_seconds)
    return speakers


# --- Deterministic tally ---------------------------------------------------

def tally(
    speakers: list[Speaker],
    items: list[AgendaItem],
    pc_ranges: list[dict] | None = None,
) -> PublicCommentTally:
    """Count speakers per topic and estimate minutes per item. No model.

    A speaker's share of the period runs from their start to the next
    speaker's (the last speaker runs to the end of the public-comment period),
    which is what `public_comment_minutes` is derived from — the same ranking
    signal as before, attributed per speaker instead of per time range.
    """
    result = PublicCommentTally(total_speakers=len(speakers))
    if not speakers:
        return result

    period_end = (
        max(r["end_seconds"] for r in pc_ranges)
        if pc_ranges
        else speakers[-1].start_seconds
    )
    titles = {i.number: i.title for i in items if i.number}
    order = {i.number: n for n, i in enumerate(items) if i.number}

    counts: dict[str, int] = {}
    minutes: dict[str, float] = {}
    off_counts: dict[str, int] = {}
    off_labels: dict[str, str] = {}   # lowercase key -> label as first written

    for n, sp in enumerate(speakers):
        nxt = speakers[n + 1].start_seconds if n + 1 < len(speakers) else period_end
        span = max(0.0, nxt - sp.start_seconds) / 60.0
        if sp.item_number:
            counts[sp.item_number] = counts.get(sp.item_number, 0) + 1
            minutes[sp.item_number] = minutes.get(sp.item_number, 0.0) + span
        elif sp.topic_label:
            key = sp.topic_label.casefold()
            off_counts[key] = off_counts.get(key, 0) + 1
            off_labels.setdefault(key, sp.topic_label)

    result.speakers_by_item = counts
    result.minutes_by_item = {k: round(v, 1) for k, v in minutes.items()}
    # Ranked by speaker count, ties broken by agenda order (off-agenda: A-Z).
    result.by_item = sorted(
        (TopicCount(label=titles.get(num, num), count=c, item_number=num)
         for num, c in counts.items()),
        key=lambda t: (-t.count, order.get(t.item_number, 999)),
    )
    result.off_agenda = sorted(
        (TopicCount(label=off_labels[k], count=c) for k, c in off_counts.items()),
        key=lambda t: (-t.count, t.label.casefold()),
    )
    return result


def apply_to_signals(scored: list, result: PublicCommentTally) -> None:
    """Write the tally's per-item figures onto each ScoredItem's Signals."""
    for s in scored:
        num = s.item.number
        s.signals.public_comment_minutes = result.minutes_by_item.get(num, 0.0)
        s.signals.public_comment_speakers = result.speakers_by_item.get(num, 0)


def to_json(
    speakers: list[Speaker],
    result: PublicCommentTally,
    pc_ranges: list[dict] | None,
) -> str:
    """The full audit record: raw speakers, derived tally, and the period.

    Persisting this closes a real gap — segmentation's ranges were previously
    never written to disk, so the public-comment boundary could not be
    re-checked without paying for another API call.
    """
    return json.dumps({
        "public_comment_period": pc_ranges or [],
        "speakers": [dataclasses.asdict(s) for s in speakers],
        "tally": dataclasses.asdict(result),
    }, indent=2)


def from_json(data: dict) -> tuple[list[Speaker], PublicCommentTally, list[dict]]:
    """Rebuild what `to_json` wrote (used by `render.py --recap`)."""
    speakers = [Speaker(**s) for s in data.get("speakers", [])]
    raw = data.get("tally", {})
    result = PublicCommentTally(
        total_speakers=raw.get("total_speakers", 0),
        by_item=[TopicCount(**t) for t in raw.get("by_item", [])],
        off_agenda=[TopicCount(**t) for t in raw.get("off_agenda", [])],
        minutes_by_item=raw.get("minutes_by_item", {}),
        speakers_by_item=raw.get("speakers_by_item", {}),
    )
    return speakers, result, data.get("public_comment_period", [])


def speaker_anchors(speakers: list[Speaker], items: list[AgendaItem]) -> list[tuple[float, str]]:
    """Section anchors for the saved transcript, one per counted speaker.

    Turns the tally into something a reviewer can click through: each counted
    speaker gets a heading in `out/transcript-meeting-<date>.md` naming the
    topic they were counted under.
    """
    titles = {i.number: i.title for i in items if i.number}
    anchors: list[tuple[float, str]] = []
    for n, sp in enumerate(speakers, 1):
        if sp.item_number:
            topic = f"Item {sp.item_number} — {titles.get(sp.item_number, '')}".rstrip(" —")
        else:
            topic = sp.topic_label or "topic unclear"
        anchors.append((sp.start_seconds, f"Speaker {n}: {topic}"))
    return anchors
