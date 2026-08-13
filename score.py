#!/usr/bin/env python3
"""Score triaged agenda items by measurable signals and rank them.

All selection lives here. write.py only writes prose for the already-ranked
top items. The composite is a transparent weighted sum of normalized signals;
weights are constants at the top of this file — tune by editing.

Signals (all normalized 0-1 before weighting):
  discussion_minutes — minutes the work session spent on this item (heaviest)
  dollar_magnitude   — log-scaled largest dollar figure in the item text
  is_vote            — non-consent Action item type
  off_consent        — vote that requires individual board action (not batched)
  is_routine         — penalty for annual/recurring boilerplate titles
  attachment_count   — number of attached documents (lightest)
  rubric_score       — 0-5 explicit-rubric score from Claude (÷5)

discussion_minutes requires a transcript + a Claude segmentation call.
rubric_score requires a Claude rubric call.
Both are skipped in --dry-run mode (shown as 0 in the checkpoint table).

The evidence line in the newsletter shows RAW signal values, never the
normalized ones — that is what keeps the ranking auditable.

Usage:
    python score.py wjcc-fixtures/agenda-20260519.html \\
                    wjcc-fixtures/meeting-meta-20260519.json
    python score.py <agenda.html> <meeting-meta.json> --dry-run
"""

import argparse
import dataclasses
import json
import math
import os
import pathlib
import re
import sys
from dataclasses import dataclass, field

import anthropic

from parse import AgendaItem, agenda_preamble, parse_agenda
from triage import TriageResult, kept_items, triage

# ---------------------------------------------------------------------------
# Weights — tune by editing. Applied to normalized (0-1) signal values.
# Positive = higher score is better. Negative = penalty.
# discussion_minutes is weighted heaviest: that is the "Most Discussed" premise.

W_DISCUSSION = 0.40
W_DOLLAR     = 0.20
W_VOTE       = 0.10
W_OFF_CONSENT = 0.10
W_ROUTINE    = -0.10   # penalty: routine items lose up to 0.10
W_ATTACHMENT = 0.05
W_RUBRIC     = 0.15

# Recap-only signals. These are 0 for every item in a preview run (no meeting
# transcript, no recorded votes), so adding their weighted terms leaves the
# preview composite — and ordering — mathematically unchanged.
W_PUBLIC_COMMENT = 0.20   # minutes residents spoke on the item at the meeting
W_VOTE_CONTESTED = 0.10   # the vote was split or failed (newsworthy by itself)

MODEL = "claude-sonnet-4-6"

_DOLLAR_RE = re.compile(r"\$[\d,]*\d(?:\.\d{2})?")
_ROUTINE_RE = re.compile(
    r"(?i)\b("
    r"annual|monthly|pay\s+scales?|bills?\s+and\s+payroll|"
    r"bills?\s+&\s+payroll|payroll\s+and\s+bills?|payroll|"
    r"approval\s+of\s+minutes|personnel\s+appointment\s+book|"
    r"superintendent'?s?\s+report|classification\s+plan"
    r")\b"
)


# --- Data model ------------------------------------------------------------

@dataclass
class Signals:
    discussion_minutes: float = 0.0   # raw minutes discussed (work session, or meeting in recap)
    dollar_magnitude: float = 0.0     # log10 of largest dollar figure, or 0
    dollar_raw: str = ""              # the verbatim dollar string (evidence line)
    is_vote: bool = False             # non-consent Action item
    off_consent: bool = False         # action requiring individual board vote
    is_routine: bool = False          # penalty flag: annual/recurring pattern
    attachment_count: int = 0         # number of PDFs attached
    rubric_score: float = -1.0        # 0-5; -1 = not yet scored
    rubric_justification: str = ""
    # Recap-only signals (0 in a preview run).
    public_comment_minutes: float = 0.0  # minutes residents spoke on this item
    vote_contested: bool = False         # parsed vote was split or failed
    # Recap deep-link anchors: earliest second the item appears in each video,
    # so the recap can link straight to "watch this item." None = not located.
    meeting_start_seconds: float | None = None
    work_session_start_seconds: float | None = None


@dataclass
class ScoredItem:
    item: AgendaItem
    signals: Signals
    normalized: dict[str, float] = field(default_factory=dict)
    composite: float = 0.0
    rank: int = 0


# --- Deterministic signals -------------------------------------------------

def _max_dollars(item: AgendaItem) -> tuple[float, str]:
    """Return (log10_magnitude, raw_string) for the largest dollar figure.

    Checks recommended_action first (most authoritative headline figure),
    then the body's dollar_figures list. Returns (0.0, "") if none found.
    """
    candidates: list[str] = []
    if item.recommended_action:
        candidates.extend(_DOLLAR_RE.findall(item.recommended_action))
    candidates.extend(item.dollar_figures)

    if not candidates:
        return 0.0, ""

    def _parse(s: str) -> float:
        try:
            return float(s.lstrip("$").replace(",", ""))
        except ValueError:
            return 0.0

    best = max(candidates, key=_parse)
    amount = _parse(best)
    if amount <= 0:
        return 0.0, ""
    return math.log10(amount), best


def compute_deterministic(
    items: list[AgendaItem], result: TriageResult
) -> list[ScoredItem]:
    """Compute all deterministic signals. Returns ScoredItems in agenda order.

    Normalized values and composite are not yet computed — call finalize()
    once all signals (including any API-derived ones) are filled in.
    """
    action_set = {i.number for i in result.action_items if i.number}

    scored: list[ScoredItem] = []
    for item in items:
        log_mag, raw_dollar = _max_dollars(item)
        is_vote = "action" in item.item_type.lower() and "consent" not in item.item_type.lower()
        scored.append(ScoredItem(
            item=item,
            signals=Signals(
                dollar_magnitude=log_mag,
                dollar_raw=raw_dollar,
                is_vote=is_vote,
                off_consent=item.number in action_set,
                is_routine=bool(_ROUTINE_RE.search(item.title)),
                attachment_count=len(item.attachments),
                vote_contested=bool(item.vote and item.vote.contested),
            ),
        ))
    return scored


# --- API-derived signals ---------------------------------------------------

def _compact_transcript(snippets: list[dict]) -> str:
    """Group snippets into 30-second windows for the segmentation prompt.

    Produces ~(total_seconds/30) lines of 'N[s] text...' — compact enough for
    the segmentation prompt while preserving checkable timestamps.
    """
    if not snippets:
        return ""
    WINDOW = 30.0
    current_start = snippets[0]["start"]
    current_texts: list[str] = []
    groups: list[str] = []
    for s in snippets:
        if s["start"] >= current_start + WINDOW and current_texts:
            groups.append(f"[{int(current_start)}s] {' '.join(current_texts)}")
            current_start = s["start"]
            current_texts = []
        t = s["text"].strip()
        if t:
            current_texts.append(t)
    if current_texts:
        groups.append(f"[{int(current_start)}s] {' '.join(current_texts)}")
    return "\n".join(groups)


_RANGE_ARRAY: dict = {
    "type": "array",
    "items": {
        "type": "object",
        "additionalProperties": False,
        "required": ["start_seconds", "end_seconds"],
        "properties": {
            "start_seconds": {"type": "number"},
            "end_seconds": {"type": "number"},
        },
    },
}

_SEGMENTATION_SCHEMA: dict = {
    "type": "object",
    "additionalProperties": False,
    "required": ["segments", "public_comment_period"],
    "properties": {
        "segments": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["item_number", "ranges", "public_comment_ranges"],
                "properties": {
                    "item_number": {"type": "string"},
                    # board discussion / debate / the vote on the item
                    "ranges": _RANGE_ARRAY,
                    # time a member of the public addressed this item's topic
                    "public_comment_ranges": _RANGE_ARRAY,
                },
            },
        },
        # the whole public-comment portion of the meeting (all speakers, on- and
        # off-agenda); empty for a work session that has no public comment
        "public_comment_period": _RANGE_ARRAY,
    },
}

_SEGMENTATION_SYSTEM = """\
You are given a timestamped transcript of a school board meeting and a list of \
agenda items. Your task is purely mechanical: for each agenda item, identify the \
time range(s) where it appears in the transcript.

For each item return two separate sets of ranges:
- ranges: where the BOARD discussed, debated, or voted on the item.
- public_comment_ranges: where a member of the PUBLIC (during the public-comment \
  period) clearly spoke about this item's topic. Only attribute a speaker to an \
  item when the connection is clear; comment on topics not on the agenda belongs \
  to no item.

Also return public_comment_period: the time range(s) covering the ENTIRE \
public-comment portion of the meeting — every speaker, whether or not their \
topic is on the agenda. Return an empty list if the transcript has no public-\
comment period (for example a work session).

Rules:
- Match by topic: use agenda item numbers and titles to find each topic.
- If an item does not appear in a category, return an empty list for it.
- Ranges should be tight: start where that segment begins, end where it moves on.
- Do not rank, evaluate, or comment on importance — your only job is timestamps.
- Timestamps are seconds from the recording start.\
"""


def segment_transcript(
    scored: list[ScoredItem],
    snippets: list[dict],
    client: anthropic.Anthropic,
    *,
    kind: str = "meeting",
    score: bool = True,
) -> dict:
    """Run the segmentation call; populate timing signals in-place.

    This is Claude used as a *measuring instrument*: given the transcript and
    the agenda titles, it identifies timestamp ranges. The output is directly
    verifiable by seeking to those seconds in the YouTube video.

    kind ("meeting" / "work_session") selects which deep-link start anchor each
    item's earliest appearance is written to. When `score` is True the
    discussion/public-comment minutes feed the ranking; pass False to record
    only the deep-link anchors (e.g. segmenting the work session in a recap,
    where the meeting transcript already supplied the scoring signals).

    Returns the parsed segmentation data (including `public_comment_period`),
    or {} when there is nothing to segment.
    """
    if not snippets:
        print("  No transcript snippets — discussion_minutes will be 0 for all items.")
        return {}

    items_list = "\n".join(
        f"  {s.item.number or 'N/A'} — {s.item.title}" for s in scored
    )
    transcript_text = _compact_transcript(snippets)

    user_msg = (
        "AGENDA ITEMS (number — title):\n"
        + items_list
        + "\n\nTRANSCRIPT (seconds from recording start):\n"
        + transcript_text
        + "\n\nFor each agenda item, return the time ranges where it was discussed."
    )

    print("  Calling Claude for transcript segmentation...", flush=True)
    try:
        resp = client.messages.create(
            model=MODEL,
            max_tokens=4096,
            output_config={
                "effort": "medium",
                "format": {"type": "json_schema", "schema": _SEGMENTATION_SCHEMA},
            },
            system=_SEGMENTATION_SYSTEM,
            messages=[{"role": "user", "content": user_msg}],
        )
    except anthropic.APIError as exc:
        raise SystemExit(f"Segmentation API call failed: {exc}")

    text = next((b.text for b in resp.content if b.type == "text"), None)
    if not text:
        raise SystemExit("Segmentation call returned no text.")

    def _minutes(ranges: list[dict]) -> float:
        return sum(max(0.0, r["end_seconds"] - r["start_seconds"]) / 60.0 for r in ranges)

    data = json.loads(text)
    disc_by_number: dict[str, float] = {}
    pubcom_by_number: dict[str, float] = {}
    start_by_number: dict[str, float] = {}
    for seg in data.get("segments", []):
        num = seg.get("item_number", "")
        ranges = seg.get("ranges", [])
        disc_by_number[num] = disc_by_number.get(num, 0.0) + _minutes(ranges)
        pubcom_by_number[num] = pubcom_by_number.get(num, 0.0) + _minutes(
            seg.get("public_comment_ranges", [])
        )
        starts = [r["start_seconds"] for r in ranges]
        if starts:
            earliest = min(starts)
            start_by_number[num] = min(start_by_number.get(num, earliest), earliest)

    for s in scored:
        key = s.item.number or "N/A"
        if score:
            s.signals.discussion_minutes = round(disc_by_number.get(key, 0.0), 1)
            s.signals.public_comment_minutes = round(pubcom_by_number.get(key, 0.0), 1)
        start = start_by_number.get(key)
        if start is not None:
            if kind == "work_session":
                s.signals.work_session_start_seconds = round(start)
            else:
                s.signals.meeting_start_seconds = round(start)

    if score:
        total = sum(s.signals.discussion_minutes for s in scored)
        pubcom = sum(s.signals.public_comment_minutes for s in scored)
        print(
            f"  Segmentation done. {total:.1f} min board discussion + "
            f"{pubcom:.1f} min public comment mapped across {len(scored)} items."
        )
    else:
        located = sum(1 for k in start_by_number if k != "N/A")
        print(f"  {kind} segmentation done. {located} items located for deep-links.")

    return data


_RUBRIC_SCHEMA: dict = {
    "type": "object",
    "additionalProperties": False,
    "required": ["scores"],
    "properties": {
        "scores": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["item_number", "score", "justification"],
                "properties": {
                    "item_number": {"type": "string"},
                    "score": {"type": "integer"},
                    "justification": {"type": "string"},
                },
            },
        },
    },
}

_RUBRIC_SYSTEM = """\
You are given a rubric and a list of school board agenda items. For each item, \
score it 0-5 according to the rubric and write a one-sentence justification.

This is a mechanical scoring task. Apply the rubric consistently and literally. \
Do not add commentary beyond the score and justification.\
"""


def score_rubric(
    scored: list[ScoredItem],
    rubric_text: str,
    client: anthropic.Anthropic,
) -> None:
    """Run the rubric call and populate rubric_score/rubric_justification in-place."""
    items_block = "\n\n".join(
        f"Item {s.item.number or 'N/A'}: {s.item.title}\n"
        f"Type: {s.item.item_type}\n"
        + (f"Recommended action: {s.item.recommended_action}\n" if s.item.recommended_action else "")
        + (f"Body:\n{s.item.body[:1000]}" if s.item.body else "")
        + (f"\nAttachments: {', '.join(a.name for a in s.item.attachments)}" if s.item.attachments else "")
        for s in scored
    )
    user_msg = f"RUBRIC:\n{rubric_text}\n\nAGENDA ITEMS:\n{items_block}\n\nScore each item 0-5."

    print("  Calling Claude for rubric scoring...", flush=True)
    try:
        resp = client.messages.create(
            model=MODEL,
            max_tokens=4096,
            output_config={
                "effort": "medium",
                "format": {"type": "json_schema", "schema": _RUBRIC_SCHEMA},
            },
            system=_RUBRIC_SYSTEM,
            messages=[{"role": "user", "content": user_msg}],
        )
    except anthropic.APIError as exc:
        raise SystemExit(f"Rubric API call failed: {exc}")

    text = next((b.text for b in resp.content if b.type == "text"), None)
    if not text:
        raise SystemExit("Rubric call returned no text.")

    data = json.loads(text)
    scores_by_number: dict[str, tuple[int, str]] = {
        entry["item_number"]: (entry["score"], entry["justification"])
        for entry in data.get("scores", [])
    }
    for s in scored:
        key = s.item.number or "N/A"
        if key in scores_by_number:
            sc, just = scores_by_number[key]
            s.signals.rubric_score = float(max(0, min(5, sc)))
            s.signals.rubric_justification = just

    scored_count = sum(1 for s in scored if s.signals.rubric_score >= 0)
    print(f"  Rubric scoring done. {scored_count}/{len(scored)} items scored.")


# --- Normalization & ranking -----------------------------------------------

def _minmax(values: list[float]) -> list[float]:
    mn, mx = min(values), max(values)
    if mx == mn:
        return [0.0] * len(values)
    return [(v - mn) / (mx - mn) for v in values]


def finalize(scored: list[ScoredItem]) -> list[ScoredItem]:
    """Normalize signals, compute composite scores, sort descending. Returns sorted list."""
    if not scored:
        return scored

    disc  = _minmax([s.signals.discussion_minutes for s in scored])
    doll  = _minmax([s.signals.dollar_magnitude for s in scored])
    att   = _minmax([float(s.signals.attachment_count) for s in scored])
    rubric = [max(0.0, s.signals.rubric_score) / 5.0 for s in scored]
    pubcom = _minmax([s.signals.public_comment_minutes for s in scored])

    for i, s in enumerate(scored):
        n = {
            "discussion_minutes": disc[i],
            "dollar_magnitude":   doll[i],
            "is_vote":            1.0 if s.signals.is_vote else 0.0,
            "off_consent":        1.0 if s.signals.off_consent else 0.0,
            "is_routine":         1.0 if s.signals.is_routine else 0.0,
            "attachment_count":   att[i],
            "rubric_score":       rubric[i],
            "public_comment_minutes": pubcom[i],
            "vote_contested":     1.0 if s.signals.vote_contested else 0.0,
        }
        s.normalized = n
        s.composite = (
            W_DISCUSSION  * n["discussion_minutes"]
            + W_DOLLAR    * n["dollar_magnitude"]
            + W_VOTE      * n["is_vote"]
            + W_OFF_CONSENT * n["off_consent"]
            + W_ROUTINE   * n["is_routine"]
            + W_ATTACHMENT * n["attachment_count"]
            + W_RUBRIC    * n["rubric_score"]
            + W_PUBLIC_COMMENT * n["public_comment_minutes"]
            + W_VOTE_CONTESTED * n["vote_contested"]
        )

    scored.sort(key=lambda s: s.composite, reverse=True)
    for i, s in enumerate(scored, 1):
        s.rank = i
    return scored


# --- Evidence line (for checkpoint + newsletter) ---------------------------

def vote_summary(item: AgendaItem) -> str:
    """Plain outcome for an item's recorded vote, e.g. 'Approved 7-0' / 'Failed 3-4'.

    Reads the deterministically parsed vote; returns 'No recorded vote' when the
    agenda carries no Motion & Voting block (a preview agenda, or an item that
    was not acted on).
    """
    v = item.vote
    if not v:
        return "No recorded vote"
    verb = "Approved" if v.passed else "Failed"
    return f"{verb} {v.tally}"


def evidence_line(s: ScoredItem) -> str:
    """One compact line of raw signal evidence, e.g. '24 min · $219M · vote (direct)'.

    In a recap the item carries a parsed vote, so the outcome ('Approved 7-0')
    replaces the preview's prospective 'vote (direct)' label. Public-comment
    minutes are a ranking signal only and are intentionally never shown here.
    """
    parts: list[str] = []
    if s.signals.discussion_minutes > 0:
        parts.append(f"{s.signals.discussion_minutes:.1f} min discussed")
    if s.signals.dollar_raw:
        parts.append(s.signals.dollar_raw)
    if s.item.vote:
        parts.append(vote_summary(s.item))
    elif s.signals.is_vote:
        parts.append("vote (direct)")
    elif "consent" in s.item.item_type.lower():
        parts.append("vote (consent)")
    else:
        parts.append("report/info")
    if s.signals.is_routine:
        parts.append("routine")
    if s.signals.attachment_count:
        parts.append(f"{s.signals.attachment_count} attach")
    return " · ".join(parts) if parts else "—"


# --- CLI -------------------------------------------------------------------

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


def _print_table(scored: list[ScoredItem], *, dry_run: bool = False) -> None:
    note = " (dry-run: disc=0, rubric=0)" if dry_run else ""
    print(f"\n{'='*72}")
    print(f"SCORE TABLE{note}")
    print(f"{'='*72}")
    hdr = f"{'Rk':>3}  {'#':>6}  {'Score':>6}  {'Disc':>5}  {'PC':>4}  {'Dollar':<12}  {'V':>1}  {'OC':>2}  {'Rt':>2}  {'At':>2}  {'Rub':>3}  {'Vote':>10}  Title"
    print(hdr)
    print("-" * 72)
    for s in scored:
        sig = s.signals
        rub = f"{sig.rubric_score:.0f}" if sig.rubric_score >= 0 else " -"
        v  = "Y" if sig.is_vote else " "
        oc = "Y" if sig.off_consent else " "
        rt = "Y" if sig.is_routine else " "
        outcome = s.item.vote.tally if s.item.vote else ""
        if s.item.vote and not s.item.vote.passed:
            outcome += " FAIL"
        print(
            f"{s.rank:>3}  {s.item.number or '-':>6}  {s.composite:>6.3f}"
            f"  {sig.discussion_minutes:>5.1f}  {sig.public_comment_minutes:>4.1f}"
            f"  {sig.dollar_raw or '':<12}"
            f"  {v}  {oc:>2}  {rt:>2}  {sig.attachment_count:>2}  {rub:>3}"
            f"  {outcome:>10}"
            f"  {s.item.title[:40]}"
        )
    print(f"{'='*72}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("html", type=pathlib.Path, help="agenda HTML file")
    parser.add_argument("meta", type=pathlib.Path, help="meeting-meta JSON file")
    parser.add_argument("--work-session", metavar="URL",
                        help="YouTube URL for the work session transcript")
    parser.add_argument("--dry-run", action="store_true",
                        help="deterministic signals only, no API calls")
    args = parser.parse_args()

    for p in (args.html, args.meta):
        if not p.is_file():
            sys.exit(f"No such file: {p}")

    html = args.html.read_text()
    items = parse_agenda(html)
    result = triage(items, json.loads(args.meta.read_text()), agenda_preamble(html))
    all_items = kept_items(result)
    scored = compute_deterministic(all_items, result)

    if not args.dry_run:
        _load_dotenv()
        if not os.environ.get("ANTHROPIC_API_KEY"):
            sys.exit("ANTHROPIC_API_KEY is not set; use --dry-run or set the key.")
        client = anthropic.Anthropic()

        if args.work_session:
            from transcript import fetch_transcript
            snippets = fetch_transcript(args.work_session)
            segment_transcript(scored, snippets, client)

        rubric_path = pathlib.Path(__file__).resolve().parent / "rubric.md"
        if rubric_path.is_file():
            score_rubric(scored, rubric_path.read_text(), client)
        else:
            print("  WARNING: rubric.md not found — rubric_score will be 0.")

    finalize(scored)
    _print_table(scored, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
