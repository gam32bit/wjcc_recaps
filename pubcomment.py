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
import re
from dataclasses import dataclass, field

import anthropic

import llmcache

from parse import AgendaItem
from score import compact_transcript
from titlematch import _LEAD_VERB_RE, _TITLE_MATCH_MIN, _content_words, _overlap

MODEL = "claude-sonnet-5"

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
    # The particular point this speaker made, in the model's own few words —
    # set for EVERY speaker, on-agenda or not. Six residents can speak to one
    # agenda item and be asking for six different things, and a count of six
    # says none of that. Labelling one turn is the same mechanical job as
    # locating it; the counting and grouping below are plain Python.
    subtopic: str = ""
    # Numberdate of the meeting this speaker was recorded at. A period recap
    # pools speakers from several meetings into one list, and a bare timestamp
    # is meaningless without knowing which video it indexes into. "" in a
    # single-meeting recap, where there is only one video.
    meeting: str = ""


@dataclass
class SpeakerAnchor:
    """Where one counted speaker can be found in the video."""
    start_seconds: float
    meeting: str = ""


@dataclass
class TopicCount:
    label: str              # agenda item title, or the off-agenda topic label
    count: int
    item_number: str = ""   # "" for off-agenda topics
    # One anchor per counted speaker, in the order they spoke. The recap links
    # each of them, so the NUMBER OF LINKS IS THE COUNT — the bullet checks
    # itself, and a reader can seek any speaker in five seconds without the
    # newsletter naming or paraphrasing anyone.
    anchors: list[SpeakerAnchor] = field(default_factory=list)
    # {word: times said} across every counted speaker's turn, most-said first.
    # See `_count_words`. Empty when tally() was given no transcript, and
    # absent from files written before this existed. Counted but not currently
    # rendered — see render.py.
    words: dict[str, int] = field(default_factory=dict)
    # This topic's speakers grouped by the point they made, biggest group
    # first. Each carries its own anchors, so the links still add up to the
    # count. Empty when no speaker carried a subtopic (every file written
    # before this existed), and the recap then lists the speakers flat.
    subtopics: list["TopicCount"] = field(default_factory=list)


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
                "required": ["start_seconds", "item_number", "topic_label",
                             "subtopic"],
                "properties": {
                    "start_seconds": {"type": "number"},
                    "item_number": {"type": "string"},
                    "topic_label": {"type": "string"},
                    "subtopic": {"type": "string"},
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
- subtopic: ALWAYS fill this in, for every speaker. The particular point this \
  speaker made, in 2-6 words, e.g. "phase in by grade" or "keep Oldtown Road \
  together". Say what they asked for or objected to, not how strongly.

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
- The same goes for subtopic: two speakers asking for the same thing must get \
  the SAME subtopic, character for character. Prefer the smallest number of \
  distinct subtopics that covers the comments. A subtopic used by one speaker \
  is fine when that speaker really is making their own point.
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

    text = llmcache.text_call(
        client,
        action="public-comment speakers",
        model=model,
        max_tokens=8192,
        # Explicitly off: Sonnet 5 thinks by default when the field is
        # omitted, and thinking would share this budget with the speaker
        # JSON. See the _THINKING note in score.py.
        thinking={"type": "disabled"},
        output_config={
            "effort": "medium",
            "format": {"type": "json_schema", "schema": _SCHEMA},
        },
        system=_SYSTEM,
        messages=[{"role": "user", "content": user_msg}],
    )

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
            subtopic=(entry.get("subtopic") or "").strip(),
        ))
    speakers.sort(key=lambda s: s.start_seconds)
    return speakers


# --- Deterministic tally ---------------------------------------------------

# Words dropped before counting. Maintainer-owned, in the way `rubric.md` is:
# a person edits this list, and every word in it is either an English function
# word, caption filler ("um", "you know"), or the procedural furniture of the
# podium — the chair's "thank you, please state your name and address" lands
# inside the first seconds of every speaker's span, because an auto-caption
# track has no speaker labels to cut it out with. Without the last group the
# top of every list reads "thank (14)".
#
# Nothing here is a topic word. Adding one WOULD be a judgment about what the
# comments were about, which is exactly what this feature exists to avoid — the
# counts are supposed to show what the speakers said, not what we think they
# meant.
_STOPWORDS = set("""
a an the and or but if then than that this these those there here
is are was were be been being am it its it's
i i'm i've me my mine we we're our ours us you you're your yours
they them their theirs he she his her him hers
of in on at to for from with without by as about into over under
through between during before after again further once
do does did doing done have has had having
will would can could should shall may might must
not no nor only own same so too very just also even still yet
what which who whom whose when where why how
all any both each few more most other some such
one two three four five six seven eight nine ten
let's that's there's here's don't doesn't didn't can't won't wasn't
isn't aren't we'll we've they're you'll gonna
um uh yeah yep okay ok like know think going want really
thing things kind lot got get go say said says come came take took
make made look looking see put give given ask asked tell told
because already always never much many well first next last back ago
thank thanks please good evening night morning afternoon
name address ms mr mrs dr madam chair chairman sir maam
""".split())

# A word has to be said at least this many times before the recap will show it.
# One mention is not "a word they used a lot", and at 1 the list fills with
# caption garble. Applied when RENDERING, not when counting: the saved tally
# keeps every word, so the file stays a full audit record and merging two
# meetings' counts cannot drop a word that was under the bar in each of them
# and over it across both.
MIN_WORD_COUNT = 2

_WORD_RE = re.compile(r"[A-Za-z][A-Za-z']*")


def _count_words(text: str) -> tuple[dict[str, int], dict[str, str]]:
    """Count words in a stretch of captions. Returns (counts, display form).

    Counting is case-insensitive; the display form is the spelling that
    appeared most often, so a proper noun the captions always capitalize comes
    back as "Lafayette" while a sentence-initial "School" stays lowercase. No
    stemming: "school" and "schools" are counted apart, because a reader
    checking a count against the video can only check the word that was said.
    """
    counts: dict[str, int] = {}
    forms: dict[str, dict[str, int]] = {}
    for word in _WORD_RE.findall(text):
        key = word.casefold()
        if len(key) < 3 or key in _STOPWORDS:
            continue
        counts[key] = counts.get(key, 0) + 1
        seen = forms.setdefault(key, {})
        seen[word] = seen.get(word, 0) + 1
    # Ties go to the lowercase spelling: a word that is capitalized exactly as
    # often as not is a sentence start, not a name.
    display = {
        key: max(seen, key=lambda w: (seen[w], w[:1].islower()))
        for key, seen in forms.items()
    }
    return counts, display


def _span_text(snippets: list[dict], start: float, end: float) -> str:
    """The caption text between two timestamps."""
    return " ".join(s["text"] for s in snippets if start <= s["start"] < end)


def _merge_words(into: dict[str, int], add: dict[str, int]) -> None:
    for word, n in add.items():
        into[word] = into.get(word, 0) + n


def _ranked_words(counts: dict[str, int], display: dict[str, str]) -> dict[str, int]:
    """Counts as an ordered {word: times said}, most-said first, ties A-Z."""
    return {
        display.get(key, key): n
        for key, n in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    }


def group_subtopics(speakers: list[Speaker]) -> list[TopicCount]:
    """Group one topic's speakers by the point each of them made.

    Arithmetic, not judgment: the labels come from the model one speaker at a
    time (the same measuring-instrument job as placing them in the video), and
    everything here is counting identical strings. Ranked by size, ties broken
    by who spoke first, so the order is the meeting's rather than the
    alphabet's.

    Groups of one are kept. Six residents spoke to the August redistricting
    item and asked for six different things ("keep Oldtown Road at Lafayette",
    "phase in by grade to reduce disruption"), and six labelled lines say far
    more than six bare timestamps. What the recap will not do is print a
    PARTIAL grouping: one unlabelled speaker and the counts stop adding up, so
    an empty list falls the whole topic back to the flat list.
    """
    groups: dict[str, list[Speaker]] = {}
    labels: dict[str, str] = {}          # casefolded -> label as first written
    for sp in speakers:
        if not sp.subtopic:
            return []                    # a partial grouping would miscount
        key = sp.subtopic.casefold()
        groups.setdefault(key, []).append(sp)
        labels.setdefault(key, sp.subtopic)
    return sorted(
        (TopicCount(
            label=labels[key],
            count=len(group),
            anchors=[SpeakerAnchor(start_seconds=sp.start_seconds,
                                   meeting=sp.meeting) for sp in group],
        ) for key, group in groups.items()),
        key=lambda t: (-t.count, t.anchors[0].start_seconds),
    )


def tally(
    speakers: list[Speaker],
    items: list[AgendaItem],
    pc_ranges: list[dict] | None = None,
    snippets: list[dict] | None = None,
) -> PublicCommentTally:
    """Count speakers per topic and estimate minutes per item. No model.

    A speaker's share of the period runs from their start to the next
    speaker's (the last speaker runs to the end of the public-comment period),
    which is what `public_comment_minutes` is derived from — the same ranking
    signal as before, attributed per speaker instead of per time range.

    Given `snippets` — the meeting's caption track — each topic also carries a
    count of the words its speakers used, over exactly those same spans. That
    is the recap's answer to "what did they say?": arithmetic over the
    captions, checkable against the video word by word, in place of a quote
    somebody had to choose.
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
    anchors: dict[str, list[SpeakerAnchor]] = {}
    words: dict[str, dict[str, int]] = {}    # topic key -> casefolded counts
    forms: dict[str, str] = {}               # casefolded word -> spelling shown
    subs: dict[str, list[Speaker]] = {}      # topic key -> its speakers, in order

    for n, sp in enumerate(speakers):
        nxt = speakers[n + 1].start_seconds if n + 1 < len(speakers) else period_end
        span = max(0.0, nxt - sp.start_seconds) / 60.0
        anchor = SpeakerAnchor(start_seconds=sp.start_seconds, meeting=sp.meeting)
        if snippets is not None:
            said, spelled = _count_words(
                _span_text(snippets, sp.start_seconds, nxt)
            )
            key = sp.item_number or sp.topic_label.casefold()
            if key:
                _merge_words(words.setdefault(key, {}), said)
                forms.update(spelled)
        subs.setdefault(sp.item_number or sp.topic_label.casefold(), []).append(sp)
        if sp.item_number:
            counts[sp.item_number] = counts.get(sp.item_number, 0) + 1
            minutes[sp.item_number] = minutes.get(sp.item_number, 0.0) + span
            anchors.setdefault(sp.item_number, []).append(anchor)
        elif sp.topic_label:
            key = sp.topic_label.casefold()
            off_counts[key] = off_counts.get(key, 0) + 1
            off_labels.setdefault(key, sp.topic_label)
            anchors.setdefault(key, []).append(anchor)

    result.speakers_by_item = counts
    result.minutes_by_item = {k: round(v, 1) for k, v in minutes.items()}
    # Ranked by speaker count, ties broken by agenda order (off-agenda: A-Z).
    result.by_item = sorted(
        (TopicCount(label=titles.get(num, num), count=c, item_number=num,
                    anchors=anchors.get(num, []),
                    words=_ranked_words(words.get(num, {}), forms),
                    subtopics=group_subtopics(subs.get(num, [])))
         for num, c in counts.items()),
        key=lambda t: (-t.count, order.get(t.item_number, 999)),
    )
    result.off_agenda = sorted(
        (TopicCount(label=off_labels[k], count=c, anchors=anchors.get(k, []),
                    words=_ranked_words(words.get(k, {}), forms),
                    subtopics=group_subtopics(subs.get(k, [])))
         for k, c in off_counts.items()),
        key=lambda t: (-t.count, t.label.casefold()),
    )
    return result


def extend_to_next_item(
    pc_ranges: list[dict],
    item_starts: list[float],
    *,
    speaker_seconds: float = 120.0,
    verbose: bool = True,
) -> list[dict]:
    """Widen the public-comment window up to the first agenda item that follows.

    The segmentation prompt asks for tight ranges, and on Aug 18 it closed the
    public-comment window at 2067s — TWO SECONDS before the eleventh and final
    speaker began at 2069s. He never reached `classify_speakers`, so the recap
    published "10 residents spoke" when eleven had. Nothing errored; the count
    was simply short by one, which is the one kind of mistake this newsletter
    cannot make.

    The board does not transact business between the last speaker and the first
    agenda item, so any daylight there belongs to public comment. Widening the
    window to the next mapped item start reclaims it, deterministically. The
    guard fires only when the gap could hold a speaker — the board allows two
    minutes each, so a gap under that is just the chair moving on.

    `item_starts` are the mapped start times of the agenda items in the SAME
    video (`meeting_start_seconds`, or the work-session equivalent).

    Checked against the seven archived meetings: it fires on Aug 18 alone. The
    other six leave 31-91 seconds between the window and the next item — one to
    three segmentation windows, the chair moving on — and the largest of those,
    June 16's 91s, sits well under the threshold.

    NOTE, it moves a SCORED signal. Widening the window moves `period_end`,
    which lengthens the last speaker's span in `tally()`, which feeds
    `public_comment_minutes` at 0.20 of the composite. The effect is tiny (the
    last speaker only ever gains the tail of their own turn) but it is not
    zero, so re-running an archived recap after this change will not reproduce
    its saved composite to the last decimal.
    """
    if not pc_ranges:
        return pc_ranges
    end = max(r["end_seconds"] for r in pc_ranges)
    later = [t for t in item_starts if t is not None and t > end]
    if not later:
        return pc_ranges
    nxt = min(later)
    if nxt - end < speaker_seconds:
        return pc_ranges
    if verbose:
        print(f"  ! public comment closed at {end:.0f}s but the next agenda item "
              f"starts at {nxt:.0f}s — a {nxt - end:.0f}s gap, long enough for a "
              f"speaker. Widening the window.")
    widened = [dict(r) for r in pc_ranges]
    last = max(widened, key=lambda r: r["end_seconds"])
    last["end_seconds"] = nxt
    return widened


def attach_off_agenda(
    speakers: list[Speaker], items: list[AgendaItem], *, verbose: bool = True
) -> list[Speaker]:
    """Move a speaker onto an agenda item when their topic clearly names one.

    Claude is asked whether a comment is "on the agenda", and it answers about
    the agenda in front of it. That is the right answer for a single meeting and
    the wrong one for a month: six residents spoke about redistricting at the
    Aug 18 meeting, where no redistricting item was on the agenda — it had been
    the Aug 4 work session's information item, which is exactly the connection a
    monthly recap exists to make.

    So the connection is drawn deterministically instead, by the same
    content-word matcher `carryforward.py` uses to project prior public comment
    onto agenda titles, at the same strict threshold. "school redistricting
    concerns" matches "Redistricting Process Update" at 1.00; "renaming James
    Blair school" scores 0.00 against it and stays off-agenda. Every promotion
    prints itself, because this silently moves counts between sections.
    """
    titles = [(i.number, i.title) for i in items if i.number]
    if not titles:
        return speakers

    keys = {
        number: _content_words(_LEAD_VERB_RE.sub("", title))
        for number, title in titles
    }
    resolved: dict[str, tuple[str, float]] = {}   # topic label -> (number, score)
    out: list[Speaker] = []
    for sp in speakers:
        if sp.item_number or not sp.topic_label:
            out.append(sp)
            continue
        label = sp.topic_label
        if label not in resolved:
            words = _content_words(label)
            best, best_score = "", _TITLE_MATCH_MIN
            for number, _ in titles:
                if (score := _overlap(words, keys[number])) > best_score:
                    best, best_score = number, score
            resolved[label] = (best, best_score)
            if best and verbose:
                title = dict(titles)[best]
                print(f'  attach: "{label}" -> [{best}] {title[:46]} '
                      f'({best_score:.2f})')
        number, _ = resolved[label]
        # The label that got them promoted is the point they made, so it
        # becomes their subtopic rather than being dropped — a speaker moved
        # onto the redistricting item for saying "school redistricting
        # concerns" is grouped under that, not left unlabelled.
        out.append(
            dataclasses.replace(
                sp, item_number=number, topic_label="",
                subtopic=sp.subtopic or label,
            )
            if number else sp
        )
    return out


def merge_tallies(
    tallies: list[PublicCommentTally], items: list[AgendaItem]
) -> PublicCommentTally:
    """Combine one tally per meeting into one tally for the period.

    Counted, never re-derived: the per-item and per-topic figures are summed
    from tallies each already computed against its own meeting's transcript.
    Merging the raw speaker lists instead would be wrong — a speaker's minutes
    run to the NEXT speaker's start, and across a two-week gap that span is the
    gap, not a turn at the podium.
    """
    if len(tallies) == 1:
        return tallies[0]

    merged = PublicCommentTally()
    off_counts: dict[str, int] = {}
    off_labels: dict[str, str] = {}
    anchors: dict[str, list[SpeakerAnchor]] = {}
    # Already in display spelling and already ranked within each meeting; summed
    # here and re-ranked below, so a word said at both meetings carries one
    # total. Spelling ties across meetings resolve to whichever came first.
    words: dict[str, dict[str, int]] = {}
    subs: dict[str, list[TopicCount]] = {}   # topic key -> groups from each meeting
    for t in tallies:
        merged.total_speakers += t.total_speakers
        for num, c in t.speakers_by_item.items():
            merged.speakers_by_item[num] = merged.speakers_by_item.get(num, 0) + c
        for num, m in t.minutes_by_item.items():
            merged.minutes_by_item[num] = round(
                merged.minutes_by_item.get(num, 0.0) + m, 1
            )
        for topic in t.by_item:
            anchors.setdefault(topic.item_number, []).extend(topic.anchors)
            _merge_words(words.setdefault(topic.item_number, {}), topic.words)
            subs.setdefault(topic.item_number, []).extend(topic.subtopics)
        for topic in t.off_agenda:
            key = topic.label.casefold()
            off_counts[key] = off_counts.get(key, 0) + topic.count
            off_labels.setdefault(key, topic.label)
            anchors.setdefault(key, []).extend(topic.anchors)
            _merge_words(words.setdefault(key, {}), topic.words)
            subs.setdefault(key, []).extend(topic.subtopics)

    titles = {i.number: i.title for i in items if i.number}
    order = {i.number: n for n, i in enumerate(items) if i.number}
    def ranked(key: str) -> dict[str, int]:
        return _ranked_words(words.get(key, {}), {})

    def regrouped(key: str) -> list[TopicCount]:
        """One topic's subtopic groups, summed across the meetings it spans."""
        groups: dict[str, TopicCount] = {}
        for group in subs.get(key, []):
            got = groups.get(group.label.casefold())
            if got is None:
                groups[group.label.casefold()] = dataclasses.replace(
                    group, anchors=list(group.anchors)
                )
                continue
            got.count += group.count
            got.anchors.extend(group.anchors)
        return sorted(
            groups.values(),
            key=lambda t: (-t.count,
                           t.anchors[0].start_seconds if t.anchors else 0.0),
        )

    merged.by_item = sorted(
        (TopicCount(label=titles.get(num, num), count=c, item_number=num,
                    anchors=anchors.get(num, []), words=ranked(num),
                    subtopics=regrouped(num))
         for num, c in merged.speakers_by_item.items()),
        key=lambda t: (-t.count, order.get(t.item_number, 999)),
    )
    merged.off_agenda = sorted(
        (TopicCount(label=off_labels[k], count=c, anchors=anchors.get(k, []),
                    words=ranked(k), subtopics=regrouped(k))
         for k, c in off_counts.items()),
        key=lambda t: (-t.count, t.label.casefold()),
    )
    return merged


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


def _topic_from_dict(raw: dict) -> TopicCount:
    """One TopicCount from saved JSON. `anchors` is absent in files written
    before per-speaker links existed, so it defaults to empty and those recaps
    re-render exactly as they did."""
    data = dict(raw)
    data["anchors"] = [SpeakerAnchor(**a) for a in data.get("anchors") or []]
    data["subtopics"] = [_topic_from_dict(t) for t in data.get("subtopics") or []]
    return TopicCount(**data)


def from_json(data: dict) -> tuple[list[Speaker], PublicCommentTally, list[dict]]:
    """Rebuild what `to_json` wrote (used by `render.py --recap`)."""
    speakers = [Speaker(**s) for s in data.get("speakers", [])]
    raw = data.get("tally", {})
    result = PublicCommentTally(
        total_speakers=raw.get("total_speakers", 0),
        by_item=[_topic_from_dict(t) for t in raw.get("by_item", [])],
        off_agenda=[_topic_from_dict(t) for t in raw.get("off_agenda", [])],
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
