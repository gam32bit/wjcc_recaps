#!/usr/bin/env python3
"""Project public-comment turnout from past meetings onto an upcoming agenda.

Before a meeting, `public_comment_minutes` is structurally unknowable — the
comment has not happened yet — and it carries 0.20 of the recap composite. The
20260616 recap made the cost concrete: `9.03 Redistricting Process Update` drew
only 3.7 minutes of work-session discussion (forecast rank 15) and became the
recap's #1 on 14 resident speakers.

Residents who organize around a topic one month tend to return the next, and
that turnout is already recorded deterministically in
`out/recap-pubcomment-<date>.json`. This module reads those saved tallies and
carries them forward as `Signals.carry_forward_speakers`.

All of it is plain Python — no LLM. Matching is on topic TEXT, never on item
number: numbering is reused between meetings (June's 9.03 is redistricting;
February's 9.03 was a purchase request), so matching by number would be wrong
in a way that is hard to spot.

Usage:
    python carryforward.py 20260616          # show what would be carried forward
"""

import argparse
import json
import pathlib
import re
import sys
from dataclasses import dataclass, field

from parse import AgendaItem, agenda_preamble, parse_agenda
from titlematch import (
    _BOILERPLATE,
    _LEAD_VERB_RE,
    _STOPWORDS,
    _TITLE_MATCH_MIN,
    _WORD_RE,
    _content_words,
    _overlap,
)
from pubcomment import from_json
from triage import kept_items, triage

PROJECT_DIR = pathlib.Path(__file__).resolve().parent
OUT_DIR = PROJECT_DIR / "out"
FIXTURES_DIR = PROJECT_DIR / "wjcc-fixtures"

# How many prior meetings to look back over, and what each is worth. A topic
# that filled the room last month is a much better predictor than one that did
# so three months ago.
DECAY = (1.0, 0.6, 0.3)

# Minimum content-word overlap (Jaccard) for a prior topic to be considered the
# same subject as an agenda item.
MATCH_THRESHOLD = 0.34

# Overlap alone is too strict for how residents actually name things. "redistricting
# neighborhood concerns" vs "Redistricting Process Update" shares one word out of
# four and scores 0.25 — yet it is plainly the same subject, and getting it wrong is
# what this module exists to prevent. So a single SHARED DISTINCTIVE WORD also counts
# as a match.
#
# Distinctiveness is measured as document frequency across EVERY fixture agenda, not
# just the one being forecast. Rarity within a single agenda is not rarity: on the
# 20260616 agenda "program" and "middle" each appear once and look distinctive, and
# matching on them wrongly tied "lacrosse program funding" to a federal grant item and
# "James Blair Middle School renaming" to a furniture purchase. Across all 223 items
# the separation is clean — redistricting 2, vhsl 1, technology 1, versus program 4,
# middle 4, staff 7, student 18, policy 54 — and it needs no hand-kept keyword list.
DISTINCTIVE_DF = 2
_DF_CACHE = PROJECT_DIR / ".cache" / "title-df.json"

_TALLY_RE = re.compile(r"recap-pubcomment-(\d{8})\.json$")



@dataclass
class MeetingTally:
    """One past meeting's saved public-comment tally, reduced to (topic, count)."""
    date: str                                  # numberdate, e.g. "20260519"
    topics: list[tuple[str, int]] = field(default_factory=list)


@dataclass
class CarryForward:
    """Prior turnout projected onto one agenda item, with its provenance.

    `sources` is the audit trail — every (date, matched label, speaker count)
    that fed `weighted`. Fuzzy text matching is the one place this design could
    silently inject weight into the ranking, so the forecast prints all of it.
    """
    weighted: float = 0.0
    sources: list["CarryForwardSource"] = field(default_factory=list)

    @property
    def speakers(self) -> int:
        """Raw head count behind this signal, undecayed."""
        return sum(s.count for s in self.sources)

    @property
    def note(self) -> str:
        """Compact provenance for the evidence line, e.g. '5 on 20260519'."""
        by_date: dict[str, int] = {}
        for s in self.sources:
            by_date[s.date] = by_date.get(s.date, 0) + s.count
        return ", ".join(f"{c} on {d}" for d, c in sorted(by_date.items(), reverse=True))


@dataclass
class CarryForwardSource:
    """One prior topic that fed a CarryForward, and why it was considered a match."""
    date: str
    label: str
    count: int
    why: str = ""



def _build_title_df(fixtures_dir: pathlib.Path) -> dict[str, int]:
    """Count how many agenda items across all fixtures use each content word."""
    # Imported here so the common cached path costs nothing.
    from parse import agenda_preamble as _preamble, parse_agenda as _parse
    from triage import kept_items as _kept, triage as _triage

    df: dict[str, int] = {}
    n = 0
    for html_path in sorted(fixtures_dir.glob("agenda-2*.html")):
        meta_path = fixtures_dir / f"meeting-meta-{html_path.stem.split('-')[1]}.json"
        if not meta_path.is_file():
            continue
        html = html_path.read_text()
        try:
            items = _kept(_triage(_parse(html), json.loads(meta_path.read_text()),
                                 _preamble(html)))
        except Exception:                      # a malformed fixture must not break scoring
            continue
        for item in items:
            n += 1
            for w in _content_words(item.title):
                df[w] = df.get(w, 0) + 1
    df["__items__"] = n
    return df


def title_df(fixtures_dir: pathlib.Path = FIXTURES_DIR) -> dict[str, int]:
    """Corpus document frequencies, cached — parsing 17 agendas is not free."""
    n_fixtures = len(list(fixtures_dir.glob("agenda-2*.html")))
    if _DF_CACHE.is_file():
        try:
            cached = json.loads(_DF_CACHE.read_text())
            if cached.get("__fixtures__") == n_fixtures:
                return cached
        except json.JSONDecodeError:
            pass
    df = _build_title_df(fixtures_dir)
    df["__fixtures__"] = n_fixtures
    _DF_CACHE.parent.mkdir(parents=True, exist_ok=True)
    _DF_CACHE.write_text(json.dumps(df, indent=2, sort_keys=True))
    return df


def _distinctive(word: str, df: dict[str, int]) -> bool:
    """Is this word rare enough across all agendas to identify a subject alone?"""
    return not word.isdigit() and df.get(word, 0) <= DISTINCTIVE_DF


def _is_match(
    topic_words: set[str], title_words: set[str], df: dict[str, int]
) -> tuple[bool, str]:
    """Does this prior topic refer to this agenda item? Returns (matched, why)."""
    shared = sorted(w for w in topic_words & title_words if _distinctive(w, df))
    if shared:
        return True, f"shared '{shared[0]}'"
    score = _overlap(topic_words, title_words)
    if score >= MATCH_THRESHOLD:
        return True, f"overlap {score:.2f}"
    return False, ""


def load_tallies(out_dir: pathlib.Path = OUT_DIR) -> list[MeetingTally]:
    """Read every saved public-comment tally, newest first."""
    tallies: list[MeetingTally] = []
    for path in sorted(out_dir.glob("recap-pubcomment-*.json"), reverse=True):
        m = _TALLY_RE.search(path.name)
        if not m:
            continue
        try:
            _, result, _ = from_json(json.loads(path.read_text()))
        except (json.JSONDecodeError, TypeError) as exc:
            print(f"  WARNING: skipping unreadable tally {path.name}: {exc}", file=sys.stderr)
            continue
        topics = [(t.label, t.count) for t in result.by_item + result.off_agenda if t.label]
        tallies.append(MeetingTally(date=m.group(1), topics=topics))
    return tallies


def match(
    items: list[AgendaItem],
    tallies: list[MeetingTally],
    target_date: str,
    *,
    lookback: int = len(DECAY),
) -> dict[str, CarryForward]:
    """Map agenda item number -> CarryForward, from meetings before target_date.

    Only tallies STRICTLY BEFORE `target_date` are used, and the decay slots are
    assigned by proximity to it. Without that filter a backtest of an old date
    would score using turnout from meetings that had not happened yet — inflating
    the very number meant to establish whether the forecast can be trusted.
    """
    prior = sorted(
        (t for t in tallies if t.date < target_date),
        key=lambda t: t.date,
        reverse=True,
    )[:lookback]

    item_words = {i.number: _content_words(i.title) for i in items if i.number}
    df = title_df()
    out: dict[str, CarryForward] = {}

    for slot, tally in enumerate(prior):
        weight = DECAY[slot] if slot < len(DECAY) else DECAY[-1]
        for label, count in tally.topics:
            words = _content_words(label)
            # Best match wins, so one topic never inflates two items.
            best: tuple[float, str, str] = (0.0, "", "")
            for number, title_words in item_words.items():
                matched, why = _is_match(words, title_words, df)
                if matched:
                    score = _overlap(words, title_words)
                    if score > best[0] or not best[1]:
                        best = (score, number, why)
            if not best[1]:
                continue
            cf = out.setdefault(best[1], CarryForward())
            cf.weighted += weight * count
            cf.sources.append(CarryForwardSource(tally.date, label, count, best[2]))

    return out


def unmatched_topics(
    items: list[AgendaItem],
    tallies: list[MeetingTally],
    target_date: str,
    *,
    lookback: int = len(DECAY),
) -> list[tuple[str, str, int]]:
    """Prior topics that matched NO item on this agenda, newest meeting first.

    These are the likely sources of off-agenda public comment: a subject
    residents keep raising that the board has not put on the agenda.
    """
    prior = sorted(
        (t for t in tallies if t.date < target_date),
        key=lambda t: t.date,
        reverse=True,
    )[:lookback]
    item_words = {i.number: _content_words(i.title) for i in items if i.number}
    df = title_df()

    leftover: list[tuple[str, str, int]] = []
    for tally in prior:
        for label, count in tally.topics:
            words = _content_words(label)
            if not any(
                _is_match(words, tw, df)[0] for tw in item_words.values()
            ):
                leftover.append((tally.date, label, count))
    return leftover


def apply_to_signals(scored: list, matches: dict[str, CarryForward]) -> None:
    """Write the carry-forward signal onto ScoredItems, in place."""
    for s in scored:
        cf = matches.get(s.item.number or "")
        if cf:
            s.signals.carry_forward_speakers = round(cf.weighted, 1)
            s.signals.carry_forward_note = cf.note


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("date", help="target meeting numberdate, e.g. 20260616")
    args = parser.parse_args()

    html_path = FIXTURES_DIR / f"agenda-{args.date}.html"
    meta_path = FIXTURES_DIR / f"meeting-meta-{args.date}.json"
    if not html_path.is_file():
        sys.exit(f"No agenda fixture for {args.date}.")

    html = html_path.read_text()
    items = kept_items(triage(parse_agenda(html),
                              json.loads(meta_path.read_text()),
                              agenda_preamble(html)))

    tallies = load_tallies()
    print(f"{len(tallies)} saved tallies; using those before {args.date}.")
    matches = match(items, tallies, args.date)
    if not matches:
        print("No carry-forward matches.")
    titles = {i.number: i.title for i in items}
    for number, cf in sorted(matches.items(), key=lambda kv: -kv[1].weighted):
        print(f"\n  [{number}] {titles.get(number, '')[:55]}   weighted={cf.weighted:.1f}")
        for src in cf.sources:
            print(f"      {src.count:>3} speakers on {src.date}: "
                  f"{src.label[:44]}  ({src.why})")

    leftover = unmatched_topics(items, tallies, args.date)
    if leftover:
        print("\nPrior topics matching nothing on this agenda (likely off-agenda comment):")
        for date, label, count in leftover:
            print(f"  {count:>3} on {date}: {label[:55]}")


if __name__ == "__main__":
    main()
