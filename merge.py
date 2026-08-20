#!/usr/bin/env python3
"""Merge several meetings' agendas into one period (a month) for the recap.

Deterministic, no LLM, no network.

The board decides a thing across TWO meetings: it reviews an item at the work
session, then votes on it at the regular meeting two weeks later. A recap of
either meeting alone tells half the story — the Aug 18 recap ranked two $10
easements above a $1.8M contract, while Redistricting, which drew 48 of the 61
mapped work-session minutes and six public-comment speakers, did not appear at
all because it was never on the regular meeting's agenda.

This module builds the month's agenda instead:

- **Numbers are namespaced by meeting** ("0804-8.01"). Both August meetings have
  an item numbered 7.01, and every downstream dict — segmentation, the
  public-comment tally, `off_consent` — is keyed on the number. Un-namespaced,
  one meeting's items silently overwrite the other's.

- **A review is folded into the vote it became.** "Review 2026-2027 CNS Food and
  Consumables Purchase" (Aug 4, §5) and "Approval of 2026-2027 Child Nutrition
  Services (CNS) Food and Consumables Purchase" (Aug 18, §7) are one decision,
  and listing both would double the month. The vote is kept, because it is the
  thing that happened; the review's discussion time reaches it for free, since
  segmentation runs against the merged list and matches by topic.

An item's ATTACHMENTS are the vote's, not the union of both packets' — the two
usually carry the same documents, so a union would double-list them. The one
exception is a vote that carries none: it inherits the review's, which is the
only place the board's presentation deck exists.

Only *reviews* fold. "Approval of Personnel Actions" appears on both agendas
and stays as two items, because those are two different votes on two different
sets of people a fortnight apart — the packet's own routing sentence is what
separates the two cases, and it only ever appears on a §5 review.
"""

import dataclasses
import datetime as dt
from dataclasses import dataclass, field

from titlematch import _LEAD_VERB_RE, _TITLE_MATCH_MIN, _content_words, _overlap
from parse import AgendaItem

# Item Type of a work-session review — the only kind of item that folds into a
# later vote. `pdfagenda._item_type` assigns it from the section title
# "Proposed Agenda Items".
REVIEW_TYPE = "proposed agenda item"


@dataclass
class SourceMeeting:
    """One meeting contributing to a period recap."""
    numberdate: str            # "20260804"
    kind: str                  # "work_session" | "meeting" — selects the signal
    items: list[AgendaItem]
    meta: dict
    preamble: str = ""
    video: str | None = None

    @property
    def prefix(self) -> str:
        """Namespace for this meeting's item numbers, e.g. "0804"."""
        return self.numberdate[4:]

    @property
    def date(self) -> dt.date:
        return dt.datetime.strptime(self.numberdate, "%Y%m%d").date()

    @property
    def label(self) -> str:
        """Human label for the meeting, e.g. "Aug 4 work session"."""
        kind = "work session" if self.kind == "work_session" else "regular meeting"
        return f"{self.date:%b} {self.date.day} {kind}"


def item_meeting(number: str, sources: list[SourceMeeting]) -> SourceMeeting | None:
    """Which source meeting a namespaced item number came from."""
    prefix, _, _ = number.partition("-")
    return next((s for s in sources if s.prefix == prefix), None)


def display_number(number: str) -> str:
    """Strip the meeting namespace back off for display, e.g. "7.06"."""
    _, sep, rest = number.partition("-")
    return rest if sep else number


def _title_key(title: str) -> set[str]:
    """Content words of a title, with its leading verb removed.

    "Review X" and "Approval of X" must reduce to the same key. This is the
    matcher `forecast._pair_rankings` already uses and Phase 9 verified against
    the near-identical Clara Byrd Baker / Norge easement pair.
    """
    return _content_words(_LEAD_VERB_RE.sub("", title))


def merge_meetings(
    sources: list[SourceMeeting], *, verbose: bool = True
) -> tuple[list[AgendaItem], list[dict]]:
    """Namespace and combine every meeting's items into one period agenda.

    Returns `(items, decisions)`. `decisions` records every fold with its
    matched pair and overlap score, so the merge is inspectable rather than
    implicit — the same standard `pdfagenda.build_items` holds itself to.
    """
    ordered = sorted(sources, key=lambda s: s.numberdate)

    namespaced: list[tuple[SourceMeeting, AgendaItem]] = []
    for source in ordered:
        for item in source.items:
            namespaced.append(
                (source, dataclasses.replace(item, number=f"{source.prefix}-{item.number}"))
            )

    reviews = [(s, i) for s, i in namespaced if i.item_type.lower() == REVIEW_TYPE]
    votes = [(s, i) for s, i in namespaced if i.item_type.lower() != REVIEW_TYPE]

    decisions: list[dict] = []
    folded: set[str] = set()
    claimed: set[str] = set()

    for source, review in reviews:
        words = _title_key(review.title)
        best, best_score = None, _TITLE_MATCH_MIN
        for vote_source, vote in votes:
            if vote.number in claimed:
                continue
            # A review can only become a vote at its own meeting or a later one.
            if vote_source.numberdate < source.numberdate:
                continue
            if (score := _overlap(words, _title_key(vote.title))) > best_score:
                best, best_score = (vote_source, vote), score

        if best is None:
            decisions.append({
                "number": review.number, "title": review.title,
                "folded": False,
                "reason": "reviewed, but no matching vote in this period — kept "
                          "as its own item",
            })
            if verbose:
                print(f"  keep [{review.number}] {review.title[:46]:<48} "
                      f"reviewed at {source.label}; no vote found in period")
            continue

        vote_source, vote = best
        folded.add(review.number)
        claimed.add(vote.number)

        # The vote KEEPS its own attachments — staff normally attach the same
        # documents to both the review and the vote, and where the names differ
        # (7.09's easement plat was re-issued between the two packets) the
        # vote's copy is the later one. But the FY27 Operating Fund vote (8.07)
        # carries NO attachments while its review (5.01) carries the
        # presentation deck the board actually looked at, so an empty vote
        # inherits the review's documents rather than publishing none.
        inherited = ""
        if not vote.attachments and review.attachments:
            # Carry the review's packet page with them: the documents are filed
            # under the review, not under the vote it became.
            vote.attachments = [
                dataclasses.replace(a, page=review.source_page)
                for a in review.attachments
            ]
            inherited = (
                f"; inherited {len(review.attachments)} attachment"
                f"{'s' if len(review.attachments) != 1 else ''} from the review "
                f"(the vote carried none)"
            )

        decisions.append({
            "number": review.number, "title": review.title,
            "folded": True, "into": vote.number, "into_title": vote.title,
            "overlap": round(best_score, 2),
            "reason": f"reviewed at {source.label}, voted at {vote_source.label}"
                      + inherited,
        })
        if verbose:
            print(f"  fold [{review.number}] -> [{vote.number}] ({best_score:.2f})  "
                  f"{vote.title[:44]}")

    items = [i for _, i in namespaced if i.number not in folded]
    if verbose:
        print(f"\n  {len(namespaced)} items across {len(ordered)} meetings -> "
              f"{len(items)} after folding {len(folded)} review(s) into their votes.")
    return items, decisions


def period_meta(sources: list[SourceMeeting]) -> dict:
    """Meeting metadata for the merged period.

    Logistics come from the LAST meeting: its packet lists the meetings that
    come next, which is what a reader of the recap wants, and its own date is
    the one the period ends on.
    """
    ordered = sorted(sources, key=lambda s: s.numberdate)
    meta = dict(ordered[-1].meta)
    meta["period_meetings"] = [
        {"numberdate": s.numberdate, "label": s.label, "video": s.video}
        for s in ordered
    ]
    return meta


def period_label(sources: list[SourceMeeting]) -> str:
    """Title for the period, e.g. "August 2026" or "August 4-18, 2026"."""
    ordered = sorted(sources, key=lambda s: s.numberdate)
    first, last = ordered[0].date, ordered[-1].date
    if first.year == last.year and first.month == last.month:
        return f"{first:%B %Y}"
    if first.year == last.year:
        return f"{first:%B} – {last:%B %Y}"
    return f"{first:%B %Y} – {last:%B %Y}"
