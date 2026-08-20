#!/usr/bin/env python3
"""Decide whether two pieces of agenda text name the same thing.

Deterministic, no LLM, no network. One matcher, four callers, one threshold —
so a pairing that is right in the forecast check is right in the merge too, and
tuning it is a single edit rather than four:

- `carryforward.py` projects a past meeting's public-comment topics onto an
  upcoming agenda.
- `forecast._pair_rankings` joins a forecast to the recap that followed it,
  whose item numbers are entirely different.
- `merge.py` folds a work session's "Review X" into the regular meeting's
  "Approval of X".
- `pubcomment.attach_off_agenda` moves a speaker onto the item they named.

Every one of those is the same question — same title, different wording — and
every one of them can do real damage by answering it wrong, which is why the
threshold is strict and every caller prints the pairings it makes.
"""

import re

# Agenda titles are dense with procedural scaffolding that carries no subject
# matter. Left in, "Approval of Purchase Request for X" and "Approval of
# Purchase Request for Y" look like the same topic.
_BOILERPLATE = {
    "approval", "approve", "approved", "request", "requests", "update",
    "updates", "revision", "revisions", "amendment", "amendments", "report",
    "reports", "presentation", "discussion", "consideration", "adoption",
    "adopt", "review", "proposed", "recommendation", "recommendations",
    "resolution", "action", "item", "items", "new", "old", "annual", "monthly",
    "meeting", "board", "school", "schools", "division", "wjcc", "williamsburg",
    "james", "city", "county", "public", "hearing", "first", "second", "reading",
}
_STOPWORDS = {
    "a", "an", "and", "as", "at", "be", "by", "for", "from", "in", "into", "is",
    "of", "on", "or", "the", "to", "with", "s",
    # General-English words with no topical content. The document-frequency
    # filter cannot screen these: it is built from formal agenda TITLES, where
    # plain verbs and adjectives are rare, so it scores them as distinctive. But
    # public-comment topic labels are written in ordinary prose and use them
    # freely. On the 20260818 agenda "use" (title df = 1) matched "Review
    # Authorization to Permit City/County Use of School Buses" against
    # "classroom technology use", projecting 3 speakers onto an unrelated item.
    "use", "used", "using", "new", "other", "more", "need", "needs", "issue",
    "issues", "concern", "concerns", "impact", "impacts", "general", "public",
}
_WORD_RE = re.compile(r"[a-z0-9]+")

# Leading verbs that name what a meeting is DOING to an item rather than which
# item it is: the same decision reads "Review X" at the work session and
# "Approval of X" at the regular meeting. Stripping the verb is what lets the
# two be recognised as one thing — by `forecast._pair_rankings` when scoring a
# forecast, and by `merge.py` when folding a review into the vote it became.
_LEAD_VERB_RE = re.compile(
    r"^\s*(?:review(?:\s+of)?|approval\s+of(?:\s+the)?|approve|adoption\s+of(?:\s+the)?|"
    r"consideration\s+of(?:\s+the)?)\s+",
    re.IGNORECASE,
)
# Below this, two titles are different items. Chosen to be strict: a false pair
# would corrupt the one number that establishes whether the forecast works.
_TITLE_MATCH_MIN = 0.6


def _content_words(text: str) -> set[str]:
    """Reduce a title or topic label to its distinguishing words."""
    words = _WORD_RE.findall(text.lower())
    return {w for w in words if w not in _STOPWORDS and w not in _BOILERPLATE}


def _overlap(a: set[str], b: set[str]) -> float:
    """Jaccard overlap, but forgiving of one side being much shorter.

    A prior off-agenda label is 2-5 words ("school redistricting") while an
    agenda title can run 15. Plain Jaccard would score that pair near zero, so
    a full containment of the shorter set counts as a match.
    """
    if not a or not b:
        return 0.0
    inter = len(a & b)
    if not inter:
        return 0.0
    if inter == min(len(a), len(b)):
        return 1.0
    return inter / len(a | b)
