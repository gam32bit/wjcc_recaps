# The Rundown — WJCC School Board recaps

A Python pipeline that turns Williamsburg-James City County School Board
meetings into a plain-English recap newsletter. It fetches the month's agenda
packets, structures them, ranks what mattered using measurable signals, and
assembles a post that a human reviews and publishes to Substack.

**Nothing in the recap is written by a language model.** Each highlight is the
agenda item's own title plus its verbatim `BACKGROUND:` text as an attributed
block quote; public comment is a counted tally of speakers, each one a bare
timestamp link into the video; vote tallies are counts read off the meeting's
roll call. No line claims how long anything took. The model's only
reader-facing words are the few that name an off-agenda topic somebody raised
— "special education inclusion" over the timestamps of the people who raised
it — and a reader who doubts any of it, label included, can check it against
the source in seconds. That is the whole design goal. See
[CLAUDE.md](CLAUDE.md) for the reasoning.

## Running it

Requires Python 3.11+, `poppler-utils` (`pdftotext`, `pdfinfo`), and
`ghostscript` (`gs`). Put an `ANTHROPIC_API_KEY` in a local `.env`.

```bash
# Recap a whole month — every meeting in it, as one newsletter.
python make_newsletter.py --period 202608 \
    --video 20260804=https://www.youtube.com/watch?v=... \
    --video 20260818=https://www.youtube.com/watch?v=...

# Deterministic signals only: no API calls, no cost.
python make_newsletter.py --period 202608 --dry-run
```

Output lands in `out/` (gitignored). Agenda fixtures live in `wjcc-fixtures/`;
fetched transcripts, packet PDFs, and Claude's answers are cached in `.cache/`.

Before publishing, read `out/pubcomment-review-<date>.md`. It prints every
public-comment speaker's label above the verbatim captions it was assigned
from, over exactly the span the tally counts — the check on both the model's
topic labels and the speaker count, which is the recap's central claim.

Every Claude call is cached by a hash of the whole request, so re-running a
recap on unchanged inputs makes no API calls and costs nothing. Editing a
prompt, bumping the model, or pointing at a different video changes the hash and
re-runs that call, so the cache cannot serve a stale answer. Pass
`--no-llm-cache` to any entry point to force a full re-run.

## What the modules do

| | |
|---|---|
| `parse.py` / `pdfagenda.py` | BoardDocs HTML and Diligent packet PDF → `AgendaItem` |
| `merge.py` | combine a month's meetings; fold each work-session review into the vote it became |
| `triage.py` | drop procedural boilerplate |
| `score.py` | rank by measurable signals (discussion minutes, dollars, turnout, rubric) |
| `pubcomment.py` | locate and count public-comment speakers |
| `votes.py` | recover roll-call vote tallies from the meeting video |
| `pdfslice.py` | cut one linkable PDF per agenda item out of the packet |
| `render.py` | assemble the Markdown recap |
| `llmcache.py` | cache every Claude answer on disk, keyed by the request that produced it |
| `forecast.py` | rank a meeting's agenda *before* it happens, and score the prediction after |

## `docs/` and GitHub Pages

Diligent publishes each meeting as a single packet PDF — 1,731 pages and 48 MB
for Aug 4 2026 — behind an opaque per-document token, with no way to link a
particular page. `pdfslice.py` cuts each item's own pages out of the packet into
`docs/packet/<date>/item-<number>.pdf` (1-2 pages, 60-80 KB), and the recap
links a trimmed excerpt to its slice so a reader can see exactly what was left
out. The slice is the district's page unaltered — nothing is re-typed or
re-set.

This needs **GitHub Pages serving from the `docs/` folder on `main`**
(Settings → Pages → Source: *Deploy from a branch*, Branch: `main`, Folder:
`/docs`). The base URL is the `PACKET_BASE_URL` constant in `render.py`; if the
repo moves, change it there. Blank it out and the recap simply omits the links
rather than printing dead ones.

The agenda packets themselves are **not** committed — they are 20-45 MB each and
everything downstream reads from the parsed JSON fixture instead.
