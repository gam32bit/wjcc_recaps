# WJCC School Board Newsletter ("The Rundown")

Turns a Williamsburg-James City County School Board meeting into a plain-English
recap newsletter. A Python pipeline fetches the agenda from BoardDocs, structures
it, ranks what matters, and assembles the post; a human reviews and publishes to
Substack. (`--preview` still produces the older upcoming-meeting product.)

## Design philosophy — read this first

**Minimize reliance on the LLM's own judgment. Decisions run on measurable
signals, not model intuition.** This is a real community newsletter — its picks
have to be honest and defensible to readers, not "our AI thinks these matter."

- **Be deterministic wherever a computer can be right.** Parsing, dropping
  boilerplate, extracting meeting logistics, computing importance signals — all
  plain Python. No LLM, no drift, no hallucination.
- **Don't ask the LLM to judge what matters.** Which agenda items are important
  is decided by measurable, inspectable signals (how long the board discussed
  an item at its work session, dollar amounts, vote vs. report, etc.) combined
  with transparent, tunable weights — never by asking the model "what's
  important here?"
- **When the LLM is used, confine it to narrow, verifiable roles:**
  - a *measuring instrument* for mechanical tasks (e.g. segmenting a meeting
    transcript by topic) — output is spot-checkable against the source;
  - applying an *explicit, human-owned rubric* — the criteria live in a file
    the maintainer edits, never hidden inside the model.
- **Don't let the LLM narrate.** The recap contains no model-written prose at
  all: each highlight is the agenda item's own title plus its verbatim
  `BACKGROUND:` text as an attributed block quote, and public comment is a
  counted tally of speakers per topic rather than a summary. Summaries read
  fluently and cost more to fact-check than they save; a quote the reader can
  check against BoardDocs costs nothing to trust. (Prose survives only in the
  `--preview` product, in write.py.)
- **Keep un-verifiable facts away from the LLM entirely.** Meeting dates,
  times, locations, dollar figures, and URLs are extracted deterministically
  into a `Logistics` record and never sent to the model.
- **Optimize for verifiability.** A reviewer should be able to check the draft
  faster than they could have written it — hence quotes-with-sources, signals
  shown as visible evidence, and per-speaker anchors written into
  `out/transcript-meeting-<date>.md` so every counted speaker is seekable in
  the video.

When adding a feature, ask first: can this be a deterministic signal instead of
an LLM judgment call? Prefer the signal.

## Working on this project

- **Dev logs live in `dev-logs/`** (gitignored, local-only). The latest
  `Phase N.md` records current state, decisions, and the active plan — read it
  first; the pipeline is mid-redesign.
- Model: `claude-sonnet-5` (the `MODEL` constant in `score.py`, `pubcomment.py`,
  and `write.py`; `calibrate.py` imports score.py's). Sonnet 5 runs adaptive
  thinking when the `thinking` field is omitted, and thinking shares the
  `max_tokens` budget with the response — so every call sets `thinking`
  explicitly rather than relying on the default.
- `ANTHROPIC_API_KEY` is in `.env` (gitignored), loaded automatically.
- Fixtures in `wjcc-fixtures/`; generated output in `out/`; fetched
  PDFs/transcripts cached in `.cache/` (`out/`, `.cache/`, `.env` all
  gitignored).
- Repo is local-only (no remote); dev logs are never committed.
