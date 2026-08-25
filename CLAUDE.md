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
  all. A highlight carries exactly three kinds of content, and a model produces
  none of them:
  1. the agenda item's own title and verbatim `BACKGROUND:` text, block-quoted
     and attributed (trimmed to two sentences, with "read more" pointing at the
     district's own page in the packet — see `pdfslice.py`);
  2. counted tallies — public-comment speakers, vote counts — each speaker
     linked to their turn in the video, so the number of links IS the count;
  3. **a quote from the meeting video, chosen by the maintainer** and recorded
     in `quotes-<period>.json`. Nothing in the pipeline picks, writes or ranks
     these; a person watches the meeting and puts the line in the file. The
     text is the video's automatic captions, so every quote renders with its
     timestamp link and says where it came from — the reader verifies it by
     listening, the same bargain the public-comment anchors make. Elisions are
     marked with an ellipsis and any other departure from the caption text is
     noted in the file, so the edit is on the record. A `speaker` is filled in
     only when the video itself establishes who is talking (a
     self-identification, or another member naming them on the record) — a
     caption SPELLING is never evidence; see the name-mangling trap below.

  Summaries read fluently and cost more to fact-check than they save; a quote
  the reader can check against the source costs nothing to trust. (Prose
  survives only in the `--preview` product, in write.py.)
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
- **Every Claude call is cached to `.cache/llm/`** by `llmcache.py`, keyed by a
  hash of the whole request (model, prompt, schema, transcript), so re-running
  the pipeline on unchanged inputs costs nothing. Edit a prompt or bump the
  model and the key changes and the call is re-made — the cache cannot go
  stale. Entries are readable JSON; delete one to force that call, or pass
  `--no-llm-cache` to re-run all of them.
- Repo is `git@github.com:gam32bit/wjcc_recaps.git` (public, branch `main`).
  Dev logs, `out/`, `.cache/`, `.env` and the packet PDFs are never committed.
- **`docs/` IS committed** — GitHub Pages serves it, and that is how the recap
  links a trimmed excerpt to the item's own page in the agenda packet. See
  `pdfslice.py` and `PACKET_BASE_URL` in `render.py`.
- Git over SSH does not work inside the tool sandbox (port 22 is not proxied,
  and `~/.ssh` is unreadable) — pushes need the sandbox disabled.

## Reading a transcript

Answering "what did the board actually discuss?" from `out/transcript-*-<date>.md`.
Timestamps there are deep links into the video when the writer was given a
`video_url=` (see `write_transcript_md`). Four traps, each of which yields a
*wrong* answer rather than a slow one:

- **The score file is not the whole meeting.** `forecast-score-<date>.json`
  holds only the items the packet routed forward, so its `Disc` column silently
  omits anything voted the same evening. At the 2026-08-04 work session, item
  5.01 (FY27 operating budget amendment) drew ~8 minutes — the night's
  second-longest discussion — and is absent from the JSON entirely, because the
  board voted it that night as 8.07. Walk the transcript's `##` headings too.
- **Segmenter anchors run early; verify boundaries against the text.** 6.01's
  saved `work_session_start_seconds` was 1691 (28:11), but the item opens at
  28:42 and the paragraph under its heading still belongs to 5.10. Corollary:
  don't rank items within the 0.5–2.6 min band against each other — a
  30-second boundary error is 100% of a one-minute item. "Only redistricting
  got real time" is robust; the ordering beneath it is not.
- **The captions mangle every name.** On 2026-08-04, Riffel appears as Riffle,
  Reiffle, Orfalea, Ruffalo and Hutchens; Chen as Chem, Chan and "Jen
  Charlton"; Hunley as Hanley and Hummel. Normalize against the roll call at
  the top of the video, and write "a board member" when no name precedes the
  turn — never guess. This is also why `votes.py` publishes counts but not
  names: keep caption-guessed names out of anything reader-facing.
- **Relative dates belong to the meeting, not to today.** "The survey opens
  tomorrow," in the work session held 2026-08-04, means August 5 — not the day
  you are reading it. Same for "this evening" and "next month."
