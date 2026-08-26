# Devlog

Session residue: what was tried and abandoned, why an approach was chosen,
what is still uncertain. The diff is in `git log`; this is what it doesn't say.

## 2026-08-25 — Recap layout pass, and where the model's words are allowed

Reworked the recap's presentation against the August issue. Two things were
tried and dropped. First, word counts over each speaker's captions, as the
answer to "what did public comment say" without quoting anyone: they worked
(`pubcomment._count_words` and the stopword list are still there, still
counted into the tally JSON) but read as noise, and the maintainer didn't want
them. Nothing renders them now; the counting stayed because it costs nothing
and the field already round-trips. Second, `group_subtopics` originally
returned nothing when every speaker got a distinct label, on the theory that a
one-per-line "grouping" is a flat list with extra words. That was wrong for
the case that matters: the six August redistricting speakers asked for six
different things, and six labelled lines carry the story that six bare
timestamps do not. The guard is gone; only a PARTIAL labelling falls back.

That made the model's labels load-bearing in a way the off-agenda topic labels
never were — they are now the only description of what each speaker asked for.
CLAUDE.md and the README were updated to say so plainly rather than keep
claiming no model words reach the page.

Adding `subtopic` to the classifier's schema changed the llmcache key, so
`classify_speakers` re-ran and every speaker anchor moved 1-13 seconds. That
silently broke the excerpt keys in `quotes-202608.json`, which are anchors by
design. They were re-mapped one-to-one within 20s and the drift hazard is now
documented in that file's own README. Worth remembering: any prompt change in
pubcomment.py invalidates hand-written speaker keys.

Members' names now reach the page in exactly one place — a roll call's
absentees — via a new `votes` block in the maintainer's quotes file. The Aug 4
budget absentee (Michael Hosang) was identified by elimination against the
roster the district publishes in its own BoardDocs vote records, because the
elimination needs "Dr. Kvassos" read as Cavazos and no matcher here can defend
that. It is a human's inference recorded in a human's file, and it is still
unconfirmed against the video.

## 2026-08-26 — Layout fixes, and a review artifact for the model's labels

Four presentation changes against the August draft, all in render.py: the
"timestamps open the recording" gloss now ends the sentence that introduces
the speaker list instead of standing as its own line above it (one constant,
`_TIMESTAMP_NOTE`, now used verbatim in both places — "More Public Comment"
had a second, differently-worded copy); the per-speaker excerpts under "More
Public Comment" are gone, because a quote beside every timestamp crowded a
list whose whole job is to be countable; upcoming meetings are Date/Time/
Location sub-bullets, since the board's time strings run a full sentence
("Call to Order & Closed Session at 4:00 p.m.; Open Session at 4:30 p.m.")
and comma-joining buried the location behind them; and the footer now points
at the district's School Board page, which lists every member's address —
the roster page rather than a mailbox, because there is no single address for
the board as a body.

The excerpts are still in `quotes-202608.json` and `_load_quotes` still reads
them. `render_recap` keeps `speaker_quotes` as an accepted-but-unrendered
parameter so the maintainer's hand-keyed work is not stranded — those keys are
anchors, and re-deriving them after a prompt change is exactly the drift
hazard the last session documented.

New: `pubcomment.review_markdown` / `write_review_md`, writing
`out/pubcomment-review-<numberdate>.md`. Each speaker's subtopic label printed
above the verbatim captions it was assigned from. The subtopic labels became
load-bearing last session and had no verification surface — checking six of
them meant scrubbing the video for six start times. The span rule was factored
out of `tally` into `speaker_spans` and both now call it, so the text under a
label is exactly the text the tally counted; if they drifted the artifact
would silently stop reviewing anything. Nothing in it goes near the
classifier's prompt, so it regenerates for free. Named `pubcomment-review-`
rather than `recap-pubcomment-` on purpose: carryforward.py globs
`recap-pubcomment-<8 digits>.json` and a review file inside that glob would be
read as a tally.

Reading the Aug 18 file: the labels hold up, but the classifier's start anchors
sit ~10s off, so most spans open on the clerk announcing the *previous*
speaker's name and close on the chair thanking them. Harmless for label
review; it would not be harmless if anything ever quoted a span's edges.

Unrelated and pre-existing: `test_llmcache.py` does not collect — every test
takes a `tmp` fixture that is not defined.

## 2026-08-26 (later) — Highlights stop claiming minutes

Cut the "Discussed 6 min at the regular meeting … and 48 min at the work
session" line that opened every highlight. It was the most confident thing on
the page and one of the least true: `score._SEGMENTATION_SYSTEM` asks for where
the BOARD discussed, debated or voted on an item, which a literal reader
excludes a staff presentation from, so August's budget item measured 0.5
minutes — the vote at 1:22:18 — while its 7:20 presentation and the discussion
after it went unmeasured. The minutes still rank items. They are just not a
claim on the page any more.

In their place, after the maintainer's quote: `_watch_line`'s "**Watch this
agenda item:** [Aug 4 work session, from 28:37] · [Aug 18 regular meeting, from
48:54]". Wording is deliberately about WHERE and not what: an item can be a
presentation, a discussion, a vote, or all three in one evening. "from X" plus
the meeting name also keeps it distinct from the quote's credit timestamp
directly above, which points at a sentence rather than a segment.

Because the segmenter's anchors are wrong often enough to matter, they are now
overridable per item per meeting in `quotes-<period>.json`'s new `watch` block
("MM:SS", "H:MM:SS" or bare seconds). Item numbers are stable — merge.py owns
them — so unlike the speaker-anchor keys these cannot drift under a prompt
change. Recorded for August: 6.01 → 28:37, 8.07 → 7:20, both read off the video
by a person.

Careful with `_signal_line`: only its timing clauses were removed. It still
emits `carry_forward_note`, the dollar fallback and "routine", none of which
August's two highlights happen to carry — deleting the call would have looked
clean in a diff and silently dropped three clause types from every later recap.

Subtopic labels no longer render under a highlight; the speakers are bare
timestamps again. `group_subtopics` still runs and the tally JSON still carries
the groups. The argument that reversed last session's decision: a few-word
label describing two minutes of speech can only be checked by listening to the
two minutes, which is exactly what the timestamp beside it already offers. The
model's words now reach the page in one place only, the off-agenda topic labels
in "More Public Comment".

**Undercount caught before publishing.** `scripts/rerender_period.py` read
`public_comment_period` straight out of the saved JSON and never called
`extend_to_next_item`, so every rerender reproduced the pre-fix Aug 18 window,
which closes two seconds before the eleventh speaker starts. Both the old and
new drafts said "10 people spoke"; the meeting had 11, and the review artifact
had been ending on the chair saying "our last speaker is Robert Fier" the whole
time. The script now mirrors run_period. Re-slicing changed the cache key, so
`classify_speakers` re-ran once (billed) and every anchor moved a few seconds —
harmless now that nothing on the page is keyed to those anchors.
