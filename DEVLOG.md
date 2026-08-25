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
