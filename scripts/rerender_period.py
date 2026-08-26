#!/usr/bin/env python3
"""Re-render an archived period recap without re-running the whole pipeline.

For iterating on render.py's LAYOUT. Rebuilds what run_period() had in memory:
merged items (merge and triage are deterministic), signals restored from
out/recap-score-<period>.json, votes from out/recap-votes-<numberdate>.json,
and the public-comment tally re-derived from the meeting's cached transcript.

NOT a substitute for the pipeline. It re-runs `classify_speakers`, which is a
Claude call — free while llmcache has the answer, billed the moment its prompt
changes. It also takes the highlight list from HIGHLIGHTS below rather than
from the checkpoint, and it rewrites out/recap-pubcomment-*.json in place.

Usage:  python scripts/rerender_period.py [out.md]
Edit PERIOD, VIDEOS, AGENDAS and HIGHLIGHTS for a period other than 202608.
"""
import json, pathlib, sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import dataclasses

import make_newsletter as mn
import merge, pubcomment, votes
from render import render_recap
from score import compute_deterministic, finalize
from triage import kept_items, triage

PERIOD = "202608"
VIDEOS = ["20260804=https://www.youtube.com/watch?v=HSimRbMP5QQ",
          "20260818=https://www.youtube.com/watch?v=dDNmOJgzQe0"]
AGENDAS = [
    "20260804=https://wjccschools.community.diligentoneplatform.com/Portal/MeetingInformation.aspx?Id=542",
    "20260818=https://wjccschools.community.diligentoneplatform.com/Portal/MeetingInformation.aspx?Id=547",
]

HIGHLIGHTS = {"0804-6.01", "0804-8.07"}

sources = mn._period_sources(PERIOD, VIDEOS, AGENDAS)
merged, _ = merge.merge_meetings(sources, verbose=False)
meta = merge.period_meta(sources)
preamble = sorted(sources, key=lambda s: s.numberdate)[-1].preamble
result = triage(merged, meta, preamble)
all_items = kept_items(result)
scored = compute_deterministic(all_items, result)

# Signals from the archived score file.
saved = {e["item"]["number"]: e for e in
         json.loads(pathlib.Path("out/recap-score-202608.json").read_text())}
for s in scored:
    if not (entry := saved.get(s.item.number)):
        continue
    for k, v in entry["signals"].items():
        setattr(s.signals, k, v)
finalize(scored)
order = {num: n for n, num in enumerate(saved)}
scored.sort(key=lambda s: order.get(s.item.number, 999))

# Votes from the archived roll-call files.
for src in sources:
    path = pathlib.Path(f"out/recap-votes-{src.numberdate}.json")
    if not path.is_file():
        continue
    raw = json.loads(path.read_text())["roll_calls"]
    mine = [i for i in all_items if i.number.partition("-")[0] == src.prefix]
    recovered, _ = votes.to_votes(raw, mine)
    votes.apply_to_items(recovered, mine)

# Public comment: saved speakers, re-tallied against the cached transcripts so
# the word counts are computed exactly as the pipeline computes them.
mn._load_dotenv()
import anthropic
client = anthropic.Anthropic()
tallies = []
all_speakers = []
all_ranges = []
for src in sorted(sources, key=lambda s: s.numberdate):
    path = pathlib.Path(f"out/recap-pubcomment-{src.numberdate}.json")
    if not path.is_file() or not src.video:
        continue
    per = json.loads(path.read_text())
    # Same reclaim run_period does. Without it a rerender reproduces whatever
    # window was saved BEFORE extend_to_next_item landed — on Aug 18 that
    # window closes two seconds before the eleventh speaker starts, and the
    # recap goes out claiming ten. The count is the product here; a stale
    # window silently undercounts it.
    pc_ranges = pubcomment.extend_to_next_item(
        per["public_comment_period"],
        [getattr(sc.signals,
                 "work_session_start_seconds" if src.kind == "work_session"
                 else "meeting_start_seconds") for sc in scored],
    )
    all_ranges += pc_ranges
    snippets = mn.fetch_transcript(src.video, cache_dir=mn.CACHE_DIR)
    text = pubcomment.compact_slice(snippets, pc_ranges)
    mine = pubcomment.classify_speakers(text, all_items, client, pc_ranges=pc_ranges)
    for sp in mine:
        sp.meeting = src.numberdate
    mine = pubcomment.attach_off_agenda(mine, all_items)
    all_speakers += mine
    tallies.append(pubcomment.tally(mine, all_items, pc_ranges, snippets))
    path.write_text(pubcomment.to_json(mine, tallies[-1], pc_ranges))
    review = pubcomment.write_review_md(
        src.numberdate, mine, all_items, snippets, pc_ranges,
        title=f"Public comment — {src.label}", video_url=src.video,
    )
    print(f"wrote {review}")
pc_tally = pubcomment.merge_tallies(tallies, all_items)
pubcomment.apply_to_signals(scored, pc_tally)
pathlib.Path(f"out/recap-pubcomment-{PERIOD}.json").write_text(
    pubcomment.to_json(all_speakers, pc_tally, all_ranges))

item_slices, attachment_slices = mn._packet_paths(
    sorted(sources, key=lambda s: s.numberdate))
quotes, speaker_quotes, vote_notes, watch_starts = mn._load_quotes(PERIOD)

top = [s for s in scored if s.item.number in HIGHLIGHTS]
body = render_recap(
    result.logistics, top, all_items, result.action_items, result.consent_agenda,
    meeting_unique=meta.get("unique"),
    meeting_video=next(s.video for s in sources if s.kind == "meeting"),
    work_session_video=next(s.video for s in sources if s.kind == "work_session"),
    public_comment=pc_tally,
    period_label=merge.period_label(sources),
    period_meetings=meta["period_meetings"],
    packet_paths=item_slices, attachment_paths=attachment_slices,
    quotes=quotes, speaker_quotes=speaker_quotes, vote_notes=vote_notes,
    watch_starts=watch_starts,
)
out = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "recap-new.md")
out.write_text(body)
print(f"wrote {out}")
