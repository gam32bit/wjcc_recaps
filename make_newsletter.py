#!/usr/bin/env python3
"""Orchestrate the WJCC agenda -> newsletter pipeline.

The default product is the RECAP of a meeting that already happened. It needs the
FINALIZED agenda (with vote counts) — re-fetch it first with
`python pull_agenda.py --date <date>` — plus the meeting video:

    python make_newsletter.py --date 20260616 \\
        --meeting https://www.youtube.com/watch?v=PvOFaqmguYg \\
        --work-session https://www.youtube.com/live/Y-LpLQm27AI   # optional, for deep-links

A full interactive recap run:
  1. parse + triage (offline)
  2. fetch the meeting transcript (and optionally the work-session transcript);
     both are also saved as out/transcript-*-<date>.md
  3. score: segmentation call(s) + public-comment speaker call + rubric call
  4. CHECKPOINT: review the ranking and the public-comment tally
  5. render -> out/recap-<date>.md

The recap itself contains no model-written prose. Highlights quote the agenda's
own wording; the public-comment section is a counted tally, not a summary.

Since the district moved to Diligent Community (2026-07-01) there is no
BoardDocs HTML to fetch. Parse the meeting's own agenda packet PDF first, into
a suffixed fixture so it does not overwrite the work-session packet's forecast
of the same meeting, then pass that fixture key to --date:

    python pdfagenda.py "wjcc-fixtures/Regular Meeting - Aug 18 2026 ....pdf" \\
        --recap --target 20260818 --suffix=-regular --fixtures
    python make_newsletter.py --date 20260818-regular \\
        --meeting      https://www.youtube.com/watch?v=dDNmOJgzQe0 \\
        --work-session https://www.youtube.com/watch?v=HSimRbMP5QQ

A Diligent packet is published BEFORE the meeting and never amended, so it
carries recommended actions rather than recorded votes. The outcomes are read
off the meeting video's roll calls instead — see votes.py. Outputs are still
named by numberdate, so `forecast.py --check <date>` finds
out/recap-score-<date>.json where it always did.

--period recaps a whole MONTH as one newsletter, which is the shape the board
actually works in: it reviews an item at the work session and votes on it two
weeks later, so either meeting alone is half a story. Reviews are folded into
the votes they became (merge.py), an item's discussion time is the work session
plus the meeting, and public comment at one meeting attaches to an item from the
other:

    python pdfagenda.py "wjcc-fixtures/Work Session ... Aug 04 2026 ....pdf" \
        --recap --target 20260804 --fixtures
    python make_newsletter.py --period 202608 \
        --video 20260804=https://www.youtube.com/watch?v=HSimRbMP5QQ \
        --video 20260818=https://www.youtube.com/watch?v=dDNmOJgzQe0 \
        --agenda-url 20260804=https://... --agenda-url 20260818=https://...

Each meeting's header links its own recording and its own agenda. Diligent
publishes no per-meeting URL the pipeline can derive, so --agenda-url is how
one gets in; without it that meeting shows its Watch link alone. Quotes chosen
by hand from the videos go in quotes-<period>.json (see that file's _README).

A period run writes out/recap-<period>.md and leaves the per-meeting
out/recap-score-<numberdate>.json files alone, so a forecast check still
measures itself against the single-meeting ranking it predicted.

Pass --preview for the upcoming-meeting product ("The Rundown"), which scores
discussion from the prior work session instead:

    python make_newsletter.py --date 20260519 --preview \\
        --work-session https://www.youtube.com/live/Y-LpLQm27AI \\
        --livestream   https://www.youtube.com/live/3np-qU3BSnQ

--dry-run stops after step 3 (deterministic signals only, no API calls).
"""

import argparse
import dataclasses
import json
import os
import pathlib
import sys

import anthropic

import merge
import pubcomment
import votes
from forecast import _agenda_item
from parse import agenda_preamble, parse_agenda, to_dict
from render import render_newsletter, render_recap
from score import (
    ScoredItem,
    compute_deterministic,
    evidence_line,
    finalize,
    score_rubric,
    segment_transcript,
    _print_table,
)
from transcript import fetch_transcript, transcript_sections, write_transcript_md
from triage import Logistics, kept_items, print_report, triage
from write import Draft, draft, validate_draft

PROJECT_DIR = pathlib.Path(__file__).resolve().parent
FIXTURES_DIR = PROJECT_DIR / "wjcc-fixtures"
CACHE_DIR    = PROJECT_DIR / ".cache"
OUT_DIR      = PROJECT_DIR / "out"


# --- Fixture discovery -----------------------------------------------------

def _discover_fixtures() -> dict[str, tuple[pathlib.Path, pathlib.Path, str]]:
    """Map each fixture KEY to (agenda_path, meta_path, numberdate).

    The key is whatever follows `meeting-meta-` in the filename. It is usually
    just the numberdate, but one meeting can have two agenda documents and they
    cannot share a name: Aug 18 2026 has the Aug 4 work-session packet's
    projection of it (`20260818`, the forecast's input) and its own regular
    packet (`20260818-regular`, the recap's). So the key may carry a suffix, and
    the numberdate — which names every output file — is read from inside the
    JSON rather than assumed from the filename.

    An agenda is either BoardDocs HTML or a Diligent packet pre-parsed to JSON
    by `pdfagenda.py --fixtures`.
    """
    fixtures: dict[str, tuple[pathlib.Path, pathlib.Path, str]] = {}
    for meta_path in FIXTURES_DIR.glob("meeting-meta-*.json"):
        key = meta_path.stem[len("meeting-meta-"):]
        agenda_path = next(
            (p for p in (FIXTURES_DIR / f"agenda-{key}.html",
                         FIXTURES_DIR / f"agenda-{key}.json") if p.is_file()),
            None,
        )
        if agenda_path is None:
            continue
        try:
            numberdate = json.loads(meta_path.read_text()).get("numberdate", "")
        except json.JSONDecodeError:
            continue
        if numberdate:
            fixtures[key] = (agenda_path, meta_path, numberdate)
    return fixtures


def _select_fixture(selector: str | None) -> tuple[str, pathlib.Path, pathlib.Path]:
    """Resolve `--date` to one fixture. Accepts a key or a plain numberdate.

    When a numberdate maps to more than one document the choice is refused
    rather than guessed — picking the wrong one produces a plausible recap of
    the wrong agenda, which is the failure mode this project can least afford.
    """
    fixtures = _discover_fixtures()
    if not fixtures:
        raise SystemExit(f"No agenda fixtures found in {FIXTURES_DIR}.")

    if not selector:
        chosen = max(fixtures, key=lambda k: (fixtures[k][2], k))
    elif selector in fixtures:
        chosen = selector
    else:
        matches = sorted(k for k, v in fixtures.items() if v[2] == selector)
        if len(matches) > 1:
            raise SystemExit(
                f"{selector} has {len(matches)} agenda documents: "
                + ", ".join(matches)
                + ".\nPass one of those to --date."
            )
        if not matches:
            available = ", ".join(sorted(fixtures)) or "none"
            raise SystemExit(f"No fixture for {selector}. Available: {available}")
        chosen = matches[0]

    agenda_path, meta_path, numberdate = fixtures[chosen]
    return numberdate, agenda_path, meta_path


# --- Checkpoint UI ---------------------------------------------------------

def _print_checkpoint(scored: list[ScoredItem], *, dry_run: bool = False) -> None:
    _print_table(scored, dry_run=dry_run)
    if dry_run:
        print("(dry-run: discussion_minutes and rubric_score are 0; no API calls made)")
    else:
        print("Scores include: discussion time, dollar magnitude, vote type, rubric score.")
    print()


def _checkpoint_interaction(scored: list[ScoredItem], default_top: int = 5) -> tuple[list[ScoredItem], int]:
    """Interactive loop: let the user approve or adjust the ranking.

    Returns (adjusted_scored_list, top_n).
    """
    print("Commands:")
    print("  [Enter]         Approve this ranking")
    print("  top N           Change how many top items to draft (default 5)")
    print(f"  exclude N.NN    Remove an item from consideration")
    print(f"  include N.NN    Re-add an excluded item")
    print()

    top_n = default_top
    excluded: set[str] = set()
    active = list(scored)

    while True:
        try:
            raw = input(f"> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            raise SystemExit("Aborted at checkpoint.")

        if not raw:
            break

        parts = raw.split()
        cmd = parts[0].lower()

        if cmd == "top" and len(parts) == 2:
            try:
                top_n = int(parts[1])
                print(f"  Top-N set to {top_n}.")
            except ValueError:
                print(f"  Bad number: {parts[1]!r}")

        elif cmd == "exclude" and len(parts) == 2:
            num = parts[1]
            match = [s for s in active if s.item.number == num]
            if not match:
                print(f"  Item {num!r} not found in active list.")
            else:
                excluded.add(num)
                active = [s for s in active if s.item.number not in excluded]
                print(f"  Excluded {num}. {len(active)} items remain.")
                _print_checkpoint(active)

        elif cmd == "include" and len(parts) == 2:
            num = parts[1]
            excluded.discard(num)
            # Re-build active from original scored list minus excluded
            active = [s for s in scored if s.item.number not in excluded]
            finalize(active)
            print(f"  Re-included {num}. {len(active)} items remain.")
            _print_checkpoint(active)

        else:
            print(f"  Unknown command: {raw!r}. Type Enter to approve.")

    return active, top_n


def _attachment_selection(
    top_items: list[ScoredItem],
) -> list[tuple[ScoredItem, int]]:
    """Ask which attachments to fetch for the top items.

    Returns a list of (ScoredItem, attachment_index) pairs to fetch.
    """
    has_attachments = [s for s in top_items if s.item.attachments]
    if not has_attachments:
        print("No attachments available for the top items.")
        return []

    print("Attachments for top items (for quote-grounding in the prose call):")
    print()
    choices: list[tuple[str, ScoredItem, int]] = []  # (label, scored_item, att_idx)
    for s in has_attachments:
        print(f"  {s.rank}. [{s.item.number}] {s.item.title[:55]}")
        for j, att in enumerate(s.item.attachments):
            label = f"{s.rank}{chr(ord('a') + j)}"
            choices.append((label, s, j))
            print(f"        {label}) {att.name}")
    print()
    all_labels = " ".join(c[0] for c in choices)
    print(f"Fetch which attachments? (all / none / space-separated labels like '{all_labels[:6]}...')")

    try:
        raw = input("> ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print()
        return []

    if not raw or raw == "none":
        return []
    if raw == "all":
        return [(s, j) for _, s, j in choices]

    requested_labels = set(raw.split())
    selected = [(s, j) for label, s, j in choices if label in requested_labels]
    unknown = requested_labels - {label for label, _, _ in choices}
    if unknown:
        print(f"  Unknown labels ignored: {', '.join(sorted(unknown))}")
    return selected


# --- Env loading -----------------------------------------------------------

def _load_dotenv() -> None:
    env_path = PROJECT_DIR / ".env"
    if not env_path.is_file():
        return
    for raw in env_path.read_text().splitlines():
        line = raw.strip().removeprefix("export ").strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def _public_comment_tally(
    date: str,
    scored: list[ScoredItem],
    all_items: list,
    snippets: list[dict],
    pc_ranges: list[dict],
    client: anthropic.Anthropic,
) -> tuple[list, "pubcomment.PublicCommentTally"]:
    """Identify public-comment speakers, count them, and write the audit file.

    The counting is plain Python; Claude only says where each speaker starts
    and which agenda item they addressed. The saved JSON holds the raw answer,
    the derived tally, and the period bounds, so the section can be re-rendered
    (and re-checked) later without another API call.
    """
    text = pubcomment.compact_slice(snippets, pc_ranges)
    speakers = pubcomment.classify_speakers(
        text, all_items, client, pc_ranges=pc_ranges
    )
    speakers = pubcomment.attach_off_agenda(speakers, all_items)
    result = pubcomment.tally(speakers, all_items, pc_ranges)
    pubcomment.apply_to_signals(scored, result)

    path = OUT_DIR / f"recap-pubcomment-{date}.json"
    path.write_text(pubcomment.to_json(speakers, result, pc_ranges))
    print(
        f"  {result.total_speakers} public-comment speakers across "
        f"{len(result.by_item)} agenda items and {len(result.off_agenda)} "
        f"off-agenda topics. Wrote {path.relative_to(PROJECT_DIR)}"
    )
    return speakers, result


def _fetch_selected_attachments(
    top_items: list[ScoredItem],
) -> dict[str, pathlib.Path]:
    """Ask which PDFs to fetch, then fetch them (preview drafting only).

    Only the preview product sends documents to Claude. The recap links to
    attachments straight from the parsed agenda and never downloads them.
    """
    selections = _attachment_selection(top_items)
    if not selections:
        return {}

    from attachments import fetch_attachment
    from pull_agenda import new_session

    print("\nFetching attachments...")
    session = new_session()
    paths: dict[str, pathlib.Path] = {}
    for s, att_idx in selections:
        att = s.item.attachments[att_idx]
        paths[att.url] = fetch_attachment(att.url, session, cache_dir=CACHE_DIR)
    return paths


def _print_public_comment(result: "pubcomment.PublicCommentTally") -> None:
    """Show the tally at the checkpoint — including the off-agenda labels.

    Those labels are the only model-authored strings that reach the published
    recap, so they get looked at before anyone hits publish.
    """
    if not result or not result.total_speakers:
        return
    print(f"\nPublic comment — {result.total_speakers} speakers:")
    for t in result.by_item:
        print(f"  {t.count:>3}  [{t.item_number}] {t.label[:55]}")
    for t in result.off_agenda:
        print(f"  {t.count:>3}  (off-agenda) {t.label}")


# --- Pipeline --------------------------------------------------------------

def _load_agenda(agenda_path: pathlib.Path) -> tuple[list, str]:
    """Parse an agenda fixture into (items, preamble).

    Two shapes, because the source changed underneath the project: BoardDocs
    agendas are HTML and parsed by `parse.py`; Diligent packets are PDFs
    pre-parsed to JSON by `pdfagenda.py --fixtures`, since a packet's item
    bodies live hundreds of pages apart and re-extracting them per run costs
    minutes for no gain. Same `AgendaItem` either way, so triage, scoring and
    rendering never learn which one they got.
    """
    if agenda_path.suffix == ".json":
        payload = json.loads(agenda_path.read_text())
        return [_agenda_item(raw) for raw in payload["items"]], payload.get("preamble", "")
    html = agenda_path.read_text()
    return parse_agenda(html), agenda_preamble(html)


def _period_sources(
    period: str, video_args: list[str], agenda_args: list[str] | None = None,
) -> list[merge.SourceMeeting]:
    """Collect every fixture whose numberdate falls in `period` (YYYYMM).

    A meeting's KIND — work session or regular — decides which discussion-time
    signal its transcript feeds, and it is read from the packet the fixture was
    built from rather than guessed from the date, because the board holds both
    kinds on Tuesdays.
    """
    def by_numberdate(specs: list[str], flag: str) -> dict[str, str]:
        out: dict[str, str] = {}
        for spec in specs:
            numberdate, sep, url = spec.partition("=")
            if not sep:
                raise SystemExit(f"{flag} wants NUMBERDATE=URL, got {spec!r}")
            out[numberdate.strip()] = url.strip()
        return out

    videos = by_numberdate(video_args, "--video")
    agendas = by_numberdate(agenda_args or [], "--agenda-url")

    sources: list[merge.SourceMeeting] = []
    for key, (agenda_path, meta_path, numberdate) in sorted(_discover_fixtures().items()):
        if not numberdate.startswith(period):
            continue
        meta = json.loads(meta_path.read_text())
        if meta.get("role") == "forecast":
            # The work-session packet's projection of the NEXT meeting. It is
            # filed under that meeting's date, so without this it would enter
            # the period a second time, as a duplicate of both meetings at once.
            continue
        items, preamble = _load_agenda(agenda_path)
        source_tag = meta.get("source", "")
        if "worksession" in source_tag:
            kind = "work_session"
        elif "regular" in source_tag:
            kind = "meeting"
        else:
            # A BoardDocs fixture names no packet kind; those are regular
            # meetings, which is what the pipeline assumed before Diligent.
            kind = "meeting"
        sources.append(merge.SourceMeeting(
            numberdate=numberdate, kind=kind, items=items, meta=meta,
            preamble=preamble, video=videos.get(numberdate),
            agenda=agendas.get(numberdate),
        ))

    if not sources:
        raise SystemExit(f"No agenda fixtures for period {period}.")
    known = {s.numberdate for s in sources}
    for flag, given in (("--video", videos), ("--agenda-url", agendas)):
        if (unknown := set(given) - known):
            raise SystemExit(
                f"{flag} given for {', '.join(sorted(unknown))}, which has no "
                f"fixture in {period}."
            )
    return sources


def _packet_paths(
    sources: list[merge.SourceMeeting],
) -> tuple[dict[str, str], dict[str, str]]:
    """Packet slices written by `pdfslice.py`, keyed the way the recap keys them.

    Returns (item details by namespaced item number, attachments by namespaced
    packet page) — see `pdfslice.slice_packet` for why the attachment map is
    keyed by page.

    `pdfslice` works on one meeting and writes the agenda's own numbering;
    `merge.py` prefixes every number with its meeting. Namespacing here keeps
    pdfslice unaware of the period, which is the same split every other
    per-meeting artifact uses.
    """
    paths: dict[str, str] = {}
    attachments: dict[str, str] = {}
    for source in sources:
        path = OUT_DIR / f"packet-{source.numberdate}.json"
        if not path.is_file():
            continue
        data = json.loads(path.read_text())
        # Files written before attachments were sliced are a flat
        # {number: path} map; read them as the details half.
        items = data.get("items", data) if "items" in data else data
        for number, rel in items.items():
            paths[f"{source.prefix}-{number}"] = rel
        for page, rel in data.get("attachments", {}).items():
            attachments[f"{source.prefix}-{page}"] = rel
    return paths, attachments


def _load_quotes(period: str) -> tuple[dict[str, dict], dict[str, dict]]:
    """Maintainer-chosen quotes for this period, from `quotes-<period>.json`.

    Returns (one quote per agenda item, one excerpt per public-comment
    speaker). Human-owned in the way `rubric.md` is: a person watches the
    meeting, picks the line, and records where it came from. Nothing in the
    pipeline writes this file, and a period without one renders no quotes.
    """
    path = PROJECT_DIR / f"quotes-{period}.json"
    if not path.is_file():
        print(f"  No {path.name} — rendering without quotes.")
        return {}, {}
    data = json.loads(path.read_text())
    items = data.get("items", {})
    speakers = data.get("speakers", {})
    n_speakers = sum(len(v) for v in speakers.values())
    print(f"  {len(items)} item quote(s) and {n_speakers} speaker excerpt(s) "
          f"from {path.name}.")
    return items, speakers


def run_period(
    period: str,
    sources: list[merge.SourceMeeting],
    *,
    top_n: int = 3,
    dry_run: bool = False,
    no_checkpoint: bool = False,
) -> None:
    """Recap a PERIOD — every meeting in a month — as one newsletter.

    The board decides across two meetings, so a recap of one of them tells half
    the story. See `merge.py` for why the agendas are combined rather than
    concatenated, and `score.segment_transcript(accumulate=True)` for why an
    item's discussion time is the work session plus the meeting.
    """
    OUT_DIR.mkdir(exist_ok=True)
    label = merge.period_label(sources)
    print(f"\n{'='*60}")
    print(f"WJCC Recap — {label} ({len(sources)} meetings)")
    print(f"{'='*60}\n")

    merged, decisions = merge.merge_meetings(sources)
    meta = merge.period_meta(sources)
    preamble = sorted(sources, key=lambda s: s.numberdate)[-1].preamble
    result = triage(merged, meta, preamble)
    print_report(result)
    all_items = kept_items(result)
    if not all_items:
        raise SystemExit("Triage kept no substantive items across the period.")

    if not any(i.vote for i in all_items):
        print(
            "\n  NOTE: Diligent agenda packets carry recommended actions, not "
            "recorded votes. Tallies will be read off the meeting videos' "
            "roll calls instead — see votes.py.\n"
        )

    scored = compute_deterministic(all_items, result)

    if dry_run:
        finalize(scored)
        _print_checkpoint(scored, dry_run=True)
        out_path = OUT_DIR / f"recap-score-dry-{period}.json"
        out_path.write_text(json.dumps([dataclasses.asdict(s) for s in scored], indent=2))
        print(f"Wrote {out_path.relative_to(PROJECT_DIR)}")
        return

    _load_dotenv()
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("ANTHROPIC_API_KEY is not set (env var or .env file).")
    client = anthropic.Anthropic()

    # --- Steps 2-3: one transcript pass per meeting ---
    tallies: list[pubcomment.PublicCommentTally] = []
    all_speakers: list[pubcomment.Speaker] = []
    pc_ranges_by_meeting: dict[str, list[dict]] = {}
    for source in sorted(sources, key=lambda s: s.numberdate):
        if not source.video:
            print(f"\n  NOTE: no video for {source.label} — its discussion time "
                  "and public comment are unmeasured.")
            continue
        print(f"\nFetching transcript — {source.label}...")
        snippets = fetch_transcript(source.video, cache_dir=CACHE_DIR)
        if not snippets:
            continue
        seg = segment_transcript(
            scored, snippets, client, kind=source.kind, score=True, accumulate=True
        )
        sections = transcript_sections(
            scored,
            "work_session_start_seconds" if source.kind == "work_session"
            else "meeting_start_seconds",
        )
        pc_ranges = seg.get("public_comment_period", [])
        if pc_ranges:
            # The model is asked for TIGHT ranges and on Aug 18 it closed public
            # comment two seconds before the last speaker started, losing him
            # from the count entirely. Reclaim any speaker-sized gap between the
            # window and the first agenda item mapped in the same video.
            start_key = ("work_session_start_seconds"
                         if source.kind == "work_session" else "meeting_start_seconds")
            pc_ranges = pubcomment.extend_to_next_item(
                pc_ranges,
                [getattr(sc.signals, start_key) for sc in scored],
            )
            pc_start = min(r["start_seconds"] for r in pc_ranges)
            sections.append((float(pc_start), "Public comment"))
            text = pubcomment.compact_slice(snippets, pc_ranges)
            speakers = pubcomment.classify_speakers(
                text, all_items, client, pc_ranges=pc_ranges
            )
            # Tag every speaker with the meeting they spoke at: a period recap
            # pools several meetings' speakers, and a bare timestamp does not
            # say which video it indexes into.
            for sp in speakers:
                sp.meeting = source.numberdate
            # Claude answers "is this on the agenda?" about the agenda in front
            # of it. In a period recap the agenda is the month's, so a speaker
            # who named an item from the OTHER meeting has to be reconnected —
            # deterministically, not by re-prompting.
            speakers = pubcomment.attach_off_agenda(speakers, all_items)
            meeting_tally = pubcomment.tally(speakers, all_items, pc_ranges)
            tallies.append(meeting_tally)
            all_speakers += speakers
            pc_ranges_by_meeting[source.numberdate] = pc_ranges
            sections += pubcomment.speaker_anchors(speakers, all_items)
            print(f"  {meeting_tally.total_speakers} public-comment speakers at "
                  f"{source.label}.")
            # Also saved per meeting, under its own numberdate: carryforward.py
            # globs `recap-pubcomment-<8 digits>.json` to build the prior-turnout
            # signal, and would not see a period file.
            (OUT_DIR / f"recap-pubcomment-{source.numberdate}.json").write_text(
                pubcomment.to_json(speakers, meeting_tally, pc_ranges)
            )
        write_transcript_md(
            source.numberdate,
            "worksession" if source.kind == "work_session" else "meeting",
            snippets, sections, video_url=source.video,
        )

        # Recover this meeting's roll-call votes from the same transcript. A
        # Diligent packet is published before the meeting, so the video is the
        # only record of what actually passed. See votes.py.
        mine = [i for i in all_items
                if i.number.partition("-")[0] == source.prefix]
        raw = votes.extract(mine, snippets, client)
        recovered, warnings = votes.to_votes(raw, mine)
        landed = votes.apply_to_items(recovered, mine)
        for note in warnings:
            print(f"  ! {note}")
        print(f"  {landed} of {len(mine)} items got a recorded vote from "
              f"{source.label}.")
        (OUT_DIR / f"recap-votes-{source.numberdate}.json").write_text(
            votes.to_json(raw, recovered, warnings)
        )

    # `vote_contested` is a scored signal and was computed before any transcript
    # existed, so it has to be refreshed now that the votes are known.
    for s_item in scored:
        s_item.signals.vote_contested = bool(
            s_item.item.vote and s_item.item.vote.contested
        )

    pc_tally = None
    if tallies:
        pc_tally = pubcomment.merge_tallies(tallies, all_items)
        pubcomment.apply_to_signals(scored, pc_tally)
        path = OUT_DIR / f"recap-pubcomment-{period}.json"
        path.write_text(pubcomment.to_json(
            all_speakers, pc_tally,
            [r for rs in pc_ranges_by_meeting.values() for r in rs],
        ))
        print(f"  Wrote {path.relative_to(PROJECT_DIR)}")

    rubric_path = PROJECT_DIR / "rubric.md"
    if rubric_path.is_file():
        score_rubric(scored, rubric_path.read_text(), client)
    else:
        print("  WARNING: rubric.md not found.")

    finalize(scored)

    # --- Step 4: checkpoint ---
    _print_checkpoint(scored)
    _print_public_comment(pc_tally)
    if no_checkpoint:
        print("(--no-checkpoint: accepting this ranking unreviewed)\n")
        active = scored
    else:
        active, top_n = _checkpoint_interaction(scored, default_top=top_n)
    top_items = active[:top_n]
    print(f"\nProceeding with top {len(top_items)} items:")
    for s in top_items:
        print(f"  {s.rank}. [{s.item.number}] {s.item.title[:60]}")
    print()

    # --- Step 5: render + save ---
    ordered = sorted(sources, key=lambda s: s.numberdate)
    item_slices, attachment_slices = _packet_paths(ordered)
    quotes, speaker_quotes = _load_quotes(period)
    body = render_recap(
        result.logistics,
        top_items,
        all_items,
        result.action_items,
        result.consent_agenda,
        meeting_unique=meta.get("unique"),
        meeting_video=next((s.video for s in ordered if s.kind == "meeting"), None),
        work_session_video=next(
            (s.video for s in ordered if s.kind == "work_session"), None
        ),
        public_comment=pc_tally,
        period_label=label,
        period_meetings=meta["period_meetings"],
        packet_paths=item_slices,
        attachment_paths=attachment_slices,
        quotes=quotes,
        speaker_quotes=speaker_quotes,
    )

    score_path = OUT_DIR / f"recap-score-{period}.json"
    score_path.write_text(json.dumps([dataclasses.asdict(s) for s in active], indent=2))
    merge_path = OUT_DIR / f"recap-merge-{period}.json"
    merge_path.write_text(json.dumps(decisions, indent=2))
    recap_path = OUT_DIR / f"recap-{period}.md"
    recap_path.write_text(body)
    print()
    for path in (score_path, merge_path, recap_path):
        print(f"Wrote {path.relative_to(PROJECT_DIR)}")


def run(
    date: str,
    html_path: pathlib.Path,
    meta_path: pathlib.Path,
    *,
    work_session_url: str | None = None,
    livestream_url: str | None = None,
    meeting_url: str | None = None,
    recap: bool = False,
    dry_run: bool = False,
    top_n: int = 5,
    no_checkpoint: bool = False,
) -> None:
    OUT_DIR.mkdir(exist_ok=True)
    meta = json.loads(meta_path.read_text())

    # --- Step 1: parse + triage ---
    kind = "Recap" if recap else "Newsletter"
    print(f"\n{'='*60}")
    print(f"WJCC {kind} — {date}")
    print(f"{'='*60}\n")
    items, preamble = _load_agenda(html_path)
    result = triage(items, meta, preamble)
    if livestream_url:
        result.logistics.livestream = livestream_url
    print_report(result)
    all_items = kept_items(result)

    if not all_items:
        raise SystemExit("Triage kept no substantive items.")

    if recap and not any(i.vote for i in all_items):
        if meta.get("source", "").startswith("diligent"):
            # A Diligent packet is published BEFORE the meeting and is never
            # amended afterwards, so it has no tallies to re-fetch. The recap
            # drops its vote lines rather than inventing outcomes.
            print(
                "\n  NOTE: a Diligent agenda packet carries recommended actions, "
                "not recorded votes. This recap will have no vote tallies; the "
                "outcomes are in the minutes, approved at the next meeting.\n"
            )
        else:
            print(
                "\n  WARNING: no Motion & Voting blocks found in this agenda. "
                "Re-fetch the FINALIZED post-meeting agenda with "
                f"`python pull_agenda.py --date {date}` before running a recap.\n"
            )

    scored = compute_deterministic(all_items, result)

    if dry_run:
        finalize(scored)
        _print_checkpoint(scored, dry_run=True)
        out_path = OUT_DIR / f"{'recap-score-dry' if recap else 'score-dry'}-{date}.json"
        out_path.write_text(json.dumps([dataclasses.asdict(s) for s in scored], indent=2))
        print(f"Wrote {out_path.relative_to(PROJECT_DIR)}")
        return

    # --- Steps 2-3: transcript(s) + scoring ---
    _load_dotenv()
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("ANTHROPIC_API_KEY is not set (env var or .env file).")
    client = anthropic.Anthropic()

    meeting_snippets: list[dict] = []
    meeting_seg_data: dict = {}
    pc_tally: pubcomment.PublicCommentTally | None = None

    if recap:
        # A recap scores discussion + public comment from the MEETING video, and
        # (when given) also segments the work session — purely to deep-link each
        # item to where it was previewed; that pass does not affect scoring.
        if not meeting_url:
            print(
                "\n  NOTE: no --meeting URL — the recap will rank on rubric + "
                "dollar + vote signals only (no transcripts, deep-links, or "
                "public-comment summary).\n"
            )
        if meeting_url:
            print("\nFetching meeting transcript...")
            meeting_snippets = fetch_transcript(meeting_url, cache_dir=CACHE_DIR)
        print("\nScoring items...")
        if meeting_snippets:
            meeting_seg_data = segment_transcript(
                scored, meeting_snippets, client, kind="meeting", score=True
            )
            # Agenda-item headings cover scored items only; public comment is
            # triaged out but its range is already segmented, so add it for free.
            sections = transcript_sections(scored, "meeting_start_seconds")
            pc_ranges = meeting_seg_data.get("public_comment_period", [])
            if pc_ranges:
                # Same tight-range guard as the period path — see
                # `pubcomment.extend_to_next_item`.
                pc_ranges = pubcomment.extend_to_next_item(
                    pc_ranges,
                    [sc.signals.meeting_start_seconds for sc in scored],
                )
                pc_start = min(r["start_seconds"] for r in pc_ranges)
                sections.append((float(pc_start), "Public comment"))
                # Who spoke about what. This runs BEFORE the checkpoint because
                # its per-item minutes feed the composite, and therefore the
                # ranking the checkpoint asks you to approve.
                speakers, pc_tally = _public_comment_tally(
                    date, scored, all_items, meeting_snippets, pc_ranges, client
                )
                sections += pubcomment.speaker_anchors(speakers, all_items)
            write_transcript_md(
                date, "meeting", meeting_snippets, sections, video_url=meeting_url
            )
        if work_session_url:
            print("\nFetching work-session transcript (for deep-links)...")
            ws_snippets = fetch_transcript(work_session_url, cache_dir=CACHE_DIR)
            if ws_snippets:
                segment_transcript(
                    scored, ws_snippets, client, kind="work_session", score=False
                )
                write_transcript_md(
                    date, "worksession", ws_snippets,
                    transcript_sections(scored, "work_session_start_seconds"),
                    video_url=work_session_url,
                )
    else:
        # A preview measures discussion from the prior work session.
        ws_snippets = []
        if work_session_url:
            print("\nFetching work-session transcript...")
            ws_snippets = fetch_transcript(work_session_url, cache_dir=CACHE_DIR)
        print("\nScoring items...")
        if ws_snippets:
            segment_transcript(
                scored, ws_snippets, client, kind="work_session", score=True
            )
            write_transcript_md(
                date, "worksession", ws_snippets,
                transcript_sections(scored, "work_session_start_seconds"),
                video_url=work_session_url,
            )

    rubric_path = PROJECT_DIR / "rubric.md"
    if rubric_path.is_file():
        score_rubric(scored, rubric_path.read_text(), client)
    else:
        print("  WARNING: rubric.md not found.")

    finalize(scored)

    # --- Step 4: checkpoint ---
    _print_checkpoint(scored)
    _print_public_comment(pc_tally)
    if no_checkpoint:
        # Unattended mode, for backfilling past meetings in bulk. The review
        # step is the point of the checkpoint, so this is only appropriate when
        # the artifact you want is the saved score/public-comment JSON rather
        # than a recap anyone will publish.
        print("(--no-checkpoint: accepting this ranking unreviewed)\n")
        active = scored
    else:
        active, top_n = _checkpoint_interaction(scored, default_top=top_n)
    top_items = active[:top_n]
    print(f"\nProceeding with top {len(top_items)} items:")
    for s in top_items:
        print(f"  {s.rank}. [{s.item.number}] {s.item.title[:60]}")
    print()

    # --- Step 5: write (preview only) ---
    # The recap needs no drafting step: every line of it is either quoted from
    # the agenda or counted from parsed data.
    the_draft: Draft | None = None
    if not recap:
        attachment_paths = _fetch_selected_attachments(top_items)
        print("\nDrafting newsletter with Claude...")
        feedback_text: str | None = None
        try:
            raw_feedback = input("Any reviewer notes for Claude? (Enter to skip) > ").strip()
            if raw_feedback:
                feedback_text = raw_feedback
        except (EOFError, KeyboardInterrupt):
            print()

        the_draft = draft(
            top_items,
            attachment_paths=attachment_paths or None,
            feedback=feedback_text,
            client=client,
        )
        for note in validate_draft(the_draft, top_items):
            print(f"  ! {note}")

    # --- Step 6: render + save ---
    prefix = "recap" if recap else "newsletter"
    score_path = OUT_DIR / f"{'recap-score' if recap else 'score'}-{date}.json"
    newsletter_path = OUT_DIR / f"{prefix}-{date}.md"
    written = [score_path]

    score_path.write_text(json.dumps([dataclasses.asdict(s) for s in active], indent=2))
    if recap:
        body = render_recap(
            result.logistics,
            top_items,
            all_items,
            result.action_items,
            result.consent_agenda,
            meeting_unique=meta.get("unique"),
            meeting_video=meeting_url,
            work_session_video=work_session_url,
            public_comment=pc_tally,
        )
    else:
        draft_path = OUT_DIR / f"draft-{date}.json"
        draft_path.write_text(json.dumps(dataclasses.asdict(the_draft), indent=2))
        written.append(draft_path)
        body = render_newsletter(
            result.logistics,
            top_items,
            the_draft,
            meeting_unique=meta.get("unique"),
        )
    newsletter_path.write_text(body)
    written.append(newsletter_path)

    print()
    for path in written:
        print(f"Wrote {path.relative_to(PROJECT_DIR)}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--date",
        help="meeting numberdate, e.g. 20260519, or a fixture key when a "
        "meeting has more than one agenda document, e.g. 20260818-regular "
        "(default: newest fixture)",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="preview an UPCOMING meeting (the old default); recap is now the default",
    )
    parser.add_argument(
        "--work-session",
        metavar="URL",
        help="YouTube URL for the work-session video (preview discussion signal; "
        "in a recap, used to deep-link each item to where it was previewed)",
    )
    parser.add_argument(
        "--meeting",
        metavar="URL",
        help="YouTube URL for the meeting video (recap discussion + public-comment signals)",
    )
    parser.add_argument(
        "--livestream",
        metavar="URL",
        help="meeting livestream URL for the newsletter footer",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="deterministic signals only; no API calls, no cost",
    )
    parser.add_argument(
        "--no-checkpoint",
        action="store_true",
        help="skip the interactive review and accept the ranking as scored "
        "(for unattended backfill runs; not for anything you intend to publish)",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=None,
        help="top-N items to draft (adjustable at checkpoint; default 3 for recap, 5 for preview)",
    )
    parser.add_argument(
        "--period",
        metavar="YYYYMM",
        help="recap EVERY meeting in this month as one newsletter, with each "
        "item's work-session review folded into the vote it became",
    )
    parser.add_argument(
        "--video",
        metavar="NUMBERDATE=URL",
        action="append",
        default=[],
        help="video for one meeting in a --period run, e.g. "
        "20260804=https://youtu.be/... (repeatable)",
    )
    parser.add_argument(
        "--agenda-url",
        metavar="NUMBERDATE=URL",
        action="append",
        default=[],
        help="the district's page for one meeting's agenda packet, e.g. "
        "20260804=https://... (repeatable). Diligent publishes no URL the "
        "pipeline can derive, so a meeting without one links Watch only",
    )
    args = parser.parse_args()

    if args.period:
        run_period(
            args.period,
            _period_sources(args.period, args.video, args.agenda_url),
            top_n=args.top if args.top is not None else 3,
            dry_run=args.dry_run,
            no_checkpoint=args.no_checkpoint,
        )
        return

    date, html_path, meta_path = _select_fixture(args.date)
    recap = not args.preview
    top_n = args.top if args.top is not None else (3 if recap else 5)

    run(
        date,
        html_path,
        meta_path,
        work_session_url=args.work_session,
        livestream_url=args.livestream,
        meeting_url=args.meeting,
        recap=recap,
        dry_run=args.dry_run,
        top_n=top_n,
        no_checkpoint=args.no_checkpoint,
    )


if __name__ == "__main__":
    main()
