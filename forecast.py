#!/usr/bin/env python3
"""Forecast a regular meeting's recap Highlights from its work session.

A PRIVATE tool. It answers one question ahead of the meeting: which agenda items
will probably be the recap's Highlights? Nothing it produces is publishable — the
output is a reading brief for the maintainer, not a newsletter.

    python forecast.py --date 20260616 \\
        --work-session https://www.youtube.com/watch?v=RO6QbFRzdMY
    python forecast.py --date 20260616 --dry-run   # deterministic signals only
    python forecast.py --check 20260616            # backtest against the real recap

Why this is not `make_newsletter.py --preview`: that flag produces a publishing
product, with an interactive checkpoint, attachment fetching, and an LLM prose
call. This runs start-to-finish unattended on two API calls (segmentation +
rubric) and writes no prose.

How well it works, measured against the two recaps that existed when it was
written (see dev-logs/Phase 7.md):

  - rank correlation with the final recap ranking: 0.75 / 0.81 (Spearman)
  - exact top-3 overlap: 2/3 and 1/3
  - the actual top-3 landed inside the forecast's top-5 more reliably

So the "Contenders" section is not padding — it is where the accuracy lives.
Work-session discussion time keeps score.py's heaviest weight (0.40) because it
predicts final recap RANK well (+0.85 / +0.41), even though it barely predicts
meeting discussion MINUTES (-0.04): the board argues at the work session and then
votes at the meeting.
"""

import argparse
import dataclasses
import json
import os
import pathlib
import re
import sys

import anthropic

import carryforward
from titlematch import _LEAD_VERB_RE, _TITLE_MATCH_MIN
from parse import AgendaItem, Attachment, Vote, agenda_preamble, parse_agenda
from render import _attachment_label, _block_quote, _item_description, _watch_url
from score import (
    ScoredItem,
    Signals,
    _print_table,
    compute_deterministic,
    evidence_line,
    finalize,
    score_rubric,
    segment_transcript,
)
from transcript import fetch_transcript, transcript_sections, write_transcript_md
from triage import kept_items, triage

PROJECT_DIR = pathlib.Path(__file__).resolve().parent
FIXTURES_DIR = PROJECT_DIR / "wjcc-fixtures"
CACHE_DIR = PROJECT_DIR / ".cache"
OUT_DIR = PROJECT_DIR / "out"

DEFAULT_TOP = 3        # matches the recap's own --top default
CONTENDERS = 5         # ranks 4..8
# Carry-forward items shown out of rank. As tallies accumulate the matcher will
# hit more items per agenda, and an unbounded list would bury the strong signals
# among 1-speaker matches — so take the heaviest few, by turnout not by rank.
PROMOTED_MAX = 3


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


def _agenda_item(raw: dict) -> "AgendaItem":
    """Rebuild an AgendaItem from its JSON form (fixture or saved signals)."""
    fields = dict(raw)
    fields["attachments"] = [Attachment(**a) for a in fields.get("attachments") or []]
    vote = fields.get("vote")
    fields["vote"] = Vote(**vote) if vote else None
    return AgendaItem(**fields)


def _load_agenda(date: str):
    """Parse and triage the agenda fixture for `date`.

    Two fixture shapes are supported, because the source changed underneath the
    project. BoardDocs agendas are saved as HTML and parsed by `parse.py`;
    Diligent work-session packets are PDFs, pre-parsed into JSON by
    `pdfagenda.py --fixtures` (the PDF's item bodies live hundreds of pages
    apart, so re-extracting them on every run would be slow for no gain).

    The forecast runs on the agenda as it stands BEFORE the meeting, so a
    missing fixture is the expected failure, not a bug.
    """
    json_path = FIXTURES_DIR / f"agenda-{date}.json"
    html_path = FIXTURES_DIR / f"agenda-{date}.html"
    meta_path = FIXTURES_DIR / f"meeting-meta-{date}.json"
    if not meta_path.is_file() or not (json_path.is_file() or html_path.is_file()):
        raise SystemExit(
            f"No agenda fixture for {date}. For a Diligent work-session packet, "
            f"run:\n  python pdfagenda.py <packet.pdf> --target {date} --fixtures"
        )
    meta = json.loads(meta_path.read_text())

    if json_path.is_file():
        payload = json.loads(json_path.read_text())
        parsed = [_agenda_item(raw) for raw in payload["items"]]
        preamble = payload.get("preamble", "")
    else:
        html = html_path.read_text()
        parsed, preamble = parse_agenda(html), agenda_preamble(html)

    result = triage(parsed, meta, preamble)
    items = kept_items(result)
    if not items:
        raise SystemExit(f"Triage kept no substantive items for {date}.")
    return items, result, meta


# --- Rendering -------------------------------------------------------------

def _highlight_block(s: ScoredItem, work_session_url: str | None) -> list[str]:
    """One predicted-highlight entry: title, evidence, deep link, agenda text."""
    lines = [f"### [{s.item.number}] {s.item.title}", ""]
    lines += [f"*{evidence_line(s)}*", ""]

    watch = _watch_url(work_session_url, s.signals.work_session_start_seconds)
    if watch and s.signals.work_session_start_seconds is not None:
        lines += [f"**Watch:** [at the work session]({watch})", ""]

    description, trimmed = _item_description(s.item)
    lines += _block_quote(description)
    if trimmed:
        lines.append("> …")
    lines.append("")

    # A Diligent packet embeds its attachments instead of linking them, so
    # there is no URL to point at — rendering a link would produce a dead one.
    # The packet page number is the working affordance: it is seekable in the
    # PDF the reader already has open.
    if s.item.source_page:
        lines += [f"**Packet:** p. {s.item.source_page}", ""]
    if s.item.attachments:
        docs = ", ".join(
            f"[{_attachment_label(a.name)}]({a.url})" if a.url
            else _attachment_label(a.name)
            for a in s.item.attachments
        )
        lines += [f"**Documents:** {docs}", ""]
    return lines


def _provenance(date: str) -> list[str]:
    """The 'how this agenda was assembled' section, for PDF-sourced agendas.

    A work session packet holds more than the next meeting's business: some of
    its items are voted the same evening and never reach the regular meeting.
    Which ones is decided by the staff's own routing sentence, so the decision
    and its verbatim wording are printed rather than left implicit — the same
    rule carry-forward follows for its fuzzy matches.
    """
    path = FIXTURES_DIR / f"agenda-{date}.json"
    if not path.is_file():
        return []
    payload = json.loads(path.read_text())
    decisions = payload.get("decisions") or []
    if not decisions:
        return []

    lines = [
        "## How this agenda was assembled",
        "",
        f"Source: `{payload.get('source_pdf', '?')}` — the work session packet, "
        "which carries full item bodies about two weeks before the meeting. "
        "Items are carried here when the packet routes them to this meeting, in "
        "the superintendent's own words:",
        "",
    ]
    for d in decisions:
        mark = "carried" if d["kept"] else "**dropped**"
        lines.append(f"- {mark} **[{d['number']}]** {d['title']} — {d['reason']}")
    lines.append("")
    return lines


def render_forecast(
    date: str,
    scored: list[ScoredItem],
    matches: dict[str, "carryforward.CarryForward"],
    leftover: list[tuple[str, str, int]],
    *,
    work_session_url: str | None,
    top_n: int = DEFAULT_TOP,
    dry_run: bool = False,
) -> str:
    """The private forecast brief. No model-written prose outside a labeled block."""
    lines = [
        f"# Forecast: WJCC School Board, {date}",
        "",
        "*Private — not for publication. Predicts which items the recap will "
        "highlight, from the work session held before the meeting.*",
        "",
    ]
    if not any(s.item.vote for s in scored):
        lines += [
            "> **Run before the meeting.** Every accuracy figure quoted below "
            "(rank correlation 0.70 / 0.84, top-3 overlap) was measured by "
            "scoring agendas *after* the fact, when vote outcomes were already "
            "visible. Those numbers are an upper bound on what a real forecast "
            "can do, and this run is not one of them. Its own accuracy stays "
            f"unknown until `forecast.py --check {date}` can run — which needs "
            "the recap side to read a Diligent agenda first, so it is a port "
            "away, not just a meeting away.",
            "",
        ]
    if dry_run:
        lines += [
            "> **Dry run.** Discussion minutes and rubric scores are 0 — this "
            "ranking reflects only dollar amounts, vote type and attachments.",
            "",
        ]

    lines += ["## Likely highlights", ""]
    for s in scored[:top_n]:
        lines += _highlight_block(s, work_session_url)

    contenders = list(scored[top_n:top_n + CONTENDERS])
    # Items with prior turnout are pulled in wherever they ranked. The composite
    # gives 0.40 of its weight to dollar amount, vote type and off-consent status,
    # so a REPORT with no price tag starts far behind — which is exactly what
    # happened to 9.03 in June: 5 prior speakers lifted it only to rank 12, well
    # outside the list below. Raising W_CARRY_FORWARD until it cleared the top 3
    # would have meant fitting one weight to one meeting; showing the item instead
    # costs nothing and hides nothing.
    shown = {s.item.number for s in scored[:top_n]} | {s.item.number for s in contenders}
    promoted = sorted(
        (s for s in scored if s.item.number in matches and s.item.number not in shown),
        key=lambda s: -matches[s.item.number].weighted,
    )[:PROMOTED_MAX]
    if contenders or promoted:
        lines += [
            "## Contenders",
            "",
            "*Historically the recap's actual top 3 lands in this combined top "
            f"{top_n + CONTENDERS} more reliably than in the top {top_n} alone.*",
            "",
        ]
        for s in contenders:
            lines.append(
                f"{s.rank}. **[{s.item.number}] {s.item.title}** — {evidence_line(s)}"
            )
        for s in promoted:
            lines.append(
                f"{s.rank}. **[{s.item.number}] {s.item.title}** — {evidence_line(s)} "
                f"— *listed out of rank: prior public comment (see below)*"
            )
        lines.append("")

    lines += ["## Carry-forward watch", ""]
    if matches:
        lines += [
            "Turnout these topics drew at recent meetings, which is the forecast's "
            "stand-in for public comment that has not happened yet:",
            "",
        ]
        titles = {s.item.number: s.item.title for s in scored}
        for number, cf in sorted(matches.items(), key=lambda kv: -kv[1].weighted):
            lines.append(
                f"- **[{number}] {titles.get(number, '')}** — "
                f"{cf.speakers} prior speakers (weighted {cf.weighted:.1f})"
            )
            for src in cf.sources:
                lines.append(
                    f"    - {src.count} on {src.date}: {src.label} *({src.why})*"
                )
        lines.append("")
    else:
        lines += [
            "_No prior public-comment topic matched an item on this agenda._",
            "",
        ]
    if leftover:
        lines += [
            "Recent public-comment topics matching nothing on this agenda — "
            "likely sources of off-agenda comment:",
            "",
        ]
        for src_date, label, count in leftover:
            lines.append(f"- {count} on {src_date}: {label}")
        lines.append("")

    lines += [
        "## What this can't see",
        "",
        "- **Public comment at the meeting itself** (0.20 of the recap composite). "
        "This is the big one. On 2026-06-16, *9.03 Redistricting Process Update* "
        "drew 3.7 minutes of work-session discussion and became the recap's #1 on "
        "14 resident speakers. Carry-forward caught the topic — 5 residents had "
        "spoken on it in May — but only lifted it from rank 15 to 12, because the "
        "composite gives 0.40 of its weight to dollar amount and vote type and "
        "9.03 is a report with no price tag. Treat the carry-forward list as its "
        "own signal, not as a correction already priced into the ranking.",
        "- **Contested votes** (0.10). The only deterministic signal that needs a "
        "recorded vote; everything else is read from item type, title and body "
        "text, all present before the meeting.",
        "- **Vote type and consent status** (0.20 combined). On a work session "
        "packet every carried item is a *review*, not yet a vote, and the packet "
        "does not say which of them the regular meeting will batch onto its "
        "consent agenda. Both signals therefore read the same for every item, and "
        "a uniform signal contributes nothing once scores are normalized. This is "
        "0.20 of the composite that is inert here, not measured — so an item that "
        "would have been lifted for requiring an individual vote is not.",
        "- **What the dollar figure means.** The largest amount in the item text "
        "is not always the decision's price tag. *5.02* shows $7,200,000, which is "
        "the revised size of the whole Grants Fund; the board is actually acting on "
        "a $100,000 grant. The figure is shown verbatim precisely so it can be "
        "checked rather than trusted.",
        "- **Agenda drift.** Scores are normalized across whatever items are "
        "present, so items added or pulled before the meeting shift every "
        "composite. Two observations so far: for 2026-05-19 the draft and finalized "
        "agendas held the same 20 items with identical titles; and every item here "
        "except 6.01 carries an explicit staff sentence routing it to this meeting.",
        "",
    ]

    lines += _provenance(date)

    rubric_notes = [s for s in scored[:top_n + CONTENDERS] if s.signals.rubric_justification]
    if rubric_notes:
        lines += [
            "## Rubric read (model-written)",
            "",
            "*Unlike everything above, these sentences were written by Claude "
            "applying rubric.md. Weigh them accordingly.*",
            "",
        ]
        for s in rubric_notes:
            lines.append(
                f"- **[{s.item.number}]** ({s.signals.rubric_score:.0f}/5) "
                f"{s.signals.rubric_justification}"
            )
        lines.append("")

    return "\n".join(lines)


# --- Re-render from saved signals ------------------------------------------

def load_scored(path: pathlib.Path) -> list[ScoredItem]:
    """Rebuild ScoredItems from a saved forecast-score JSON.

    Segmentation and rubric scoring are the only paid steps, and both write
    their results here. Rebuilding from disk means the brief's wording and
    layout can be changed and re-rendered without buying the same answers twice.
    """
    out: list[ScoredItem] = []
    for raw in json.loads(path.read_text()):
        out.append(ScoredItem(
            item=_agenda_item(raw["item"]),
            signals=Signals(**raw["signals"]),
            normalized=raw.get("normalized", {}),
            composite=raw.get("composite", 0.0),
            rank=raw.get("rank", 0),
        ))
    return out


def render_only(
    date: str, *, work_session_url: str | None, top_n: int, dry_run: bool = False
) -> None:
    """Re-render the brief for `date` from its saved signals. No API calls.

    The dry-run suffix is carried through both filenames deliberately: signals
    saved by a dry run are all zeros, and re-rendering them into the unsuffixed
    brief would present "0.0 min discussed" as a measurement rather than as a
    step that never ran.
    """
    suffix = "-dry" if dry_run else ""
    score_path = OUT_DIR / f"forecast-score-{date}{suffix}.json"
    if not score_path.is_file():
        raise SystemExit(f"No {score_path.relative_to(PROJECT_DIR)} to render from.")
    scored = load_scored(score_path)

    items, _result, _meta = _load_agenda(date)
    # The score file caches SIGNALS; the fixture owns the agenda items. Refresh
    # the item side from the fixture so a re-render picks up item fields the
    # saved file predates, instead of silently dropping them.
    by_number = {i.number: i for i in items}
    for s in scored:
        if (fresh := by_number.get(s.item.number)) is not None:
            s.item = fresh
    tallies = carryforward.load_tallies(OUT_DIR)
    matches = carryforward.match(items, tallies, date)
    leftover = carryforward.unmatched_topics(items, tallies, date)

    body = render_forecast(
        date, scored, matches, leftover,
        work_session_url=work_session_url, top_n=top_n, dry_run=dry_run,
    )
    md_path = OUT_DIR / f"forecast-{date}{suffix}.md"
    md_path.write_text(body)
    print(f"Re-rendered {md_path.relative_to(PROJECT_DIR)} from saved signals.")


# --- Backtest --------------------------------------------------------------

def _spearman(pairs: list[tuple[int, int]]) -> float | None:
    n = len(pairs)
    if n < 3:
        return None
    d2 = sum((a - b) ** 2 for a, b in pairs)
    return 1 - 6 * d2 / (n * (n * n - 1))


# An item is titled "Review X" when the work session previews it and "Approval
# of X" when the meeting acts on it, so the leading verb has to come off before
# two rankings can be compared by title.


def _pair_rankings(forecast: list[dict], recap: list[dict]) -> dict[str, int]:
    """Map each recap item number to the forecast rank of the same item.

    Item numbers cannot carry the pairing. The forecast reads a work session
    packet, where an item is numbered in section 5; the recap reads the regular
    meeting agenda, where the same item is renumbered into the consent agenda or
    action items. For 2026-08-18 the two sets are entirely disjoint (5.02–6.01
    versus 7.03–7.11), so a number-keyed comparison reports "0 items in common"
    and reads as a failed forecast rather than a failed join.

    This is the lesson carryforward.py already learned — match on topic TEXT,
    never on item number — arriving late here because on BoardDocs the forecast
    and the recap scored the same document and numbers matched for free.

    Numbers are still tried first, so BoardDocs-era checks behave exactly as
    before; title matching only fills what is left.
    """
    by_number = {s["item"]["number"]: s["rank"] for s in forecast}
    paired: dict[str, int] = {}
    used: set[int] = set()

    for s in recap:
        number = s["item"]["number"]
        if number in by_number:
            paired[number] = by_number[number]
            used.add(by_number[number])

    remaining = [f for f in forecast if f["rank"] not in used]
    for s in recap:
        number = s["item"]["number"]
        if number in paired or not remaining:
            continue
        words = carryforward._content_words(_LEAD_VERB_RE.sub("", s["item"]["title"]))
        best, score = None, _TITLE_MATCH_MIN
        for f in remaining:
            f_words = carryforward._content_words(
                _LEAD_VERB_RE.sub("", f["item"]["title"])
            )
            if (overlap := carryforward._overlap(words, f_words)) > score:
                best, score = f, overlap
        if best is not None:
            paired[number] = best["rank"]
            remaining.remove(best)
            print(f"    paired by title: recap {number} <- forecast "
                  f"{best['item']['number']} ({score:.2f})")
    return paired


def check(date: str) -> None:
    """Compare a saved forecast ranking against the recap that actually ran.

    Every new recap adds a data point. This is how the forecast earns trust over
    time rather than asserting it.
    """
    forecast_path = OUT_DIR / f"forecast-score-{date}.json"
    recap_path = OUT_DIR / f"recap-score-{date}.json"
    for p in (forecast_path, recap_path):
        if not p.is_file():
            raise SystemExit(f"Missing {p.relative_to(PROJECT_DIR)} — cannot check {date}.")

    forecast = json.loads(forecast_path.read_text())
    recap = json.loads(recap_path.read_text())
    titles = {s["item"]["number"]: s["item"]["title"] for s in recap}

    print(f"\n{'='*72}\nFORECAST CHECK — {date}\n{'='*72}")
    f_rank = _pair_rankings(forecast, recap)
    common = [s for s in recap if s["item"]["number"] in f_rank]
    print(f"{len(common)} items in both rankings "
          f"({len(forecast)} forecast, {len(recap)} recap).")
    if not common:
        print("\n  No item could be paired — the two rankings share neither "
              "numbers nor titles. Nothing below would be meaningful.\n")
        return

    r_top3 = [s["item"]["number"] for s in recap[:3]]

    def in_top(limit: int) -> int:
        """How many of the recap's top 3 the forecast ranked within `limit`."""
        return sum(
            1 for n in r_top3 if (r := f_rank.get(n)) is not None and r <= limit
        )

    print(f"\n  top-3 overlap:            {in_top(3)}/3")
    print(f"  recap top-3 in forecast 5: {in_top(5)}/3")

    rho = _spearman([(f_rank[s["item"]["number"]], s["rank"]) for s in common])
    if rho is not None:
        print(f"  Spearman rho:             {rho:+.2f}")

    print("\n  Recap top 5, and where the forecast put them:")
    for s in recap[:5]:
        num = s["item"]["number"]
        fr = f_rank.get(num)
        move = "not in forecast" if fr is None else f"forecast #{fr}"
        print(f"    recap #{s['rank']:<3} {num:>6}  ({move:<16})  {titles[num][:44]}")
    print()


# --- Pipeline --------------------------------------------------------------

def run(
    date: str,
    *,
    work_session_url: str | None,
    dry_run: bool = False,
    top_n: int = DEFAULT_TOP,
) -> None:
    OUT_DIR.mkdir(exist_ok=True)
    print(f"\n{'='*60}\nWJCC Forecast — {date}\n{'='*60}\n")

    items, result, _meta = _load_agenda(date)
    print(f"Agenda: {len(items)} substantive items.")
    if any(i.vote for i in items):
        print(
            "  NOTE: this agenda already carries recorded votes, so it is the "
            "FINALIZED version. A live forecast would see the draft; treat the "
            "ranking as an optimistic backtest, not a true forecast."
        )

    scored = compute_deterministic(items, result)

    # Carry-forward stands in for the public comment that has not happened yet.
    tallies = carryforward.load_tallies(OUT_DIR)
    matches = carryforward.match(items, tallies, date)
    leftover = carryforward.unmatched_topics(items, tallies, date)
    carryforward.apply_to_signals(scored, matches)
    usable = [t for t in tallies if t.date < date]
    print(f"Carry-forward: {len(usable)} prior tallies, {len(matches)} matched items.")
    for number, cf in sorted(matches.items(), key=lambda kv: -kv[1].weighted):
        for src in cf.sources:
            print(f"  [{number}] <- {src.count} speakers on {src.date}: "
                  f"{src.label[:40]} ({src.why})")

    if not dry_run:
        _load_dotenv()
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise SystemExit("ANTHROPIC_API_KEY is not set (env var or .env file).")
        client = anthropic.Anthropic()

        if work_session_url:
            print("\nFetching work-session transcript...")
            snippets = fetch_transcript(work_session_url, cache_dir=CACHE_DIR)
            if snippets:
                segment_transcript(
                    scored, snippets, client, kind="work_session", score=True
                )
                write_transcript_md(
                    date, "worksession", snippets,
                    transcript_sections(scored, "work_session_start_seconds"),
                    out_dir=OUT_DIR, video_url=work_session_url,
                )
        else:
            print(
                "\n  NOTE: no --work-session URL — the strongest signal "
                "(discussion time) will be 0 for every item.\n"
            )

        rubric_path = PROJECT_DIR / "rubric.md"
        if rubric_path.is_file():
            score_rubric(scored, rubric_path.read_text(), client)
        else:
            print("  WARNING: rubric.md not found.")

    finalize(scored)
    _print_table(scored, dry_run=dry_run)

    body = render_forecast(
        date, scored, matches, leftover,
        work_session_url=work_session_url, top_n=top_n, dry_run=dry_run,
    )
    suffix = "-dry" if dry_run else ""
    md_path = OUT_DIR / f"forecast-{date}{suffix}.md"
    score_path = OUT_DIR / f"forecast-score-{date}{suffix}.json"
    md_path.write_text(body)
    score_path.write_text(json.dumps([dataclasses.asdict(s) for s in scored], indent=2))
    for p in (score_path, md_path):
        print(f"Wrote {p.relative_to(PROJECT_DIR)}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--date", help="meeting numberdate, e.g. 20260616")
    parser.add_argument(
        "--work-session", metavar="URL",
        help="YouTube URL for the work session preceding this meeting",
    )
    parser.add_argument(
        "--check", metavar="DATE",
        help="backtest a saved forecast against out/recap-score-<DATE>.json",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="deterministic signals only; no API calls, no cost",
    )
    parser.add_argument(
        "--render-only", action="store_true",
        help="re-render the brief from out/forecast-score-<date>.json; no API calls",
    )
    parser.add_argument(
        "--top", type=int, default=DEFAULT_TOP,
        help=f"how many likely highlights to detail (default {DEFAULT_TOP})",
    )
    args = parser.parse_args()

    if args.check:
        check(args.check)
        return
    if not args.date:
        parser.error("--date is required (or use --check DATE)")

    if args.render_only:
        render_only(
            args.date, work_session_url=args.work_session, top_n=args.top,
            dry_run=args.dry_run,
        )
        return

    run(
        args.date,
        work_session_url=args.work_session,
        dry_run=args.dry_run,
        top_n=args.top,
    )


if __name__ == "__main__":
    main()
