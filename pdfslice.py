#!/usr/bin/env python3
"""Cut one PDF per agenda item out of a Diligent packet, for linking.

Deterministic, no LLM, no network.

Diligent publishes a meeting as a single packet PDF — 1,731 pages and 48 MB for
Aug 4 2026 — with an opaque per-document `handle=` token, so the recap could
only ever say "in the agenda packet, p. 1603" and leave the reader to find it.
A `#page=` fragment against the district's own URL was tested and does not work.

So the recap links its own slice instead. For each item this writes the item's
DETAILS pages — the "Agenda Item Details" template page, and any continuation
page still carrying template labels — and nothing else:

    26 items, 1-2 pages each, 60-80 KB each, 1.75 MB for the whole packet.

The attachments are cut SEPARATELY, into `item-<n>-attachments.pdf`, so the
recap's "1 attachment" can link at the actual document rather than name a file
the reader cannot open. They stay out of the details slice for two reasons:
they are what make a packet enormous (item 5.06's playground RFP proposals run
1,446 pages), and the "read more" link under an excerpt is answering "what did
the newsletter cut from this description?", which the deck does not answer.

Because an attachment run can be enormous, one over `_MAX_ATTACHMENT_PAGES` is
not cut at all — `docs/` is committed and served by GitHub Pages. The recap
then prints "1 attachment" as plain text, which is the honest degrade.

Each slice is the district's own page, unaltered and complete — nothing is
re-typed, re-set or summarized, so a reader can check the excerpt against it
and see exactly what the newsletter left out.

    python pdfslice.py <packet.pdf> <agenda-fixture.json> --items 6.01,8.07
    python pdfslice.py <packet.pdf> <agenda-fixture.json> --all

When a highlighted vote INHERITED its attachments from a work-session review
(`merge.py` folds 5.01 into 8.07), slice the review too — the documents live on
the review's pages, so 8.07's attachment link comes out of 5.01's slice.
"""

import argparse
import json
import pathlib
import subprocess
import sys

from pdfagenda import _ANY_LABEL_RE, extract_text

PROJECT_DIR = pathlib.Path(__file__).resolve().parent
# GitHub Pages serves from docs/ on the default branch. It has to be Pages and
# not a repo URL: github.com's PDF viewer and raw.githubusercontent.com both
# render the file through their own machinery, and a reader who wants to save
# or print the page gets something other than the district's document.
DOCS_DIR = PROJECT_DIR / "docs" / "packet"

# How far past its first page an item's details are allowed to run before we
# stop looking. Two is the observed maximum across the Aug 4 and Aug 18
# packets; the cap only bounds the pdftotext calls.
_MAX_DETAIL_PAGES = 8

# An attachment run longer than this is left unsliced. `docs/` is committed and
# served by Pages, and 5.06's 1,446-page RFP response would be most of the
# repository. 6.01's redistricting deck (34 pages, the largest thing actually
# highlighted so far) sits comfortably under it.
_MAX_ATTACHMENT_PAGES = 120

# ...and a byte cap too, because pages are a poor proxy: a scanned 100-page
# deck can outweigh a thousand pages of text. Checked after the cut and the
# file removed if it is over, so the recap prints "1 attachment" unlinked.
_MAX_ATTACHMENT_BYTES = 8 * 1024 * 1024


def page_count(pdf: pathlib.Path) -> int:
    """Total pages, via `pdfinfo` (poppler, same toolchain as pdfagenda)."""
    try:
        out = subprocess.run(
            ["pdfinfo", str(pdf)], capture_output=True, text=True, check=True
        ).stdout
    except FileNotFoundError:
        raise SystemExit("pdfinfo not found — install poppler-utils.")
    return int(out.split("Pages:")[1].split()[0])


def detail_range(pdf: pathlib.Path, first: int, hard_last: int) -> tuple[int, int]:
    """The item's own pages: `first`, plus any page still carrying its labels.

    This is `parse_detail`'s rule applied a page at a time. An item's
    attachments follow it immediately and carry no "Agenda Item Details"
    header, so the only thing separating body from attachment is whether the
    page still uses the staff template (TOPIC / BACKGROUND / COST BUDGETED...).
    """
    last = first
    for page in range(first + 1, min(hard_last, first + _MAX_DETAIL_PAGES) + 1):
        if not _ANY_LABEL_RE.search(extract_text(pdf, page, page, layout=False)):
            break
        last = page
    return first, last


def slice_pages(pdf: pathlib.Path, first: int, last: int, out: pathlib.Path) -> None:
    """Write pages [first, last] of `pdf` to `out` with Ghostscript."""
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run([
            "gs", "-q", "-dNOPAUSE", "-dBATCH", "-sDEVICE=pdfwrite",
            f"-dFirstPage={first}", f"-dLastPage={last}",
            f"-sOutputFile={out}", str(pdf),
        ], check=True, capture_output=True)
    except FileNotFoundError:
        raise SystemExit("gs not found — install ghostscript.")


def slice_packet(
    pdf: pathlib.Path,
    fixture: pathlib.Path,
    numbers: list[str] | None,
    *,
    verbose: bool = True,
) -> dict[str, dict[str, str]]:
    """Write the details and attachment PDFs for each requested item.

    Returns {"items": {number: path}, "attachments": {page: path}}. The
    attachment map is keyed by the owning item's PACKET PAGE, not its number,
    because a folded review's documents travel with their page: `merge.py`
    gives the vote (8.07) its review's attachments stamped with the review's
    source_page (5.01's p.7), which is the only key both sides can agree on.

    Item numbers are the agenda's own ("6.01"), NOT the meeting-namespaced form
    `merge.py` produces — a fixture is one meeting, and the namespace belongs to
    the period that combines them.
    """
    data = json.loads(fixture.read_text())
    numberdate = data["numberdate"]
    items = sorted(
        (i for i in data["items"] if i.get("source_page")),
        key=lambda i: i["source_page"],
    )
    starts = [i["source_page"] for i in items]
    total = page_count(pdf)

    written: dict[str, str] = {}
    attachments: dict[str, str] = {}
    for n, item in enumerate(items):
        if numbers is not None and item["number"] not in numbers:
            continue
        page = item["source_page"]
        hard_last = starts[n + 1] - 1 if n + 1 < len(starts) else total
        first, last = detail_range(pdf, page, hard_last)
        rel = f"{numberdate}/item-{item['number']}.pdf"
        out = DOCS_DIR / rel
        slice_pages(pdf, first, last, out)
        written[item["number"]] = rel
        if verbose:
            pages = "1 page" if first == last else f"{last - first + 1} pages"
            print(f"  [{item['number']:>5}] p.{first}{'' if first == last else f'-{last}'}"
                  f"  {pages:>7}  {out.stat().st_size / 1024:5.0f} KB  "
                  f"{item['title'][:44]}")

        # Everything between the end of the details and the next item is this
        # item's attachments — the packet embeds them there and gives them no
        # header of their own, which is why `detail_range` has to find the seam.
        if not item.get("attachments") or last >= hard_last:
            continue
        run = hard_last - last
        if run > _MAX_ATTACHMENT_PAGES:
            if verbose:
                print(f"  [{item['number']:>5}] p.{last + 1}-{hard_last}  "
                      f"{run} pages — too large to publish, left unlinked")
            continue
        rel = f"{numberdate}/item-{item['number']}-attachments.pdf"
        out = DOCS_DIR / rel
        slice_pages(pdf, last + 1, hard_last, out)
        if out.stat().st_size > _MAX_ATTACHMENT_BYTES:
            out.unlink()
            if verbose:
                print(f"  [{item['number']:>5}] p.{last + 1}-{hard_last}  "
                      f"{run} pages — too large to publish, left unlinked")
            continue
        attachments[str(page)] = rel
        if verbose:
            print(f"  [{item['number']:>5}] p.{last + 1}-{hard_last}  "
                  f"{run:>2} pages  {out.stat().st_size / 1024:5.0f} KB  "
                  f"attachments")

    missing = set(numbers or []) - set(written)
    if missing:
        raise SystemExit(
            f"No item with a packet page for: {', '.join(sorted(missing))}"
        )
    return {"items": written, "attachments": attachments}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("pdf", type=pathlib.Path, help="the Diligent packet PDF")
    parser.add_argument("fixture", type=pathlib.Path, help="agenda-<date>.json")
    parser.add_argument(
        "--items",
        help="comma-separated item numbers to slice, e.g. 6.01,8.07 "
             "(un-namespaced — the agenda's own numbering)",
    )
    parser.add_argument("--all", action="store_true", help="slice every item")
    parser.add_argument(
        "--map", type=pathlib.Path,
        help="write {item number: path} JSON here, for render.py",
    )
    args = parser.parse_args()

    if not args.items and not args.all:
        sys.exit("Give --items 6.01,8.07 or --all.")
    for path in (args.pdf, args.fixture):
        if not path.is_file():
            sys.exit(f"No such file: {path}")

    numbers = None if args.all else [n.strip() for n in args.items.split(",")]
    written = slice_packet(args.pdf, args.fixture, numbers)
    count = len(written["items"]) + len(written["attachments"])
    print(f"\n{count} slice(s) under {DOCS_DIR.relative_to(PROJECT_DIR)}/")
    if args.map:
        args.map.write_text(json.dumps(written, indent=2))
        print(f"Wrote {args.map}")


if __name__ == "__main__":
    main()
