"""Disk cache for Claude calls, keyed by the exact request that produced them.

Every LLM call in this pipeline is a *measuring instrument*: a pinned model
reading a fixed source and returning schema-checked JSON. Run twice on the same
transcript with the same prompt and the answer should be the same — so paying
for it twice buys nothing. Transcripts and packet PDFs were already cached in
`.cache/`; this closes the last uncached leg.

The key is a SHA-256 over the *whole* request — model, max_tokens, thinking,
output_config, system, messages — so the cache can never serve a stale answer:
edit a prompt, bump the model, or hand it a different transcript and the key
changes and the call is re-made. That precision matters for `calibrate.py`,
where a `rubric.md` edit must re-run the rubric call but has no business
re-running segmentation of transcripts that did not move.

Entries are plain JSON under `.cache/llm/`, named `<action>-<sha12>.json`, so a
suspect answer can be read (and deleted) by hand. Nothing expires; delete the
directory to force a cold run, or pass `--no-llm-cache` on any entry point.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import pathlib

import anthropic

PROJECT_DIR = pathlib.Path(__file__).resolve().parent
CACHE_DIR = PROJECT_DIR / ".cache" / "llm"

# Flipped off by --no-llm-cache. Reads are skipped; writes still happen, so a
# forced re-run leaves the cache warm and correct for the next ordinary run.
_enabled = True


def set_enabled(value: bool) -> None:
    """Enable or disable cache *reads* for this process."""
    global _enabled
    _enabled = value


def _slug(action: str) -> str:
    return "".join(c if c.isalnum() else "-" for c in action.lower()).strip("-")


def _key(request: dict) -> str:
    """Hash the full request. sort_keys makes the digest order-independent."""
    blob = json.dumps(request, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()


def _path(action: str, key: str) -> pathlib.Path:
    return CACHE_DIR / f"{_slug(action)}-{key[:12]}.json"


def text_call(
    client: anthropic.Anthropic,
    *,
    action: str,
    stream: bool = False,
    **request,
) -> str:
    """Return the text of a Claude response, from cache when we have it.

    `action` names the call in progress messages and cache filenames (e.g.
    "transcript segmentation"). `stream` routes through `client.messages.stream`
    for requests whose `max_tokens` is too large to send non-streaming; it does
    not change the cache key, because it does not change what was asked.
    Everything else in `**request` is passed to the API verbatim.
    """
    key = _key(request)
    path = _path(action, key)

    if _enabled and path.is_file():
        try:
            entry = json.loads(path.read_text())
            print(f"  Cached {action} ({key[:12]}).", flush=True)
            return entry["text"]
        except (json.JSONDecodeError, KeyError):
            # A truncated or hand-mangled entry is not worth a crash; the call
            # below rewrites it.
            print(f"  Ignoring unreadable cache entry {path.name}.", flush=True)

    print(f"  Calling Claude for {action}...", flush=True)
    try:
        if stream:
            with client.messages.stream(**request) as s:
                resp = s.get_final_message()
        else:
            resp = client.messages.create(**request)
    except anthropic.APIError as exc:
        raise SystemExit(f"{action.capitalize()} API call failed: {exc}")

    if resp.stop_reason == "refusal":
        raise SystemExit(f"Claude refused the {action} request.")
    if resp.stop_reason == "max_tokens":
        raise SystemExit(
            f"{action.capitalize()} hit the token limit; raise max_tokens."
        )

    text = next((b.text for b in resp.content if b.type == "text"), None)
    if not text:
        raise SystemExit(f"{action.capitalize()} returned no text.")

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "action": action,
        "key": key,
        "model": request.get("model"),
        "created": dt.datetime.now().isoformat(timespec="seconds"),
        "text": text,
    }, indent=2))

    return text
