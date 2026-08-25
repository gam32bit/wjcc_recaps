"""Checks on the Claude answer cache. Run directly: `python3 test_llmcache.py`.

The cache is only safe because its key covers the *whole* request — if a prompt
edit or a model bump could ever hit a stale entry, the pipeline would quietly
publish an answer nobody asked for. These tests pin that property down, along
with the two response shapes (`create` and `stream`) the pipeline uses. No API
key needed; the client is a stub that counts calls.
"""

from __future__ import annotations

import contextlib
import pathlib
import shutil
import sys
import tempfile
import types

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import llmcache

REQUEST = {
    "model": "claude-sonnet-5",
    "max_tokens": 8192,
    "thinking": {"type": "disabled"},
    "system": "sys",
    "messages": [{"role": "user", "content": "transcript A"}],
}


def _stub(calls: list, text: str = '{"segments": []}'):
    """A client that records requests and returns `text` for both call shapes."""
    resp = types.SimpleNamespace(
        stop_reason="end_turn",
        content=[types.SimpleNamespace(type="text", text=text)],
    )

    class Client:
        class messages:
            @staticmethod
            def create(**kw):
                calls.append(kw)
                return resp

            @staticmethod
            @contextlib.contextmanager
            def stream(**kw):
                calls.append(kw)
                yield types.SimpleNamespace(get_final_message=lambda: resp)

    return Client, resp


def test_key_covers_the_whole_request(tmp: pathlib.Path) -> None:
    llmcache.CACHE_DIR = tmp / "a"
    calls: list = []
    client, _ = _stub(calls)
    call = lambda **kw: llmcache.text_call(
        client, action="transcript segmentation", **{**REQUEST, **kw}
    )

    assert call() == call() == '{"segments": []}'
    assert len(calls) == 1, "an identical repeat must come from disk"

    call(messages=[{"role": "user", "content": "transcript B"}])
    assert len(calls) == 2, "a different transcript must re-run"

    call(model="claude-opus-5")
    assert len(calls) == 3, "a different model must re-run"

    call(system="a reworded prompt")
    assert len(calls) == 4, "an edited prompt must re-run"

    # Same request, different kwarg order — the digest sorts keys, so this is
    # the *same* call and must not re-run.
    call(**{k: REQUEST[k] for k in reversed(list(REQUEST))})
    assert len(calls) == 4, "kwarg order must not change the key"


def test_no_llm_cache_reruns_but_stays_warm(tmp: pathlib.Path) -> None:
    llmcache.CACHE_DIR = tmp / "b"
    calls: list = []
    client, _ = _stub(calls)

    llmcache.text_call(client, action="rubric scoring", **REQUEST)
    llmcache.set_enabled(False)
    llmcache.text_call(client, action="rubric scoring", **REQUEST)
    assert len(calls) == 2, "--no-llm-cache must skip the read"

    llmcache.set_enabled(True)
    llmcache.text_call(client, action="rubric scoring", **REQUEST)
    assert len(calls) == 2, "--no-llm-cache must still refresh the entry"


def test_streaming_path(tmp: pathlib.Path) -> None:
    llmcache.CACHE_DIR = tmp / "c"
    calls: list = []
    client, _ = _stub(calls, '{"highlights": []}')
    req = {**REQUEST, "max_tokens": 32000, "thinking": {"type": "adaptive"}}

    a = llmcache.text_call(client, action="newsletter draft", stream=True, **req)
    b = llmcache.text_call(client, action="newsletter draft", stream=True, **req)
    assert a == b == '{"highlights": []}'
    assert len(calls) == 1, "the streamed draft must cache like any other call"
    assert "stream" not in calls[0], "`stream` is ours, not an API parameter"


def test_bad_responses_stop_the_run(tmp: pathlib.Path) -> None:
    llmcache.CACHE_DIR = tmp / "d"
    for reason, fragment in [("refusal", "refused"), ("max_tokens", "token limit")]:
        calls: list = []
        client, resp = _stub(calls)
        resp.stop_reason = reason
        try:
            llmcache.text_call(client, action="roll-call votes", **REQUEST)
        except SystemExit as exc:
            assert fragment in str(exc), f"{reason}: unhelpful message {exc!r}"
        else:
            raise AssertionError(f"stop_reason={reason} should have stopped the run")


def test_unreadable_entry_is_replaced(tmp: pathlib.Path) -> None:
    llmcache.CACHE_DIR = tmp / "e"
    calls: list = []
    client, _ = _stub(calls)

    llmcache.text_call(client, action="public-comment speakers", **REQUEST)
    entry = next(iter(llmcache.CACHE_DIR.iterdir()))
    entry.write_text("{ truncated")

    # A half-written entry must cost one re-run, not a crash.
    assert llmcache.text_call(
        client, action="public-comment speakers", **REQUEST
    ) == '{"segments": []}'
    assert len(calls) == 2


def main() -> int:
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="llmcache-test-"))
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    try:
        for test in tests:
            llmcache.set_enabled(True)
            try:
                test(tmp)
            except AssertionError as exc:
                print(f"FAIL {test.__name__}: {exc}")
                failed += 1
            else:
                print(f"ok   {test.__name__}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
