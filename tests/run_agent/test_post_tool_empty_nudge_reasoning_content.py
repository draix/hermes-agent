"""Regression tests for issue #21811.

The post-tool empty-response nudge previously fired even when the upstream
provider had split the model's chain-of-thought into a separate
``reasoning_content`` / ``reasoning`` channel (Ollama qwen3.x PARSER,
DeepSeek-R1, Moonshot, Novita, etc.). In that case ``content`` is empty
and there is no inline ``<think>`` tag (the parser already stripped it),
so the old detection wrongly classified the response as silent and
triggered a wasteful retry round-trip.

These tests pin down the new behaviour:

* ``_has_separate_reasoning`` recognises reasoning_content / reasoning on
  both pydantic-style attribute objects and plain dict payloads, on
  ``model_extra`` fall-throughs, and ignores empty values.
* Genuinely empty messages (no inline tag, no separate reasoning) still
  trigger the nudge — no regression.
* Inline ``<think>`` blocks in content keep their existing detection
  path (``_has_inline_thinking``).
* When inline thinking AND separate reasoning are both present, both
  guards are true (and the nudge is skipped).
"""

from types import SimpleNamespace

import pytest

import run_agent
from run_agent import _has_separate_reasoning


# --------------------------------------------------------------------------- #
# _has_separate_reasoning unit tests                                          #
# --------------------------------------------------------------------------- #


class _PydanticLikeMessage(SimpleNamespace):
    """Stand-in for openai.types.chat.ChatCompletionMessage which exposes
    provider-extra fields (e.g. ``reasoning_content``) via attributes and
    via ``model_extra``."""


def test_has_separate_reasoning_attr_reasoning_content():
    msg = _PydanticLikeMessage(
        content="",
        reasoning_content="Thought: the user wants me to save…",
        reasoning=None,
    )
    assert _has_separate_reasoning(msg) is True


def test_has_separate_reasoning_attr_reasoning():
    msg = _PydanticLikeMessage(
        content="",
        reasoning_content=None,
        reasoning="thinking through the steps",
    )
    assert _has_separate_reasoning(msg) is True


def test_has_separate_reasoning_model_extra_fallback():
    """OpenAI SDK hides unknown provider fields under ``model_extra`` —
    we still need to detect them."""
    msg = SimpleNamespace(
        content="",
        model_extra={"reasoning_content": "hidden in model_extra"},
    )
    assert _has_separate_reasoning(msg) is True


def test_has_separate_reasoning_dict_payload():
    msg = {"content": "", "reasoning_content": "raw dict reasoning"}
    assert _has_separate_reasoning(msg) is True


def test_has_separate_reasoning_empty_string_is_falsy():
    msg = _PydanticLikeMessage(
        content="",
        reasoning_content="",
        reasoning="",
    )
    assert _has_separate_reasoning(msg) is False


def test_has_separate_reasoning_none_message():
    assert _has_separate_reasoning(None) is False


def test_has_separate_reasoning_no_reasoning_fields():
    msg = _PydanticLikeMessage(content="actual answer")
    assert _has_separate_reasoning(msg) is False


def test_has_separate_reasoning_inline_think_only_does_not_count():
    """Inline ``<think>`` lives in ``content``; this helper only checks
    the separate-channel fields. Inline thinking has its own detector
    (``_has_inline_thinking``)."""
    msg = _PydanticLikeMessage(
        content="<think>reasoning here</think>",
        reasoning_content=None,
        reasoning=None,
    )
    assert _has_separate_reasoning(msg) is False


def test_has_separate_reasoning_both_inline_and_separate():
    """When inline thinking AND separate reasoning coexist (rare, but
    possible when a parser only partially splits), the helper still
    reports the separate channel — and the nudge guard short-circuits
    on either signal."""
    msg = _PydanticLikeMessage(
        content="<think>some inline</think>",
        reasoning_content="and also separate",
    )
    assert _has_separate_reasoning(msg) is True


# --------------------------------------------------------------------------- #
# Source-level guard test                                                     #
#                                                                             #
# The nudge decision lives deep in the agent loop. Rather than spin up a      #
# full integration harness, pin the guard wiring at the source level so a    #
# future refactor can't silently drop it.                                     #
# --------------------------------------------------------------------------- #


def _read_agent_source() -> str:
    import inspect
    return inspect.getsource(run_agent)


def test_nudge_guard_uses_has_separate_reasoning():
    src = _read_agent_source()
    # Helper must exist and be the one we wired in.
    assert "def _has_separate_reasoning(" in src
    # The post-tool empty nudge condition must consult it.
    # We look for the conjunction with `_has_separate_reasoning_channel`
    # — the local variable bound from the helper inside the loop.
    assert "_has_separate_reasoning_channel = _has_separate_reasoning(" in src
    assert "and not _has_separate_reasoning_channel" in src


def test_nudge_guard_keeps_inline_thinking_check():
    """Regression: don't accidentally remove the existing inline-think
    guard while adding the new one."""
    src = _read_agent_source()
    assert "_has_inline_thinking = bool(" in src
    assert "and not _has_inline_thinking" in src


def test_nudge_guard_emits_status_string_unchanged():
    """The user-visible nudge status string is documented in the issue
    and external tooling may grep for it. Pin it."""
    src = _read_agent_source()
    assert "Model returned empty after tool calls" in src
