"""
Abstract base class for LLMs.
"""

import re
from abc import ABC, abstractmethod

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_TOOL_CALL_RE = re.compile(r"<tool_call>.*?</tool_call>", re.DOTALL | re.IGNORECASE)
# When skip_special_tokens=True is used during decode, <think> is stripped as a
# special token but the content and </think> remain.  This pattern removes that
# orphaned tail (everything from the start of the string up to </think>).
_ORPHAN_CLOSE_RE = re.compile(r"^.*?</think>", re.DOTALL | re.IGNORECASE)
_ORPHAN_TOOL_CLOSE_RE = re.compile(r"^.*?</tool_call>", re.DOTALL | re.IGNORECASE)


def extract_thinking(text: str) -> str:
    """Return the raw <think>…</think> and <tool_call>…</tool_call> block(s) from text, or '' if none present."""
    parts = _THINK_RE.findall(text) + _TOOL_CALL_RE.findall(text)
    if parts:
        return "\n".join(parts)
    lower = text.lower()
    if "</think>" in lower:
        m = _ORPHAN_CLOSE_RE.match(text)
        if m:
            return m.group(0)
    if "</tool_call>" in lower:
        m = _ORPHAN_TOOL_CLOSE_RE.match(text)
        if m:
            return m.group(0)
    return ""


def strip_thinking(text: str) -> str:
    """Remove <think>…</think> and <tool_call>…</tool_call> blocks emitted by reasoning models (DeepSeek-R1, Qwen3).

    Also handles the case where the <think> start token was stripped by the
    tokenizer's skip_special_tokens=True but the closing </think> and the
    reasoning content are still present in the decoded output.
    """
    text = _THINK_RE.sub("", text)
    text = _TOOL_CALL_RE.sub("", text)
    lower = text.lower()
    if "</think>" in lower:
        text = _ORPHAN_CLOSE_RE.sub("", text)
    if "</tool_call>" in text.lower():
        text = _ORPHAN_TOOL_CLOSE_RE.sub("", text)
    return text.strip()


class BaseLLM(ABC):
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate a response from the model."""
        pass

    def generate_batch(self, prompts: list, **kwargs) -> list:
        """Generate responses for a list of prompts. Override for true batching."""
        return [self.generate(p, **kwargs) for p in prompts]

    def __repr__(self):
        return f"{self.__class__.__name__}(name={self.name})"
