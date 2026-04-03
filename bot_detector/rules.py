"""
Layer 1: High-precision hard rules.
Flag immediately when near-zero false-positive indicators fire.
"""
from __future__ import annotations

from typing import Optional

PROMPT_LEAKAGE_PHRASES = [
    "you are a content generation assistant",
    "here are some of my recent tweets",
    "trained on data up to",
    "sample social media posts for research",
    "as an ai language model",
    "i am an ai",
    "i'm an ai",
    "provide sample social media",
    "you are a helpful assistant",
]


def check_prompt_leakage(posts: list[dict]) -> bool:
    """Return True if any post contains an LLM prompt-leakage phrase."""
    for post in posts:
        text_lower = post["text"].lower()
        if any(phrase in text_lower for phrase in PROMPT_LEAKAGE_PHRASES):
            return True
    return False


def apply_hard_rules(user_id: str, posts: list[dict]) -> Optional[float]:
    """
    Return 1.0 if a hard rule fires (high-confidence bot), else None.
    Returning None means the ML layer should decide.
    """
    if check_prompt_leakage(posts):
        return 1.0
    return None
