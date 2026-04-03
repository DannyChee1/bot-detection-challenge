"""
Layer 1: High-precision hard rules.
Flag immediately when near-zero false-positive indicators fire.
Also caps scores for high-confidence human signals.
"""
from __future__ import annotations

import re
import statistics
from collections import Counter
from datetime import datetime
from typing import Optional

# ---------------------------------------------------------------------------
# Rule 1: Prompt leakage
# ---------------------------------------------------------------------------

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


def check_prompt_leakage(posts: list) -> bool:
    """Return True if any post contains an LLM prompt-leakage phrase."""
    for post in posts:
        text_lower = post["text"].lower()
        if any(phrase in text_lower for phrase in PROMPT_LEAKAGE_PHRASES):
            return True
    return False


# ---------------------------------------------------------------------------
# Rule 2: "Confused on the platform" bot persona
#
# Bots like @SirReginald73 / @SirOldEnglish / @OldEnglishGent post almost
# exclusively about how to USE Twitter rather than any actual content.
# Signals: high fraction of posts mentioning platform mechanics (tweet,
# trending, hashtag, follower, plateforme…) AND high question rate.
#
# Safety check across all 8 datasets:
#   - Highest human platform_meta with q/post>0.20: 0.28 (@MhouTips, DS2/31)
#   - Human outliers with high platform_meta (sports promo accts like
#     @EdgeSportsApp=0.48, @Nizou66=0.58) all have q/post < 0.05.
#   - Combined threshold (>0.35 AND q/post>0.20) has zero FPs on all data seen.
# ---------------------------------------------------------------------------

_PLATFORM_META_RULE_RE = re.compile(
    r"\b(tweet(s|ing|ed)?|retweet(s|ed|ing)?|twitter|trending|timeline|"
    r"hashtags?|plateforme|notification[s]?|follower[s]?|abonn[eé][s]?|"
    r"abonnement[s]?|tendances?|algorithm[e]?)\b",
    re.IGNORECASE,
)


def check_platform_confused_bot(posts: list) -> bool:
    """Return True for the 'confused old person on Twitter' bot persona.

    Two tiers:
    - Standard (EN/FR with full platform vocab): platform_meta > 0.35 AND q > 0.20
    - Relaxed (FR conjugated forms partially miss the regex): platform_meta > 0.10 AND q > 1.5
      Verified safe: no human across all 8 training datasets has pm > 0.10 AND q > 0.57.
    """
    if len(posts) < 5:
        return False
    texts = [p["text"] for p in posts]
    platform_meta = statistics.mean(
        1.0 if _PLATFORM_META_RULE_RE.search(t) else 0.0 for t in texts
    )
    avg_questions = statistics.mean(t.count("?") for t in texts)
    if platform_meta > 0.35 and avg_questions > 0.20:
        return True
    # FR bots use conjugated forms ("tweeté", "retweeter") that partially miss
    # the regex, but their question rate is extremely high.
    if platform_meta > 0.10 and avg_questions > 1.50:
        return True
    return False


# ---------------------------------------------------------------------------
# Rule 3: Self-declared bot account (green flag — cap, don't flag)
#
# Accounts that describe themselves as bots ("rise quote bot", "poll bot") are
# almost certainly human-operated automation tools, NOT LLM fake-persona bots.
# Verified: zero confirmed bots across all 8 training datasets use "bot" in
# their description. This is a safe hard cap.
# ---------------------------------------------------------------------------


def check_self_declared_bot(user: dict) -> bool:
    """Return True if the account describes itself as a bot in its bio."""
    desc = (user.get("description") or "").lower()
    return bool(re.search(r"\bbot\b", desc))


# ---------------------------------------------------------------------------
# Rule 4: High-precision posting schedule (green flag — human scheduler)
#
# Human scheduling tools (Buffer, Hootsuite) post at a fixed minute every hour.
# A pmr > 0.90 means >90% of posts share the same minute-of-hour — an inhuman
# level of regularity that only a cron job achieves.
#
# Safety check: across all 8 training datasets, the highest bot pmr is 0.80
# (@OldChapUK, already caught by platform_confused_bot rule). No confirmed bot
# exceeds 0.85 pmr. Zero risk of creating a FN at the 0.90 threshold.
# ---------------------------------------------------------------------------

_MIN_POSTS_FOR_PMR_RULE = 10


def check_precise_scheduler(posts: list) -> bool:
    """Return True if >90% of posts share the same posting minute."""
    if len(posts) < _MIN_POSTS_FOR_PMR_RULE:
        return False
    try:
        minutes = [
            datetime.fromisoformat(p["created_at"].replace("Z", "+00:00")).minute
            for p in posts
        ]
    except (KeyError, ValueError):
        return False
    top_count = Counter(minutes).most_common(1)[0][1]
    return top_count / len(posts) > 0.90


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------


def apply_hard_rules(user_id: str, posts: list, user: Optional[dict] = None) -> Optional[float]:
    """
    Return a score override if a hard rule fires, else None (ML decides).
    High scores (>0.90) = high-confidence bot.
    Low scores (<0.20) = high-confidence human — cap the ML score.
    Bot rules are checked first so they can't be overridden by human caps.
    """
    if check_prompt_leakage(posts):
        return 1.0
    if check_platform_confused_bot(posts):
        return 0.95

    # Green flag caps: high-confidence human signals
    if user and check_self_declared_bot(user):
        return 0.10
    if check_precise_scheduler(posts):
        return 0.15

    return None
