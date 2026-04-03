"""
Layer 2: Per-account feature engineering.
Produces one feature vector per user from their posts + profile fields.
"""

import re
import statistics
from datetime import datetime, timezone


def _parse_time(ts: str) -> datetime:
    return datetime.fromisoformat(ts.replace("Z", "+00:00"))


def _gaps_seconds(sorted_posts: list[dict]) -> list[float]:
    if len(sorted_posts) < 2:
        return []
    times = [_parse_time(p["created_at"]) for p in sorted_posts]
    return [(times[i + 1] - times[i]).total_seconds() for i in range(len(times) - 1)]


def _hashtag_count(text: str) -> int:
    return len(re.findall(r"#\w+", text))


def _emoji_count(text: str) -> int:
    return len(re.findall(r"[^\x00-\x7F]", text))


def _unique_word_ratio(texts: list[str]) -> float:
    all_words = " ".join(texts).lower().split()
    if not all_words:
        return 0.0
    return len(set(all_words)) / len(all_words)


def extract_features(user: dict, posts: list[dict]) -> dict:
    """
    Return a flat dict of numeric features for one user.
    `user` is the users[] entry; `posts` is all posts by this user.
    """
    if not posts:
        # Shouldn't happen (users have 10-100 posts), but be safe
        return _empty_features(user)

    sorted_posts = sorted(posts, key=lambda p: p["created_at"])
    texts = [p["text"] for p in sorted_posts]

    # --- Profile ---
    desc = user.get("description") or ""
    loc = user.get("location") or ""
    username = user.get("username") or ""

    has_description = 1.0 if desc.strip() else 0.0
    has_location = 1.0 if loc.strip() else 0.0
    username_length = float(len(username))
    z_score = float(user.get("z_score") or 0.0)
    tweet_count = float(user.get("tweet_count") or len(posts))

    # --- Temporal ---
    gaps = _gaps_seconds(sorted_posts)
    same_time_posts = float(len(posts) - len({p["created_at"] for p in posts}))
    mean_gap = statistics.mean(gaps) if gaps else 0.0
    std_gap = statistics.stdev(gaps) if len(gaps) > 1 else 0.0
    burstiness = std_gap / mean_gap if mean_gap > 0 else 0.0

    # --- Text style ---
    lengths = [len(t) for t in texts]
    avg_len = statistics.mean(lengths)
    std_len = statistics.stdev(lengths) if len(lengths) > 1 else 0.0

    hashtags_per_post = statistics.mean(_hashtag_count(t) for t in texts)
    links_per_post = statistics.mean(1.0 if "http" in t else 0.0 for t in texts)
    exclaim_per_post = statistics.mean(t.count("!") for t in texts)
    questions_per_post = statistics.mean(t.count("?") for t in texts)
    emoji_per_post = statistics.mean(_emoji_count(t) for t in texts)

    dup_rate = 1.0 - len(set(texts)) / len(texts)

    # Uppercase ratio: fraction of alpha chars that are uppercase
    def _upper_ratio(text):
        alpha = [c for c in text if c.isalpha()]
        if not alpha:
            return 0.0
        return sum(1 for c in alpha if c.isupper()) / len(alpha)

    uppercase_ratio = statistics.mean(_upper_ratio(t) for t in texts)
    lexical_diversity = _unique_word_ratio(texts)

    return {
        # profile
        "has_description": has_description,
        "has_location": has_location,
        "username_length": username_length,
        "z_score": z_score,
        "tweet_count": tweet_count,
        # temporal
        "same_time_posts": same_time_posts,
        "mean_gap": mean_gap,
        "std_gap": std_gap,
        "burstiness": burstiness,
        # text style
        "avg_len": avg_len,
        "std_len": std_len,
        "hashtags_per_post": hashtags_per_post,
        "links_per_post": links_per_post,
        "exclaim_per_post": exclaim_per_post,
        "questions_per_post": questions_per_post,
        "emoji_per_post": emoji_per_post,
        "dup_rate": dup_rate,
        "uppercase_ratio": uppercase_ratio,
        "lexical_diversity": lexical_diversity,
    }


def _empty_features(user: dict) -> dict:
    return {k: 0.0 for k in [
        "has_description", "has_location", "username_length", "z_score", "tweet_count",
        "same_time_posts", "mean_gap", "std_gap", "burstiness",
        "avg_len", "std_len", "hashtags_per_post", "links_per_post",
        "exclaim_per_post", "questions_per_post", "emoji_per_post",
        "dup_rate", "uppercase_ratio", "lexical_diversity",
    ]}


FEATURE_NAMES = list(_empty_features({}).keys())
