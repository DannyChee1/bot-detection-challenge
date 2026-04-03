"""
Lightweight cross-account coordination detection.
Adds one feature: how many other users share exact post text with this user.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List


def compute_cross_dup_counts(all_user_posts: Dict[str, List[dict]]) -> Dict[str, int]:
    """
    For each user, count how many distinct other users share at least one
    identical post text with them.

    Returns: {user_id: cross_dup_count}
    """
    # text -> set of user_ids that posted it
    text_to_users = defaultdict(set)
    for uid, posts in all_user_posts.items():
        for post in posts:
            text = post["text"].strip()
            if text:
                text_to_users[text].add(uid)

    # For each user, count distinct other users sharing any text
    cross_dup = defaultdict(int)
    for text, user_set in text_to_users.items():
        if len(user_set) > 1:
            for uid in user_set:
                cross_dup[uid] += len(user_set) - 1

    return dict(cross_dup)
