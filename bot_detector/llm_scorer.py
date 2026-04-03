"""
GPT-4o scoring layer for borderline accounts.

Response format enforced in prompt:
  [[[YES|0.92|posts follow a templated structure with no personal voice]]]
  [[[NO|0.08|account shows genuine personal anecdotes and replies]]]

Conservative rule:
  - LLM YES (confidence >= 0.75) + RF score >= 0.25 → score = max(rf, confidence * 0.9)
  - LLM NO (confidence >= 0.80) → score = max(0, rf_score - 0.10)
  - Parse failure after 3 retries → abstain (keep RF score)
  - LLM never applied to RF score < 0.25 (prevents hallucination-driven FPs)
"""

from __future__ import annotations

import os
import re
import time
from typing import Optional

_VERDICT_RE = re.compile(r"\[\[\[([^\]]+)\]\]\]")

SYSTEM_PROMPT = """You are an expert bot detection analyst. You will be given a social media account's profile and posts. Your job is to determine whether the account is a bot (automated program) or a real human.

After your analysis, you MUST include your verdict in this exact format on its own line:
[[[YES|confidence|brief_reason]]] if you believe it's a bot
[[[NO|confidence|brief_reason]]] if you believe it's a human

Where confidence is a decimal between 0.0 and 1.0, and brief_reason is a short explanation (under 20 words).

Example: [[[YES|0.91|posts follow identical templates with no personal voice or reactions]]]
Example: [[[NO|0.07|genuine personal anecdotes, typos, and replies to others]]]"""


def _build_prompt(user: dict, posts: list) -> str:
    profile_lines = [
        f"Username: {user.get('username', '?')}",
        f"Display name: {user.get('name', '?')}",
        f"Description: {user.get('description') or '(empty)'}",
        f"Location: {user.get('location') or '(empty)'}",
        f"Tweet count in dataset: {user.get('tweet_count', '?')}",
        f"Z-score: {user.get('z_score', '?')}",
    ]
    sorted_posts = sorted(posts, key=lambda p: p["created_at"])
    post_lines = [f"[{p['created_at']}] {p['text']}" for p in sorted_posts]
    # Cap at 40 posts to stay within token budget
    if len(post_lines) > 40:
        post_lines = post_lines[:20] + ["... (truncated) ..."] + post_lines[-10:]

    return (
        "PROFILE:\n"
        + "\n".join(profile_lines)
        + "\n\nPOSTS (chronological):\n"
        + "\n".join(post_lines)
    )


def _parse_verdict(text: str) -> Optional[tuple]:
    """
    Extract (verdict, confidence, reason) from response text.
    Returns None if pattern not found or malformed.
    """
    match = _VERDICT_RE.search(text)
    if not match:
        return None
    parts = match.group(1).split("|")
    if len(parts) < 2:
        return None
    verdict = parts[0].strip().upper()
    if verdict not in ("YES", "NO"):
        return None
    try:
        confidence = float(parts[1].strip())
    except ValueError:
        return None
    reason = parts[2].strip() if len(parts) > 2 else ""
    return verdict, confidence, reason


def score_accounts(
    accounts: list,          # list of {"user": {...}, "posts": [...]}
    rf_scores: dict,         # {user_id: rf_score}
    threshold: float = 0.42,
    low_cutoff: float = 0.25,
    model: str = "gpt-4o",
    max_retries: int = 3,
    retry_delay: float = 1.0,
) -> dict:
    """
    Run GPT-4o on borderline accounts (rf_score in [low_cutoff, threshold + 0.05]).

    Returns {user_id: {"verdict": "YES"/"NO"/"ABSTAIN", "confidence": float,
                        "reason": str, "original_score": float, "final_score": float}}
    """
    try:
        from openai import OpenAI
    except ImportError:
        print("openai package not installed. Skipping LLM scoring.")
        return {}

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("OPENAI_API_KEY not set. Skipping LLM scoring.")
        return {}

    client = OpenAI(api_key=api_key)
    results = {}

    borderline = [
        a for a in accounts
        if low_cutoff <= rf_scores.get(a["user"]["id"], 0) < threshold + 0.05
    ]

    if not borderline:
        return {}

    print(f"  LLM scoring {len(borderline)} borderline accounts...")

    for account in borderline:
        uid = account["user"]["id"]
        rf_score = rf_scores[uid]
        prompt = _build_prompt(account["user"], account["posts"])

        verdict_result = None
        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.1,
                    max_tokens=400,
                )
                text = response.choices[0].message.content or ""
                verdict_result = _parse_verdict(text)
                if verdict_result is not None:
                    break
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                else:
                    print(f"    Error on {uid[:8]}...: {e}")

        if verdict_result is None:
            results[uid] = {
                "verdict": "ABSTAIN",
                "confidence": 0.0,
                "reason": "parse failed after retries",
                "original_score": rf_score,
                "final_score": rf_score,
            }
            continue

        verdict, confidence, reason = verdict_result
        final_score = rf_score

        if verdict == "YES" and confidence >= 0.75 and rf_score >= low_cutoff:
            final_score = max(rf_score, confidence * 0.9)
        elif verdict == "NO" and confidence >= 0.80:
            final_score = max(0.0, rf_score - 0.10)

        results[uid] = {
            "verdict": verdict,
            "confidence": confidence,
            "reason": reason,
            "original_score": rf_score,
            "final_score": final_score,
        }
        username = account["user"].get("username", "?")
        print(f"    @{username:<20} rf={rf_score:.3f} → {verdict} ({confidence:.2f}) → {final_score:.3f}  [{reason[:50]}]")

    return results


def apply_llm_scores(rf_scores: dict, llm_results: dict) -> dict:
    """Merge LLM adjustments into the RF score dict."""
    updated = dict(rf_scores)
    for uid, result in llm_results.items():
        if result["verdict"] != "ABSTAIN":
            updated[uid] = result["final_score"]
    return updated
