"""
Competition score computation and threshold sweep.
Score = +2 TP  -2 FN  -6 FP
"""
from __future__ import annotations

from typing import Optional


def competition_score(
    predicted_bots: set[str],
    true_bots: set[str],
    all_users: set[str],
) -> dict:
    tp = len(predicted_bots & true_bots)
    fn = len(true_bots - predicted_bots)
    fp = len(predicted_bots - true_bots)
    score = 2 * tp - 2 * fn - 6 * fp
    return {"tp": tp, "fn": fn, "fp": fp, "score": score}


def sweep_threshold(
    user_scores: dict[str, float],
    true_bots: set[str],
    all_users: set[str],
    thresholds: Optional[list] = None,
) -> tuple[float, dict]:
    """
    Try each threshold and return (best_threshold, best_result).
    `user_scores` maps user_id -> probability score in [0, 1].
    """
    if thresholds is None:
        thresholds = [t / 100 for t in range(40, 96, 1)]

    best_thresh = 0.60
    best_result = {"tp": 0, "fn": len(true_bots), "fp": 0, "score": -2 * len(true_bots)}

    for thresh in thresholds:
        predicted = {uid for uid, s in user_scores.items() if s >= thresh}
        result = competition_score(predicted, true_bots, all_users)
        if result["score"] > best_result["score"]:
            best_result = result
            best_thresh = thresh

    return best_thresh, best_result
