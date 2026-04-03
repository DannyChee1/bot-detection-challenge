#!/usr/bin/env python3
"""
Final evaluation script.

Usage:
    python run_detection.py <dataset_path> <team_name> [--threshold 0.62]

Example:
    python run_detection.py /path/to/dataset.posts&users.eval.json myteam

The script:
  1. Detects language from dataset metadata
  2. Trains on all practice datasets of that language
  3. Runs detection on the evaluation dataset
  4. Writes <team_name>.detections.<lang>.txt

Thresholds found by CV:
  EN: set via --threshold (default 0.60) or use CV output
  FR: same
"""

import argparse
import json
import os
import sys

from bot_detector.model import BotDetector, load_dataset, load_bots, _build_user_posts
from bot_detector.evaluate import sweep_threshold
from bot_detector.llm_scorer import score_accounts, apply_llm_scores

BASE = os.path.dirname(os.path.abspath(__file__))

# Default thresholds from leave-one-out CV on practice datasets (v3 features)
DEFAULT_THRESHOLDS = {"en": 0.50, "fr": 0.43}

EN_TRAIN = [(1, "en"), (3, "en"), (5, "en"), (30, "en")]
FR_TRAIN = [(2, "fr"), (4, "fr"), (6, "fr"), (31, "fr")]


def dataset_paths(idx: int):
    ds = os.path.join(BASE, f"dataset.posts&users.{idx}.json")
    bots = os.path.join(BASE, f"dataset.bots.{idx}.txt")
    return ds, bots


def main():
    parser = argparse.ArgumentParser(description="Bot or Not — detection runner")
    parser.add_argument("dataset", help="Path to evaluation dataset JSON")
    parser.add_argument("team_name", help="Your team name (used in output filename)")
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Decision threshold (default: per-language from CV)",
    )
    parser.add_argument(
        "--lang",
        choices=["en", "fr"],
        default=None,
        help="Override language detection",
    )
    parser.add_argument(
        "--llm",
        action="store_true",
        default=False,
        help="Run GPT-4o on borderline accounts (requires OPENAI_API_KEY)",
    )
    args = parser.parse_args()

    # Load eval dataset
    meta, eval_users, eval_posts = load_dataset(args.dataset)
    lang = args.lang or meta.get("lang", "en").lower()
    print(f"Dataset language: {lang}")
    print(f"Eval users: {len(eval_users)}  posts: {len(eval_posts)}")

    # Select training datasets
    if lang == "en":
        train_indices = [idx for idx, _ in EN_TRAIN]
    else:
        train_indices = [idx for idx, _ in FR_TRAIN]

    # Load training data
    train_data = []
    for idx in train_indices:
        ds_path, bots_path = dataset_paths(idx)
        if not os.path.exists(ds_path):
            print(f"Warning: practice dataset {idx} not found, skipping", file=sys.stderr)
            continue
        _, users, posts = load_dataset(ds_path)
        bot_ids = load_bots(bots_path)
        train_data.append((users, posts, bot_ids))

    if not train_data:
        print("Error: no training data found", file=sys.stderr)
        sys.exit(1)

    print(f"Training on {len(train_data)} practice dataset(s)...")

    # Train — XGBoost for EN (better boundary precision), RF for FR (fewer FPs)
    detector = BotDetector(use_xgb=(lang == "en"))
    detector.fit(train_data)

    # Threshold
    threshold = args.threshold or DEFAULT_THRESHOLDS.get(lang, 0.60)
    detector.threshold = threshold
    print(f"Using threshold: {threshold}")

    # Predict (RF + chain-ban post-processing already applied inside predict_scores)
    scores = detector.predict_scores(eval_users, eval_posts)

    # Optional LLM layer for borderline accounts
    if args.llm:
        user_posts_map = _build_user_posts(eval_posts)
        users_map = {u["id"]: u for u in eval_users}
        accounts = [
            {"user": users_map[uid], "posts": user_posts_map.get(uid, [])}
            for uid in scores
        ]
        llm_results = score_accounts(accounts, scores, threshold=threshold)
        if llm_results:
            scores = apply_llm_scores(scores, llm_results)
            print(f"  LLM adjusted {sum(1 for r in llm_results.values() if r['verdict'] != 'ABSTAIN')} accounts")

    predicted_bots = {uid for uid, s in scores.items() if s >= threshold}
    print(f"\nDetected {len(predicted_bots)} bot accounts out of {len(eval_users)} users")

    # Show borderline cases just below threshold for awareness
    borderline = {
        uid: s for uid, s in scores.items()
        if (threshold - 0.10) <= s < threshold
    }
    if borderline:
        users_map = {u["id"]: u for u in eval_users}
        print(f"\nBorderline accounts ({len(borderline)} near threshold, NOT flagged):")
        for uid, s in sorted(borderline.items(), key=lambda x: -x[1])[:10]:
            uname = users_map.get(uid, {}).get("username", uid[:8])
            print(f"  @{uname:<20} score={s:.3f}")

    # Write output
    out_filename = f"{args.team_name}.detections.{lang}.txt"
    out_path = os.path.join(BASE, out_filename)
    with open(out_path, "w") as f:
        for uid in sorted(predicted_bots):
            f.write(uid + "\n")

    print(f"\nOutput written to: {out_path}")
    print("Done.")


if __name__ == "__main__":
    main()
