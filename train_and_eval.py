#!/usr/bin/env python3
"""
Cross-validation across all 6 practice datasets.
Strategy: leave-one-dataset-out within each language group.
  EN: datasets 1, 3, 5
  FR: datasets 2, 4, 6

For each fold, train on 2 datasets, validate on 1, sweep threshold,
report competition score. Then print feature importances.
"""

import os
import sys

from bot_detector.model import BotDetector, load_dataset, load_bots
from bot_detector.evaluate import competition_score, sweep_threshold
from bot_detector.coordination import compute_cross_dup_counts
from bot_detector.model import _build_user_posts

BASE = os.path.dirname(os.path.abspath(__file__))


def dataset_paths(idx: int):
    ds = os.path.join(BASE, f"dataset.posts&users.{idx}.json")
    bots = os.path.join(BASE, f"dataset.bots.{idx}.txt")
    return ds, bots


EN_INDICES = [1, 3, 5, 30]
FR_INDICES = [2, 4, 6, 31]


def run_cv(lang_indices: list[int], lang: str, use_xgb: bool = False):
    print(f"\n{'='*55}")
    print(f"  Leave-one-out CV  —  {lang.upper()}")
    print(f"{'='*55}")

    total_score = 0
    total_tp = total_fn = total_fp = 0
    best_thresholds = []

    for val_idx in lang_indices:
        train_indices = [i for i in lang_indices if i != val_idx]

        # Load train
        train_data = []
        for idx in train_indices:
            ds_path, bots_path = dataset_paths(idx)
            _, users, posts = load_dataset(ds_path)
            bot_ids = load_bots(bots_path)
            train_data.append((users, posts, bot_ids))

        # Load val
        val_ds, val_bots_path = dataset_paths(val_idx)
        _, val_users, val_posts = load_dataset(val_ds)
        val_bot_ids = load_bots(val_bots_path)
        all_val_users = {u["id"] for u in val_users}

        # Train
        detector = BotDetector(use_xgb=use_xgb)
        detector.fit(train_data)

        # Score sweep on validation set
        scores = detector.predict_scores(val_users, val_posts)
        best_thresh, best_result = sweep_threshold(scores, val_bot_ids, all_val_users)

        best_thresholds.append(best_thresh)
        total_score += best_result["score"]
        total_tp += best_result["tp"]
        total_fn += best_result["fn"]
        total_fp += best_result["fp"]

        print(
            f"  Val dataset {val_idx}: "
            f"thresh={best_thresh:.2f}  "
            f"TP={best_result['tp']}  FN={best_result['fn']}  FP={best_result['fp']}  "
            f"score={best_result['score']:+d}"
        )

    avg_thresh = sum(best_thresholds) / len(best_thresholds)
    print(f"  {'─'*50}")
    print(
        f"  TOTAL: TP={total_tp} FN={total_fn} FP={total_fp} "
        f"score={total_score:+d}  avg_thresh={avg_thresh:.2f}"
    )
    return avg_thresh


def print_feature_importances(lang_indices: list[int], lang: str, threshold: float, use_xgb: bool = False):
    print(f"\n  Feature importances ({lang.upper()}, trained on all {lang} data):")
    all_data = []
    for idx in lang_indices:
        ds_path, bots_path = dataset_paths(idx)
        _, users, posts = load_dataset(ds_path)
        bot_ids = load_bots(bots_path)
        all_data.append((users, posts, bot_ids))

    detector = BotDetector(use_xgb=use_xgb)
    detector.fit(all_data)
    detector.threshold = threshold
    for feat, imp in detector.feature_importances()[:10]:
        print(f"    {feat:<25} {imp:.4f}")


def main():
    en_thresh = run_cv(EN_INDICES, "en", use_xgb=True)
    fr_thresh = run_cv(FR_INDICES, "fr", use_xgb=False)

    print_feature_importances(EN_INDICES, "en", en_thresh, use_xgb=True)
    print_feature_importances(FR_INDICES, "fr", fr_thresh, use_xgb=False)

    print(f"\nRecommended thresholds: EN={en_thresh:.2f}  FR={fr_thresh:.2f}")


if __name__ == "__main__":
    main()
