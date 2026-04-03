"""
Core detection function used by both train_and_eval.py and run_detection.py.
"""

from .model import BotDetector, load_dataset, load_bots


def build_and_run(train_paths, eval_dataset_path, threshold=0.60):
    """
    Train a detector on `train_paths`, run it on `eval_dataset_path`.
    Returns (predicted_bot_ids, score_map).
    """
    train_data = []
    for ds_path, bots_path in train_paths:
        _, users, posts = load_dataset(ds_path)
        bot_ids = load_bots(bots_path)
        train_data.append((users, posts, bot_ids))

    detector = BotDetector()
    detector.fit(train_data)
    detector.threshold = threshold

    _, eval_users, eval_posts = load_dataset(eval_dataset_path)
    scores = detector.predict_scores(eval_users, eval_posts)
    predicted = {uid for uid, s in scores.items() if s >= threshold}
    return predicted, scores
