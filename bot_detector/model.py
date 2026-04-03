"""
Layer 2 model: Random Forest trained on per-account features.
Handles feature assembly, training, and prediction.
"""

import json
from collections import defaultdict

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from .features import extract_features, FEATURE_NAMES
from .coordination import compute_cross_dup_counts
from .rules import apply_hard_rules


def load_dataset(path: str):
    """Return (metadata, users_list, posts_list)."""
    with open(path) as f:
        data = json.load(f)
    return data["metadata"], data["users"], data["posts"]


def load_bots(path: str) -> set[str]:
    with open(path) as f:
        return {line.strip() for line in f if line.strip()}


def _build_user_posts(posts):
    user_posts: dict[str, list] = defaultdict(list)
    for p in posts:
        user_posts[p["author_id"]].append(p)
    return dict(user_posts)


def _build_feature_matrix(users, user_posts, cross_dups):
    """Build (X, user_ids) where X rows correspond to user_ids order."""
    user_ids = [u["id"] for u in users]
    rows = []
    for u in users:
        uid = u["id"]
        posts = user_posts.get(uid, [])
        feats = extract_features(u, posts)
        feats["cross_dup_count"] = float(cross_dups.get(uid, 0))
        rows.append([feats[k] for k in FEATURE_NAMES + ["cross_dup_count"]])
    return np.array(rows, dtype=np.float64), user_ids


class BotDetector:
    def __init__(self, n_estimators: int = 300, random_state: int = 42):
        self.clf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_features="sqrt",
            min_samples_leaf=2,
            random_state=random_state,
            n_jobs=-1,
        )
        self.scaler = StandardScaler()
        self.threshold = 0.60
        self.feature_names = FEATURE_NAMES + ["cross_dup_count"]

    def fit(self, datasets) -> "BotDetector":
        """
        Train on multiple datasets combined.
        Each element of `datasets` is (users, posts, true_bot_ids).
        """
        X_parts, y_parts = [], []
        for users, posts, bot_ids in datasets:
            user_posts = _build_user_posts(posts)
            cross_dups = compute_cross_dup_counts(user_posts)
            X, user_ids = _build_feature_matrix(users, user_posts, cross_dups)
            y = np.array([1 if uid in bot_ids else 0 for uid in user_ids])
            X_parts.append(X)
            y_parts.append(y)

        X_all = np.vstack(X_parts)
        y_all = np.concatenate(y_parts)
        self.scaler.fit(X_all)
        self.clf.fit(self.scaler.transform(X_all), y_all)
        return self

    def predict_scores(self, users, posts):
        """
        Return {user_id: bot_probability} for all users.
        Hard rules override ML when they fire.
        """
        user_posts = _build_user_posts(posts)
        cross_dups = compute_cross_dup_counts(user_posts)
        X, user_ids = _build_feature_matrix(users, user_posts, cross_dups)
        probs = self.clf.predict_proba(self.scaler.transform(X))[:, 1]

        scores = {}
        for uid, prob in zip(user_ids, probs):
            rule_score = apply_hard_rules(uid, user_posts.get(uid, []))
            scores[uid] = rule_score if rule_score is not None else float(prob)
        return scores

    def predict(self, users, posts):
        """Return set of user_ids predicted as bots at self.threshold."""
        scores = self.predict_scores(users, posts)
        return {uid for uid, s in scores.items() if s >= self.threshold}

    def feature_importances(self):
        importances = self.clf.feature_importances_
        return sorted(
            zip(self.feature_names, importances),
            key=lambda x: x[1],
            reverse=True,
        )
