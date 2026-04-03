"""
Layer 2 model: Random Forest trained on per-account features.
Handles feature assembly, training, stylometry, username clustering, and prediction.
"""

import json
from collections import defaultdict

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBClassifier
    _XGB_AVAILABLE = True
except ImportError:
    _XGB_AVAILABLE = False

from .features import extract_features, FEATURE_NAMES
from .coordination import compute_cross_dup_counts
from .rules import apply_hard_rules
from .stylometry import compute_cross_style_sims
from .username_cluster import build_clusters, chain_ban_boost

# Dataset-level injected features (appended after FEATURE_NAMES)
_INJECTED = [
    "cross_dup_count",
    "style_sim_max",
    "style_sim_mean5",
    "username_cluster_size",
]


def load_dataset(path: str):
    """Return (metadata, users_list, posts_list)."""
    with open(path) as f:
        data = json.load(f)
    return data["metadata"], data["users"], data["posts"]


def load_bots(path: str):
    with open(path) as f:
        return {line.strip() for line in f if line.strip()}


def _build_user_posts(posts):
    user_posts = defaultdict(list)
    for p in posts:
        user_posts[p["author_id"]].append(p)
    return dict(user_posts)


def _build_feature_matrix(users, user_posts, cross_dups,
                           style_sims=None, cluster_sizes=None):
    """
    Build (X, user_ids).
    style_sims:    {uid: (max_sim, mean_top5)} or None
    cluster_sizes: {uid: cluster_size}         or None
    """
    n_users = len(users)
    user_ids = [u["id"] for u in users]
    rows = []
    for u in users:
        uid = u["id"]
        posts = user_posts.get(uid, [])
        feats = extract_features(u, posts)

        feats["cross_dup_count"] = float(cross_dups.get(uid, 0))

        sim_max, sim_mean5 = style_sims[uid] if style_sims and uid in style_sims else (0.0, 0.0)
        feats["style_sim_max"] = sim_max
        feats["style_sim_mean5"] = sim_mean5

        cs = cluster_sizes.get(uid, 1) if cluster_sizes else 1
        feats["username_cluster_size"] = float(cs) / max(n_users, 1)

        rows.append([feats[k] for k in FEATURE_NAMES + _INJECTED])
    return np.array(rows, dtype=np.float64), user_ids


class BotDetector:
    def __init__(self, n_estimators: int = 300, random_state: int = 42, use_xgb: bool = False):
        if use_xgb and _XGB_AVAILABLE:
            # XGBoost: better for EN (tighter decision boundaries, handles feature interactions)
            self.clf = XGBClassifier(
                n_estimators=400,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.7,
                min_child_weight=3,
                reg_alpha=0.1,
                reg_lambda=1.0,
                random_state=random_state,
                eval_metric="logloss",
                verbosity=0,
            )
        else:
            self.clf = RandomForestClassifier(
                n_estimators=n_estimators,
                max_features="sqrt",
                min_samples_leaf=3,
                random_state=random_state,
                n_jobs=-1,
            )
        self.use_xgb = use_xgb and _XGB_AVAILABLE
        self.scaler = StandardScaler()
        self.threshold = 0.60
        self.feature_names = FEATURE_NAMES + _INJECTED

    def fit(self, datasets) -> "BotDetector":
        """
        Train on multiple datasets combined.
        Each element of `datasets` is (users, posts, true_bot_ids).
        """
        X_parts, y_parts = [], []
        for users, posts, bot_ids in datasets:
            user_posts = _build_user_posts(posts)
            cross_dups = compute_cross_dup_counts(user_posts)
            style_sims = compute_cross_style_sims(users, user_posts)
            cluster_sizes = build_clusters(users)
            X, user_ids = _build_feature_matrix(
                users, user_posts, cross_dups, style_sims, cluster_sizes
            )
            y = np.array([1 if uid in bot_ids else 0 for uid in user_ids])
            X_parts.append(X)
            y_parts.append(y)

        X_all = np.vstack(X_parts)
        y_all = np.concatenate(y_parts)
        self.scaler.fit(X_all)
        if self.use_xgb:
            n_pos = int(y_all.sum())
            n_neg = len(y_all) - n_pos
            self.clf.set_params(scale_pos_weight=n_neg / max(n_pos, 1))
        self.clf.fit(self.scaler.transform(X_all), y_all)
        return self

    def predict_scores(self, users, posts) -> dict:
        """
        Return {user_id: bot_probability} for all users.
        Includes hard-rule overrides, cross-user stylometry, username cluster
        features, and chain-ban post-processing.
        """
        user_posts = _build_user_posts(posts)
        cross_dups = compute_cross_dup_counts(user_posts)
        style_sims = compute_cross_style_sims(users, user_posts)
        cluster_sizes = build_clusters(users)

        X, user_ids = _build_feature_matrix(
            users, user_posts, cross_dups, style_sims, cluster_sizes
        )
        probs = self.clf.predict_proba(self.scaler.transform(X))[:, 1]

        scores = {}
        for uid, prob in zip(user_ids, probs):
            rule_score = apply_hard_rules(uid, user_posts.get(uid, []))
            scores[uid] = rule_score if rule_score is not None else float(prob)

        # Chain-ban: boost all cluster members when any member is flagged
        scores = chain_ban_boost(scores, users, self.threshold)
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
