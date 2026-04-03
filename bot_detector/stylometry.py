"""
Stylometric cross-user similarity.

Instead of comparing accounts to "known training bots" (which causes data
leakage — training bots would match themselves perfectly), we compare every
user to every OTHER user in the same dataset using character-level TF-IDF.

Accounts from the same bot generator share writing style fingerprints even
when content varies: same n-gram rhythms, punctuation patterns, sentence
endings, etc.

Features produced per user:
  style_sim_max    — similarity to the most style-similar OTHER account
  style_sim_mean5  — mean similarity to the top-5 most similar other accounts
"""

from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def compute_cross_style_sims(users: list, user_posts: dict) -> dict:
    """
    Return {user_id: (max_sim, mean_top5_sim)} for all users.

    Fits a fresh TF-IDF vectorizer on the current dataset (no stored state,
    no leakage). Self-similarity is excluded (diagonal zeroed out).
    """
    if not users:
        return {}

    user_ids = [u["id"] for u in users]
    texts = []
    for uid in user_ids:
        text = " ".join(p["text"] for p in user_posts.get(uid, []))
        texts.append(text if text.strip() else " ")

    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        max_features=3000,
        sublinear_tf=True,
        min_df=1,
    )
    X = vectorizer.fit_transform(texts)
    sims = cosine_similarity(X)           # (n_users, n_users)
    np.fill_diagonal(sims, 0.0)           # exclude self

    result = {}
    for i, uid in enumerate(user_ids):
        row = sims[i]
        sorted_sims = np.sort(row)[::-1]
        max_sim = float(sorted_sims[0])
        top5 = sorted_sims[:min(5, len(sorted_sims))]
        mean_top5 = float(top5.mean())
        result[uid] = (max_sim, mean_top5)

    return result
