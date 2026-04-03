"""
Layer 2: Per-account feature engineering.
Produces one feature vector per user from their posts + profile fields.

Features are split into:
  Red flags  — higher value = more bot-like
  Green flags — higher value = more human-like (RF learns negative weight)
"""

from __future__ import annotations

import re
import statistics
from collections import Counter
from datetime import datetime


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# LLM brand names that shouldn't appear in normal usernames.
# "claude" excluded — it's a common French given name. We catch "claude123"-style
# via a separate pattern requiring trailing digits.
_LLM_NAMES_RE = re.compile(
    r"\b(gpt|chatgpt|gemini|llama|mistral|bard|openai|copilot|claude\d+)\b",
    re.IGNORECASE,
)

_FIRST_PERSON_RE = re.compile(
    r"\b(i|me|my|mine|myself|we|our|ours|ourselves)\b", re.IGNORECASE
)

# Informal speech markers — presence indicates human authenticity
# EN: contractions, slang, abbreviations common on social media
# FR: internet slang, verlan-adjacent, shortened forms
_INFORMAL_RE = re.compile(
    r"\b(u|ur|lol|omg|wtf|tbh|imo|ngl|smh|brb|irl|gonna|wanna|gotta|kinda|sorta|"
    r"lemme|gimme|ya|nah|yep|yup|dude|bro|sis|fam|lowkey|highkey|deadass|nvm|idk|"
    r"ikr|omfg|lmao|lmfao|bruh|rly|tho|thx|plz|pls|haha|hahaha|wit|aight|smh|"
    r"mdr|ptdr|wesh|ouais|nan|jsp|bah|bof|ptn|ouf|chelou|relou|bg|franchement|"
    r"kiffer|lmaoo|jsuis)\b",
    re.IGNORECASE,
)
# Apostrophe contractions: don't, can't, I'm, j'ai, c'est, etc.
_CONTRACTION_RE = re.compile(r"\b\w+'\w+\b")

# Organic emphasis: 2+ letter ALL-CAPS word OR 2+ consecutive exclamation marks
_CAPS_EMPHASIS_RE = re.compile(r"\b[A-Z]{2,}\b")

# JSON-artifact pattern: bot generated tweets as a JSON array; some weren't unwrapped.
# A post that IS entirely wrapped in quotes/brackets — e.g. ["Tweet text here"] — is a bot tell.
_JSON_WRAPPED_RE = re.compile(
    r'^(?:'
    r'"\s*.{10,}\s*"'       # "...content..."
    r"|'\s*.{10,}\s*'"      # '...content...'
    r'|\\["\'].{10,}["\']\\]'  # ["..."] or ['...']
    r')$',
    re.DOTALL,
)

# Hex artifact: standalone 3-char token of 0-9/a-f that contains at least one digit AND one letter.
# These are template IDs / debug tokens left in generated text: "4a0", "3c0", "60d", "4aa".
_HEX_ARTIFACT_RE = re.compile(
    r'(?<![/\w])([0-9][0-9a-f]{1,3}[a-f]|[a-f][0-9a-f]{0,2}[0-9][0-9a-f]*)(?![/\w])',
    re.IGNORECASE,
)
_HEX_COMMON_WORDS = frozenset([
    'bad', 'bed', 'cab', 'cafe', 'dead', 'deaf', 'face', 'fade', 'feed',
    'beef', 'bead', 'bead', 'aced', 'dace', 'fad', 'ace', 'add',
])

# "Confused on the platform" bot persona: posts that discuss Twitter/social-media mechanics
# AS THE TOPIC rather than using the platform to discuss sports/music/news.
# @SirReginald73 / @SirOldEnglish / @grandpapa_uk all fit this pattern.
_PLATFORM_META_RE = re.compile(
    r"\b(tweet(s|ing|ed)?|retweet(s|ed|ing)?|twitter|trending|timeline|"
    r"hashtags?|plateforme|notification[s]?|follower[s]?|abonn[eé][s]?|"
    r"abonnement[s]?|tendances?|algorithm[e]?)\b",
    re.IGNORECASE,
)

# Engagement-farming bots paste call-to-action copy verbatim: "like and retweet to win",
# "DOIT ME SUIVRE" ("MUST FOLLOW ME"), "aimez et retweetez pour l'avoir".
_ENGAGEMENT_FARM_RE = re.compile(
    r"(like and retweet|retweet (and|to) (win|get|follow)|follow ?(for|4) ?follow|"
    r"\bf4f\b|aimez.{0,6}retweetez|retweetez pour|doit me suivre|must follow|"
    r"abonne-?toi|s.abonner pour|rt (and|pour) follow|rt pour)",
    re.IGNORECASE,
)

# Garbled character substitution: LLM output where characters were replaced by numbers/hex,
# e.g. "c74r" (cœur), "1ides" (idées), "7ao". Pattern: letter(s) + 2+ digits + letter(s).
# This is distinct from common tokens like "10am", "4th", "NBA2K" (only 1 embedded digit).
_GARBLED_CHAR_RE = re.compile(r"\b[a-zA-ZÀ-ÿ]+[0-9]{2,}[a-zA-ZÀ-ÿ]+\b")

# LLM prompt-artifact phrases: an LLM asked to "write tweets" often produces a meta-intro
# before the tweets themselves, revealing it answered the prompt rather than composing naturally.
_META_TWEET_RE = re.compile(
    r"(voici quelques|here are (my|some|a few)|here'?s (my|some|a few)|"
    r"my (recent|latest) tweets?|mes (récents?|derniers?) tweets?|"
    r"quelques-uns de mes tweets?)",
    re.IGNORECASE,
)

_STOPWORDS = {
    # English
    "the", "a", "an", "in", "on", "at", "to", "for", "of", "and", "or",
    "is", "are", "was", "were", "it", "this", "that", "with", "from",
    # French
    "le", "la", "les", "de", "du", "un", "une", "des", "et", "en",
    "au", "aux", "est", "sont", "je", "tu", "il", "elle",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_json_wrapped(t: str) -> bool:
    t = t.strip()
    if len(t) < 12:
        return False
    return bool(_JSON_WRAPPED_RE.match(t))


def _has_hex_artifact(t: str) -> bool:
    """True if the post contains a standalone hex-like token that looks like a template ID."""
    for m in _HEX_ARTIFACT_RE.finditer(t):
        token = m.group(0).lower()
        if token not in _HEX_COMMON_WORDS:
            return True
    return False


def _repeated_opening_ratio(texts: list) -> float:
    """Fraction of posts whose first 3 words appear as an opening in at least one other post.
    Template bots reuse the same opener; humans almost never do.
    """
    if len(texts) < 2:
        return 0.0
    def opening(t):
        words = t.strip().split()
        return " ".join(w.lower() for w in words[:3]) if len(words) >= 3 else None
    openings = [opening(t) for t in texts]
    counts = Counter(o for o in openings if o)
    repeated = sum(1 for o in openings if o and counts[o] > 1)
    return repeated / len(texts)


def _informality_score(text: str) -> float:
    """0–1 score: fraction of informal markers present (informal words + contractions)."""
    words = text.split()
    if not words:
        return 0.0
    informal_hits = len(_INFORMAL_RE.findall(text))
    contraction_hits = len(_CONTRACTION_RE.findall(text))
    return min(1.0, (informal_hits + contraction_hits) / len(words))


def _parse_time(ts: str) -> datetime:
    return datetime.fromisoformat(ts.replace("Z", "+00:00"))


def _gaps_seconds(sorted_posts: list) -> list:
    if len(sorted_posts) < 2:
        return []
    times = [_parse_time(p["created_at"]) for p in sorted_posts]
    return [(times[i + 1] - times[i]).total_seconds() for i in range(len(times) - 1)]


def _posting_minute_regularity(sorted_posts: list) -> float:
    """Fraction of posts sharing the most common posting minute.
    Human scheduling tools (Buffer/Hootsuite) post at a fixed minute every hour.
    LLM bots have no schedule → top minute is typically <0.15.
    """
    if len(sorted_posts) < 5:
        return 0.0
    minutes = [_parse_time(p["created_at"]).minute for p in sorted_posts]
    top_count = Counter(minutes).most_common(1)[0][1]
    return top_count / len(sorted_posts)


def _hashtag_count(text: str) -> int:
    return len(re.findall(r"#\w+", text))


def _emoji_count(text: str) -> int:
    return len(re.findall(r"[^\x00-\x7F]", text))


def _unique_word_ratio(texts: list) -> float:
    all_words = " ".join(texts).lower().split()
    if not all_words:
        return 0.0
    return len(set(all_words)) / len(all_words)


def _tokenize(text: str) -> set:
    """Lowercase word tokens, stopwords removed."""
    return {w for w in re.findall(r"[a-zA-ZÀ-ÿ]+", text.lower()) if w not in _STOPWORDS and len(w) > 2}


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    union = len(a | b)
    return len(a & b) / union if union else 0.0


def _pairwise_similarity_stats(texts: list) -> tuple:
    """Mean and std of word-Jaccard similarity between all post pairs.
    Samples first 20 posts (max 190 pairs) for efficiency.
    Returns (mean, std).
    """
    sample = texts[:20]
    sets = [_tokenize(t) for t in sample]
    sims = []
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            sims.append(_jaccard(sets[i], sets[j]))
    if not sims:
        return 0.0, 0.0
    mean = statistics.mean(sims)
    std = statistics.stdev(sims) if len(sims) > 1 else 0.0
    return mean, std


def _time_concentration(sorted_posts: list) -> float:
    """Max fraction of posts falling in any 2-hour rolling window."""
    if len(sorted_posts) < 3:
        return 0.0
    times = [_parse_time(p["created_at"]) for p in sorted_posts]
    window = 7200.0  # seconds
    max_count = 0
    for i, t_start in enumerate(times):
        count = sum(1 for t in times if 0 <= (t - t_start).total_seconds() <= window)
        max_count = max(max_count, count)
    return max_count / len(times)


def _profile_post_overlap(description: str, texts: list) -> float:
    """Jaccard overlap between description keywords and all-post keywords."""
    if not description.strip():
        return 0.0
    desc_tokens = _tokenize(description)
    post_tokens = _tokenize(" ".join(texts))
    if not desc_tokens:
        return 0.0
    return _jaccard(desc_tokens, post_tokens)


def _upper_ratio(text: str) -> float:
    alpha = [c for c in text if c.isalpha()]
    if not alpha:
        return 0.0
    return sum(1 for c in alpha if c.isupper()) / len(alpha)


def _name_username_divergence(name: str, username: str) -> float:
    """1 - Jaccard(name_words, username_words). High = profile feels fabricated."""
    name_tokens = set(re.findall(r"[a-zA-ZÀ-ÿ]+", name.lower()))
    user_tokens = set(re.findall(r"[a-zA-ZÀ-ÿ]+", username.lower()))
    if not name_tokens and not user_tokens:
        return 0.0
    return 1.0 - _jaccard(name_tokens, user_tokens)


# ---------------------------------------------------------------------------
# Main feature extractor
# ---------------------------------------------------------------------------

def extract_features(user: dict, posts: list) -> dict:
    """Return a flat dict of numeric features for one user."""
    if not posts:
        return _empty_features(user)

    sorted_posts = sorted(posts, key=lambda p: p["created_at"])
    texts = [p["text"] for p in sorted_posts]

    desc = user.get("description") or ""
    loc = user.get("location") or ""
    username = user.get("username") or ""
    name = user.get("name") or ""

    # ── Profile ─────────────────────────────────────────────────────────────
    has_description = 1.0 if desc.strip() else 0.0
    has_location = 1.0 if loc.strip() else 0.0
    username_length = float(len(username))
    z_score = float(user.get("z_score") or 0.0)
    tweet_count = float(user.get("tweet_count") or len(posts))

    # ── Temporal ────────────────────────────────────────────────────────────
    gaps = _gaps_seconds(sorted_posts)
    same_time_posts = float(len(posts) - len({p["created_at"] for p in posts}))
    mean_gap = statistics.mean(gaps) if gaps else 0.0
    std_gap = statistics.stdev(gaps) if len(gaps) > 1 else 0.0
    burstiness = std_gap / mean_gap if mean_gap > 0 else 0.0

    # ── Text style ───────────────────────────────────────────────────────────
    lengths = [len(t) for t in texts]
    avg_len = statistics.mean(lengths)
    std_len = statistics.stdev(lengths) if len(lengths) > 1 else 0.0

    hashtags_per_post = statistics.mean(_hashtag_count(t) for t in texts)
    links_per_post = statistics.mean(1.0 if "http" in t else 0.0 for t in texts)
    exclaim_per_post = statistics.mean(t.count("!") for t in texts)
    questions_per_post = statistics.mean(t.count("?") for t in texts)
    emoji_per_post = statistics.mean(_emoji_count(t) for t in texts)
    dup_rate = 1.0 - len(set(texts)) / len(texts)
    uppercase_ratio = statistics.mean(_upper_ratio(t) for t in texts)
    lexical_diversity = _unique_word_ratio(texts)

    # ── Red flags: EN behavioral blindspots ──────────────────────────────────
    # Max posts sharing exact timestamp (humans can't post 3+ in same millisecond)
    max_same_time_cluster = float(Counter(p["created_at"] for p in posts).most_common(1)[0][1])

    # Posts where stripping the URL leaves < 30 chars of actual content
    def _is_link_only(t: str) -> float:
        stripped = re.sub(r"http\S+", "", t).strip()
        return 1.0 if len(stripped) < 30 else 0.0
    link_only_post_ratio = statistics.mean(_is_link_only(t) for t in texts)

    # Hashtag mid-word: "wh#at", "#gef IT", etc.
    mid_word_hashtag_ratio = statistics.mean(
        1.0 if re.search(r"[a-zA-Z0-9]#[a-zA-Z]", t) else 0.0 for t in texts
    )

    # ── Red flags: FR behavioral blindspots ──────────────────────────────────
    # Bullet-list style posts — a FR-bot signature (~75% of FR FN bots, ~0.3% of humans)
    dash_start_ratio = statistics.mean(
        1.0 if (t.startswith("- ") or t.startswith("– ")) else 0.0 for t in texts
    )

    # Scheduled/batch automation: many posts within a 2-hour window
    time_concentration = _time_concentration(sorted_posts)

    # ── Red flags: Universal ─────────────────────────────────────────────────
    # LLM brand name in username or display name
    combined_identity = f"{username} {name}"
    llm_name_score = 1.0 if _LLM_NAMES_RE.search(combined_identity) else 0.0

    # Name is mostly numeric — template artifact like "5a5499476"
    if name:
        numeric_name_score = sum(c.isdigit() for c in name) / len(name)
    else:
        numeric_name_score = 0.0

    # ── Green flags: Semantic coherence / Human indicators ───────────────────
    # Personal language (I/me/my/we) — bots tend to speak impersonally
    first_person_ratio = statistics.mean(
        1.0 if _FIRST_PERSON_RE.search(t) else 0.0 for t in texts
    )

    # Posts starting with @mention = genuine conversation reply
    reply_mention_ratio = statistics.mean(
        1.0 if t.startswith("@") else 0.0 for t in texts
    )

    # Profile description keyword overlap with posts (coherent persona = human)
    profile_post_overlap = _profile_post_overlap(desc, texts)

    # Pairwise post similarity: high mean + low std = uniformly templated bot
    post_sim_mean, post_sim_std = _pairwise_similarity_stats(texts)

    # Name and username share no words = possibly fabricated persona
    name_username_divergence = _name_username_divergence(name, username)

    # ── Red flags: Generation artifacts (new) ───────────────────────────────
    # Posts discussing the posting platform as the topic ("How do I find my tweets?")
    platform_meta_ratio = statistics.mean(
        1.0 if _PLATFORM_META_RE.search(t) else 0.0 for t in texts
    )

    # Template structure: \n\n inside a post = formatted template output (betting picks, lists)
    double_newline_ratio = statistics.mean(1.0 if "\n\n" in t else 0.0 for t in texts)

    # Engagement farming: copy-pasted call-to-action text
    engagement_farm_ratio = statistics.mean(
        1.0 if _ENGAGEMENT_FARM_RE.search(t) else 0.0 for t in texts
    )

    # Character substitution artifacts: tokens like "c74r", "1ides" where chars became numbers
    garbled_char_ratio = statistics.mean(
        1.0 if _GARBLED_CHAR_RE.search(t) else 0.0 for t in texts
    )

    # ── Red flags: Template generation artifacts ────────────────────────────
    # Posts entirely wrapped in quotes/brackets = JSON list not properly parsed
    quoted_post_ratio = statistics.mean(1.0 if _is_json_wrapped(t) else 0.0 for t in texts)

    # Hex-like tokens in post text = template IDs / debug tokens leaked in generation
    hex_artifact_ratio = statistics.mean(1.0 if _has_hex_artifact(t) else 0.0 for t in texts)

    # ── Red flags: LLM meta-artifacts ───────────────────────────────────────
    # Posts that reference the act of tweeting ("Voici quelques-uns de mes tweets")
    # reveal the LLM was answering a prompt rather than composing organically.
    meta_tweet_score = statistics.mean(
        1.0 if _META_TWEET_RE.search(t) else 0.0 for t in texts
    )

    # Repeated opening phrase: template bots reuse openers, humans don't
    repeated_opening_ratio = _repeated_opening_ratio(texts)

    # ── Green flags: Human authenticity markers ──────────────────────────────
    # Description containing "bot" = human-declared automation (e.g. quote-bot, poll-bot)
    # LLM fake bots don't usually self-identify this way.
    self_bot_description = 1.0 if re.search(r"\bbot\b", desc, re.IGNORECASE) else 0.0

    # Informal speech: slang/contractions/shorthand — bots write too cleanly
    informality_scores = [_informality_score(t) for t in texts]
    informal_ratio = statistics.mean(informality_scores)

    # Organic emphasis: CAPS words or !! — spontaneous human enthusiasm
    uppercase_exclamation_ratio = statistics.mean(
        1.0 if (_CAPS_EMPHASIS_RE.search(t) or t.count("!") >= 2) else 0.0 for t in texts
    )

    # Register variance: humans are inconsistent in formality across posts
    register_variance = statistics.stdev(informality_scores) if len(informality_scores) > 1 else 0.0

    # Posting-minute regularity: human schedulers post at a fixed minute every hour
    posting_minute_regularity = _posting_minute_regularity(sorted_posts)

    return {
        # ── Original features ────────────────────────────
        "has_description": has_description,
        "has_location": has_location,
        "username_length": username_length,
        "z_score": z_score,
        "tweet_count": tweet_count,
        "same_time_posts": same_time_posts,
        "mean_gap": mean_gap,
        "std_gap": std_gap,
        "burstiness": burstiness,
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
        # ── Red flags: EN ────────────────────────────────
        "max_same_time_cluster": max_same_time_cluster,
        "link_only_post_ratio": link_only_post_ratio,
        "mid_word_hashtag_ratio": mid_word_hashtag_ratio,
        # ── Red flags: FR ────────────────────────────────
        "dash_start_ratio": dash_start_ratio,
        "time_concentration": time_concentration,
        # ── Red flags: Universal ─────────────────────────
        "llm_name_score": llm_name_score,
        "numeric_name_score": numeric_name_score,
        # ── Green flags ──────────────────────────────────
        "first_person_ratio": first_person_ratio,
        "reply_mention_ratio": reply_mention_ratio,
        "profile_post_overlap": profile_post_overlap,
        "post_sim_mean": post_sim_mean,
        "post_sim_std": post_sim_std,
        "name_username_divergence": name_username_divergence,
        # ── Red flags: Generation artifacts ──────────────
        "platform_meta_ratio": platform_meta_ratio,
        "double_newline_ratio": double_newline_ratio,
        "engagement_farm_ratio": engagement_farm_ratio,
        "garbled_char_ratio": garbled_char_ratio,
        # ── Red flags: Template artifacts ────────────────
        "quoted_post_ratio": quoted_post_ratio,
        "hex_artifact_ratio": hex_artifact_ratio,
        # ── Red flags: LLM meta-artifacts ────────────────
        "meta_tweet_score": meta_tweet_score,
        "repeated_opening_ratio": repeated_opening_ratio,
        # ── Green flags: Human authenticity ──────────────
        "self_bot_description": self_bot_description,
        "informal_ratio": informal_ratio,
        "uppercase_exclamation_ratio": uppercase_exclamation_ratio,
        "register_variance": register_variance,
        "posting_minute_regularity": posting_minute_regularity,
    }


def _empty_features(user: dict) -> dict:
    return {k: 0.0 for k in [
        # original
        "has_description", "has_location", "username_length", "z_score", "tweet_count",
        "same_time_posts", "mean_gap", "std_gap", "burstiness",
        "avg_len", "std_len", "hashtags_per_post", "links_per_post",
        "exclaim_per_post", "questions_per_post", "emoji_per_post",
        "dup_rate", "uppercase_ratio", "lexical_diversity",
        # red flags EN
        "max_same_time_cluster", "link_only_post_ratio", "mid_word_hashtag_ratio",
        # red flags FR
        "dash_start_ratio", "time_concentration",
        # red flags universal
        "llm_name_score", "numeric_name_score",
        # green flags
        "first_person_ratio", "reply_mention_ratio", "profile_post_overlap",
        "post_sim_mean", "post_sim_std", "name_username_divergence",
        # red flags: generation artifacts
        "platform_meta_ratio", "double_newline_ratio",
        "engagement_farm_ratio", "garbled_char_ratio",
        # red flags: template artifacts
        "quoted_post_ratio", "hex_artifact_ratio",
        # red flags: LLM meta-artifacts
        "meta_tweet_score", "repeated_opening_ratio",
        # green flags: human authenticity
        "self_bot_description", "informal_ratio", "uppercase_exclamation_ratio", "register_variance",
        "posting_minute_regularity",
    ]}


FEATURE_NAMES = list(_empty_features({}).keys())
