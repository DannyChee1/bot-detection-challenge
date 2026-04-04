# KlodKode — Bot or Not Detector

## How It Works

The detector runs three layers in sequence: hard rules, a machine learning model, and an optional GPT-4o pass for borderline accounts.

### Layer 1: Hard Rules

Before the ML model sees anything, a set of hand-crafted rules fire on high-confidence signals:

- **Prompt leakage** — post contains "as an AI language model" or similar phrases → instant bot
- **Platform-confused persona** — account posts many questions about how Twitter works ("How do I retweet?", "What is a hashtag?") with high frequency → bot score 0.95
- **Self-declared automation** — account description contains the word "bot" → capped at 0.10 (human-operated tool, not a fake persona)
- **Precise scheduler** — 90%+ of posts share the exact same minute-of-hour (e.g. every post at :02) → capped at 0.15 (Buffer/Hootsuite user, not an LLM bot)

### Layer 2: Machine Learning

We use **XGBoost for English** and **Random Forest for French** (RF had fewer false positives on FR data). Each model is trained on all available practice datasets for that language.

Each account is turned into a ~52-feature vector covering:

- **Temporal**: posting gaps, burstiness, same-millisecond clusters, time concentration
- **Text style**: avg length, hashtags/post, link ratio, duplicate rate, lexical diversity
- **Generation artifacts**: platform meta ratio, double newline ratio, hex artifact tokens, JSON-wrapped posts, engagement farming phrases, garbled character substitutions
- **LLM meta-artifacts**: repeated opening phrases, "here are my tweets" style intros
- **Green flags** (higher = more human): informality score, contractions, register variance, first-person ratio, reply mention ratio, posting minute regularity
- **Cross-dataset signals**: duplicate content across users (coordination), stylometric similarity between accounts, username clustering

The model outputs a probability score between 0 and 1.

### Layer 3: GPT-4o (borderline accounts only)

Accounts with ML scores in the borderline range (roughly 0.25–0.50 for EN, 0.25–0.48 for FR) are sent to GPT-4o for a second opinion. GPT-4o reads the full profile and up to 40 posts and returns a YES/NO verdict with a confidence score.

- LLM YES (≥0.75 confidence) → score boosted above threshold
- LLM NO (≥0.80 confidence) → score nudged down
- Accounts already clearly flagged or clearly human are not sent to the LLM

### Decision Threshold

- **English**: 0.56
- **French**: 0.43

These were selected by leave-one-out cross-validation across the 8 practice datasets, optimizing for the competition's asymmetric scoring (+2 TP, −2 FN, −6 FP).

## Running It

```bash
# English
python run_detection.py dataset.posts&users.7.json KlodKode --threshold 0.56

# French
python run_detection.py dataset.posts&users.8.json KlodKode --threshold 0.43

# With GPT-4o borderline pass (requires OPENAI_API_KEY)
python run_detection.py dataset.posts&users.7.json KlodKode --threshold 0.56 --llm
```

Output: `KlodKode.detections.en.txt` and `KlodKode.detections.fr.txt` — one user ID per line.

## English vs French Differences

| | English | French |
|---|---|---|
| Model | XGBoost | Random Forest |
| Threshold | 0.56 | 0.43 |
| Key signals | same-time clusters, link-only posts, hex artifacts | hashtags/post, quoted post ratio, stylometry |

The French detector is more conservative (lower threshold, RF instead of XGBoost) because false positives were harder to avoid on French data with a more aggressive model.
