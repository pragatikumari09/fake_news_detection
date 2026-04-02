"""
config.py — Central configuration for the Fake News Detection System.
All paths, hyperparameters, and constants live here for easy reproducibility.
"""

import os
from pathlib import Path

# ── Reproducibility ──────────────────────────────────────────────────────────
RANDOM_SEED = 42

# ── Project Paths ─────────────────────────────────────────────────────────────
BASE_DIR        = Path(__file__).parent
DATA_DIR        = BASE_DIR / "data"
MODEL_DIR       = BASE_DIR / "models"
REPORT_DIR      = BASE_DIR / "reports"
VIZ_DIR         = BASE_DIR / "visualizations"
SRC_DIR         = BASE_DIR / "src"

for d in [DATA_DIR, MODEL_DIR, REPORT_DIR, VIZ_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Dataset ───────────────────────────────────────────────────────────────────
# Kaggle "Fake and Real News" dataset files (place them in data/)
FAKE_CSV  = DATA_DIR / "Fake.csv"
REAL_CSV  = DATA_DIR / "True.csv"
MERGED_CSV = DATA_DIR / "merged_dataset.csv"

# ── Data Splits ───────────────────────────────────────────────────────────────
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15

# ── Preprocessing ─────────────────────────────────────────────────────────────
MAX_FEATURES   = 50_000     # TF-IDF vocabulary size
MAX_SEQ_LEN    = 512        # For transformer / LSTM tokenisation
MIN_WORD_LEN   = 2          # Discard tokens shorter than this
NGRAM_RANGE    = (1, 2)     # Unigrams + bigrams for TF-IDF

# ── Sensational / Clickbait keywords ─────────────────────────────────────────
SENSATIONAL_WORDS = [
    "shocking", "unbelievable", "breaking", "exclusive", "scandal",
    "exposed", "revealed", "alert", "urgent", "bombshell", "secret",
    "conspiracy", "hoax", "fraud", "lie", "fake", "truth", "proof",
    "miracle", "danger", "warning", "crisis", "disaster"
]

# ── TF-IDF ────────────────────────────────────────────────────────────────────
TFIDF_CONFIG = dict(
    max_features   = MAX_FEATURES,
    ngram_range    = NGRAM_RANGE,
    sublinear_tf   = True,
    min_df         = 3,
    max_df         = 0.90,
    analyzer       = "word",
)

# ── Traditional ML Hyper-parameters ──────────────────────────────────────────
LR_PARAMS = dict(
    C             = 5.0,
    max_iter      = 1000,
    solver        = "lbfgs",
    class_weight  = "balanced",
    random_state  = RANDOM_SEED,
)

NB_PARAMS = dict(alpha=0.1)          # Complement Naive Bayes

RF_PARAMS = dict(
    n_estimators  = 300,
    max_depth     = None,
    class_weight  = "balanced",
    n_jobs        = -1,
    random_state  = RANDOM_SEED,
)

# ── LSTM hyper-parameters ─────────────────────────────────────────────────────
LSTM_CONFIG = dict(
    vocab_size      = 30_000,
    embed_dim       = 128,
    hidden_dim      = 256,
    num_layers      = 2,
    dropout         = 0.4,
    bidirectional   = True,
    batch_size      = 64,
    epochs          = 10,
    lr              = 1e-3,
    clip_grad       = 1.0,
)

# ── BERT hyper-parameters (fine-tuning) ──────────────────────────────────────
BERT_MODEL_NAME = "distilbert-base-uncased"   # lighter than full BERT
BERT_CONFIG = dict(
    max_len        = 256,
    batch_size     = 16,
    epochs         = 3,
    lr             = 2e-5,
    warmup_ratio   = 0.1,
    weight_decay   = 0.01,
)

# ── Evaluation ────────────────────────────────────────────────────────────────
METRICS = ["accuracy", "precision", "recall", "f1"]

# ── Streamlit App ─────────────────────────────────────────────────────────────
APP_TITLE   = "🔍 Fake News Detection System"
APP_FAVICON = "🔍"
DEFAULT_MODEL = "Logistic Regression"   # shown on first load
