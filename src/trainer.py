"""
src/trainer.py
--------------
Orchestrates the full training workflow:
  1. Train / validation / test split (stratified)
  2. Optional SMOTE over-sampling for class imbalance
  3. Feature pipeline fitting (TF-IDF + meta)
  4. Training all classical models + optional LSTM / BERT
  5. Cross-validation scoring
  6. Persisting all artefacts to disk
"""

import sys
import logging
from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.metrics         import make_scorer, f1_score

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    TRAIN_RATIO, VAL_RATIO, TEST_RATIO,
    RANDOM_SEED, MODEL_DIR, LSTM_CONFIG,
)
from src.feature_engineering import FeaturePipeline
from src.models              import (
    LogisticRegressionModel, NaiveBayesModel, RandomForestModel,
    get_all_classical_models,
)

logger = logging.getLogger(__name__)


# ── Data splitting ─────────────────────────────────────────────────────────────

def split_data(
    df: pd.DataFrame,
    X_text: np.ndarray,
    y: np.ndarray,
) -> dict:
    """
    Stratified 70 / 15 / 15 split.

    Returns a dict with keys:
      train_idx, val_idx, test_idx  (integer index arrays)
    and the corresponding text / label arrays.
    """
    assert abs(TRAIN_RATIO + VAL_RATIO + TEST_RATIO - 1.0) < 1e-6, \
        "Split ratios must sum to 1."

    # First split off test set
    val_test_ratio = VAL_RATIO + TEST_RATIO
    idx_all = np.arange(len(df))

    idx_train, idx_temp, y_train, y_temp = train_test_split(
        idx_all, y,
        test_size    = val_test_ratio,
        stratify     = y,
        random_state = RANDOM_SEED,
    )
    # Split val / test from the temp set
    relative_val = VAL_RATIO / val_test_ratio
    idx_val, idx_test, y_val, y_test = train_test_split(
        idx_temp, y_temp,
        test_size    = 1 - relative_val,
        stratify     = y_temp,
        random_state = RANDOM_SEED,
    )

    splits = {
        "train_idx": idx_train, "val_idx": idx_val, "test_idx": idx_test,
        "X_train": X_text[idx_train], "X_val": X_text[idx_val],   "X_test": X_text[idx_test],
        "y_train": y_train,           "y_val":  y_val,             "y_test": y_test,
        "df_train": df.iloc[idx_train].reset_index(drop=True),
        "df_val":   df.iloc[idx_val].reset_index(drop=True),
        "df_test":  df.iloc[idx_test].reset_index(drop=True),
    }
    logger.info(
        "Split → train=%d  val=%d  test=%d",
        len(idx_train), len(idx_val), len(idx_test)
    )
    return splits


# ── Class-imbalance handling ──────────────────────────────────────────────────

def apply_smote(X: csr_matrix, y: np.ndarray,
                ratio: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply SMOTE over-sampling to training data.
    Falls back gracefully if imbalanced-learn is not installed.
    """
    try:
        from imblearn.over_sampling import SMOTE
        sm = SMOTE(sampling_strategy=ratio, random_state=RANDOM_SEED, n_jobs=-1)
        X_res, y_res = sm.fit_resample(X, y)
        logger.info("SMOTE: %s → %s", np.bincount(y), np.bincount(y_res))
        return X_res, y_res
    except ImportError:
        logger.warning("imbalanced-learn not installed — skipping SMOTE.")
        return X, y


# ── Build LSTM vocabulary & sequences ────────────────────────────────────────

def build_lstm_sequences(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
    max_vocab: int = 30_000,
    max_len: int   = 512,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """
    Build a word-to-index vocabulary from training texts and convert
    all splits to padded integer sequences.

    Returns (train_seq, val_seq, test_seq, vocab)
    """
    from collections import Counter
    from nltk.tokenize import word_tokenize

    # Build vocab
    counter = Counter()
    for text in X_train:
        counter.update(text.split())

    vocab = {"<PAD>": 0, "<UNK>": 1}
    for word, _ in counter.most_common(max_vocab - 2):
        vocab[word] = len(vocab)

    def encode(texts):
        seqs = []
        for text in texts:
            ids = [vocab.get(w, 1) for w in text.split()][:max_len]
            pad = max_len - len(ids)
            seqs.append(ids + [0] * pad)
        return np.array(seqs, dtype=np.int32)

    return encode(X_train), encode(X_val), encode(X_test), vocab


# ── Main training orchestrator ────────────────────────────────────────────────

class Trainer:
    """
    Full training pipeline for the Fake News Detection System.

    Usage
    -----
    trainer = Trainer()
    trainer.fit(df, X_text, y)
    results = trainer.results   # dict of {model_name: metrics}
    """

    def __init__(
        self,
        use_meta:    bool = True,
        use_smote:   bool = False,
        train_lstm:  bool = False,
        train_bert:  bool = False,
        cv_folds:    int  = 5,
    ):
        self.use_meta   = use_meta
        self.use_smote  = use_smote
        self.train_lstm = train_lstm
        self.train_bert = train_bert
        self.cv_folds   = cv_folds

        self.feature_pipeline: Optional[FeaturePipeline] = None
        self.splits:  dict = {}
        self.results: dict = {}
        self.trained_models: dict = {}

    # ── Internals ─────────────────────────────────────────────────────────────

    def _build_features(self, splits: dict) -> Tuple[csr_matrix, csr_matrix, csr_matrix]:
        fp = FeaturePipeline(use_meta=self.use_meta)
        X_tr = fp.fit_transform(splits["X_train"], splits["df_train"])
        X_va = fp.transform(splits["X_val"],   splits["df_val"])
        X_te = fp.transform(splits["X_test"],  splits["df_test"])
        self.feature_pipeline = fp
        fp.save()
        logger.info("Feature matrix shapes — train:%s  val:%s  test:%s",
                    X_tr.shape, X_va.shape, X_te.shape)
        return X_tr, X_va, X_te

    def _cross_validate(self, model, X_tr: np.ndarray, y_tr: np.ndarray) -> dict:
        skf     = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=RANDOM_SEED)
        scoring = {
            "accuracy":  "accuracy",
            "f1":        make_scorer(f1_score, average="weighted"),
            "precision": "precision_weighted",
            "recall":    "recall_weighted",
        }
        cv_res = cross_validate(model.clf, X_tr, y_tr, cv=skf, scoring=scoring, n_jobs=-1)
        return {k: float(v.mean()) for k, v in cv_res.items() if k.startswith("test_")}

    # ── Public API ────────────────────────────────────────────────────────────

    def fit(
        self,
        df:     pd.DataFrame,
        X_text: np.ndarray,
        y:      np.ndarray,
    ) -> "Trainer":
        # 1. Split
        self.splits = split_data(df, X_text, y)

        # 2. Feature engineering
        X_tr, X_va, X_te = self._build_features(self.splits)

        # 3. Optional SMOTE
        if self.use_smote:
            X_tr, self.splits["y_train"] = apply_smote(X_tr, self.splits["y_train"])

        # 4. Train classical models
        classical = get_all_classical_models()
        for mname, model in classical.items():
            logger.info("=== Training: %s ===", mname)

            # NB requires non-negative features → use TF-IDF only (no meta)
            X_tr_fit = X_tr
            if isinstance(model, NaiveBayesModel):
                # Re-extract TF-IDF-only sparse matrix (no meta columns)
                X_tr_fit = self.feature_pipeline.tfidf.transform(self.splits["X_train"])

            model.fit(X_tr_fit, self.splits["y_train"])
            model._tfidf_only = isinstance(model, NaiveBayesModel)  # flag for prediction

            cv_scores = self._cross_validate(model, X_tr_fit, self.splits["y_train"])
            logger.info("[%s] CV scores: %s", mname, cv_scores)

            model.save()
            self.trained_models[mname] = model
            self.results[mname] = {"cv": cv_scores}

        # 5. LSTM (optional)
        if self.train_lstm:
            try:
                self._train_lstm(self.splits)
            except Exception as exc:
                logger.error("LSTM training failed: %s", exc)

        # 6. BERT (optional)
        if self.train_bert:
            try:
                self._train_bert(self.splits)
            except Exception as exc:
                logger.error("BERT training failed: %s", exc)

        # Store feature matrices for evaluation
        self._X_tr = X_tr
        self._X_va = X_va
        self._X_te = X_te

        return self

    def _train_lstm(self, splits: dict):
        from src.models import LSTMClassifier

        logger.info("=== Training: BiLSTM ===")
        tr_seq, va_seq, te_seq, vocab = build_lstm_sequences(
            splits["X_train"], splits["X_val"], splits["X_test"],
            max_vocab = LSTM_CONFIG["vocab_size"],
            max_len   = 512,
        )
        lstm = LSTMClassifier(vocab_size=len(vocab))
        lstm.fit(
            tr_seq, splits["y_train"],
            X_val=va_seq, y_val=splits["y_val"],
        )
        lstm.save()
        self.trained_models["BiLSTM"] = lstm
        self._lstm_sequences = {
            "train": tr_seq, "val": va_seq, "test": te_seq
        }

    def _train_bert(self, splits: dict):
        from src.models import BERTClassifier

        logger.info("=== Training: DistilBERT ===")
        bert = BERTClassifier()
        bert.fit(
            list(splits["X_train"]), splits["y_train"],
            texts_val=list(splits["X_val"]), y_val=splits["y_val"],
        )
        bert.save()
        self.trained_models["DistilBERT"] = bert

    def get_test_predictions(self) -> dict:
        """
        Run all trained models on the held-out test set.
        Returns {model_name: {"y_pred": ..., "y_proba": ...}}.
        """
        preds = {}
        for name, model in self.trained_models.items():
            if name == "BiLSTM":
                X = self._lstm_sequences["test"]
            elif name == "DistilBERT":
                X = list(self.splits["X_test"])
            elif getattr(model, "_tfidf_only", False):
                # Naive Bayes: use TF-IDF only (no meta cols)
                X = self.feature_pipeline.tfidf.transform(self.splits["X_test"])
            else:
                X = self._X_te

            y_pred  = model.predict(X)
            y_proba = model.predict_proba(X)[:, 1]
            preds[name] = {"y_pred": y_pred, "y_proba": y_proba}

        return preds
