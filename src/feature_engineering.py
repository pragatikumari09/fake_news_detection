"""
src/feature_engineering.py
---------------------------
Builds feature matrices from cleaned text and meta-features.

Supported feature extractors
  • TF-IDF (unigrams + bigrams)  — fast, strong baseline
  • Meta-features                — sentiment, length, sensational words, etc.
  • Combined (TF-IDF + meta)     — best for traditional ML

Word2Vec / GloVe mean-vector features are also supported as an option.
BERT contextual embeddings are handled separately inside the BERT model class.
"""

import sys
import logging
import pickle
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack, issparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import TFIDF_CONFIG, MODEL_DIR, RANDOM_SEED

logger = logging.getLogger(__name__)

# Meta-feature column names (must match preprocessing.py)
META_COLS = [
    "char_len", "word_count", "avg_word_len",
    "sentiment_polarity", "sentiment_subjectivity",
    "sensational_count", "url_count",
    "exclamation_count", "question_count",
    "capital_ratio",
]


# ── TF-IDF ────────────────────────────────────────────────────────────────────

class TfidfFeatureExtractor:
    """Thin wrapper around scikit-learn TfidfVectorizer."""

    def __init__(self, **kwargs):
        config = {**TFIDF_CONFIG, **kwargs}
        self.vectorizer = TfidfVectorizer(**config)
        self._fitted = False

    def fit(self, texts: np.ndarray) -> "TfidfFeatureExtractor":
        logger.info("[TF-IDF] Fitting on %d documents …", len(texts))
        self.vectorizer.fit(texts)
        self._fitted = True
        logger.info("[TF-IDF] Vocabulary size: %d", len(self.vectorizer.vocabulary_))
        return self

    def transform(self, texts: np.ndarray) -> csr_matrix:
        if not self._fitted:
            raise RuntimeError("Call .fit() first.")
        return self.vectorizer.transform(texts)

    def fit_transform(self, texts: np.ndarray) -> csr_matrix:
        return self.fit(texts).transform(texts)

    def get_feature_names(self):
        return self.vectorizer.get_feature_names_out()

    def save(self, path: Optional[Path] = None):
        path = path or (MODEL_DIR / "tfidf_vectorizer.pkl")
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info("[TF-IDF] Saved → %s", path)

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "TfidfFeatureExtractor":
        path = path or (MODEL_DIR / "tfidf_vectorizer.pkl")
        with open(path, "rb") as f:
            obj = pickle.load(f)
        logger.info("[TF-IDF] Loaded ← %s", path)
        return obj


# ── Meta-feature scaler ───────────────────────────────────────────────────────

class MetaFeatureExtractor:
    """Extracts and scales the hand-crafted numeric meta-features."""

    def __init__(self, cols=META_COLS):
        self.cols   = cols
        self.scaler = StandardScaler()
        self._fitted = False

    def _get_array(self, df: pd.DataFrame) -> np.ndarray:
        missing = [c for c in self.cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing meta-feature columns: {missing}")
        return df[self.cols].values.astype(float)

    def fit(self, df: pd.DataFrame) -> "MetaFeatureExtractor":
        X = self._get_array(df)
        self.scaler.fit(X)
        self._fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Call .fit() first.")
        return self.scaler.transform(self._get_array(df))

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        return self.fit(df).transform(df)

    def save(self, path: Optional[Path] = None):
        path = path or (MODEL_DIR / "meta_scaler.pkl")
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "MetaFeatureExtractor":
        path = path or (MODEL_DIR / "meta_scaler.pkl")
        with open(path, "rb") as f:
            return pickle.load(f)


# ── Combined feature builder ──────────────────────────────────────────────────

class FeaturePipeline:
    """
    Combines TF-IDF sparse matrix with dense meta-feature matrix.

    Usage
    -----
    fp = FeaturePipeline()
    X_train = fp.fit_transform(X_text_train, df_train)
    X_test  = fp.transform(X_text_test,  df_test)
    """

    def __init__(self, use_meta: bool = True, **tfidf_kwargs):
        self.tfidf    = TfidfFeatureExtractor(**tfidf_kwargs)
        self.meta     = MetaFeatureExtractor() if use_meta else None
        self.use_meta = use_meta
        self._fitted  = False

    def fit(self, X_text: np.ndarray, df: pd.DataFrame) -> "FeaturePipeline":
        self.tfidf.fit(X_text)
        if self.use_meta:
            self.meta.fit(df)
        self._fitted = True
        return self

    def transform(self, X_text: np.ndarray, df: pd.DataFrame) -> csr_matrix:
        if not self._fitted:
            raise RuntimeError("Call .fit() first.")
        X_tfidf = self.tfidf.transform(X_text)
        if self.use_meta:
            X_meta  = self.meta.transform(df)
            X_dense = csr_matrix(X_meta)
            return hstack([X_tfidf, X_dense], format="csr")
        return X_tfidf

    def fit_transform(self, X_text: np.ndarray, df: pd.DataFrame) -> csr_matrix:
        return self.fit(X_text, df).transform(X_text, df)

    def save(self, path: Optional[Path] = None):
        path = path or (MODEL_DIR / "feature_pipeline.pkl")
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info("[FeaturePipeline] Saved → %s", path)

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "FeaturePipeline":
        path = path or (MODEL_DIR / "feature_pipeline.pkl")
        with open(path, "rb") as f:
            obj = pickle.load(f)
        logger.info("[FeaturePipeline] Loaded ← %s", path)
        return obj


# ── Word2Vec mean-vector (optional) ──────────────────────────────────────────

def word2vec_mean_vectors(texts: list, model) -> np.ndarray:
    """
    Compute mean Word2Vec embedding for each document.

    Parameters
    ----------
    texts : list of tokenised strings (already preprocessed)
    model : gensim KeyedVectors or Word2Vec model

    Returns
    -------
    np.ndarray of shape (n_docs, embedding_dim)
    """
    dim   = model.vector_size
    vecs  = []
    for text in texts:
        tokens = text.split()
        token_vecs = [
            model[t] for t in tokens if t in model
        ]
        if token_vecs:
            vecs.append(np.mean(token_vecs, axis=0))
        else:
            vecs.append(np.zeros(dim))
    return np.array(vecs, dtype=np.float32)


# ── CLI smoke-test ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from sklearn.datasets import fetch_20newsgroups
    cats = ["sci.space", "talk.politics.misc"]
    bunch = fetch_20newsgroups(subset="train", categories=cats, remove=("headers",))
    texts = np.array(bunch.data)

    fp = FeaturePipeline(use_meta=False)   # skip meta (no DataFrame here)
    fp.tfidf.fit(texts)
    X = fp.tfidf.transform(texts)
    print("TF-IDF shape:", X.shape)
