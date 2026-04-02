"""
src/models.py
-------------
Defines all classification models used in the Fake News Detection System.

Models implemented
  Traditional ML (sklearn-compatible)
    • LogisticRegressionModel
    • NaiveBayesModel         (ComplementNB — excels on text)
    • RandomForestModel
    • VotingEnsembleModel     (soft-vote across the three above)

  Deep Learning (PyTorch)
    • LSTMClassifier          (Bidirectional LSTM)
    • BERTClassifier          (DistilBERT fine-tuning via HuggingFace)

All traditional models expose a uniform sklearn API:  fit / predict / predict_proba.
"""

import sys
import logging
import pickle
from pathlib import Path
from typing import Optional, List

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    LR_PARAMS, NB_PARAMS, RF_PARAMS, LSTM_CONFIG,
    BERT_MODEL_NAME, BERT_CONFIG, MODEL_DIR, RANDOM_SEED,
)

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# Traditional ML wrappers
# ══════════════════════════════════════════════════════════════════════════════

from sklearn.linear_model  import LogisticRegression
from sklearn.naive_bayes   import ComplementNB
from sklearn.ensemble      import RandomForestClassifier, VotingClassifier
from sklearn.pipeline      import Pipeline
from sklearn.calibration   import CalibratedClassifierCV


class _BaseModel:
    """Mixin providing save / load helpers for sklearn models."""

    name: str = "base"

    def save(self, path: Optional[Path] = None):
        path = path or (MODEL_DIR / f"{self.name}.pkl")
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info("[%s] Saved → %s", self.name, path)

    @classmethod
    def load(cls, path: Optional[Path] = None):
        name = getattr(cls, "name", cls.__name__.lower())
        path = path or (MODEL_DIR / f"{name}.pkl")
        with open(path, "rb") as f:
            obj = pickle.load(f)
        logger.info("[%s] Loaded ← %s", name, path)
        return obj


class LogisticRegressionModel(_BaseModel):
    """L2-regularised Logistic Regression with class-weight balancing."""
    name = "logistic_regression"

    def __init__(self, **kwargs):
        params = {**LR_PARAMS, **kwargs}
        self.clf = LogisticRegression(**params)

    def fit(self, X, y):
        logger.info("[LR] Training …")
        self.clf.fit(X, y)
        return self

    def predict(self, X):
        return self.clf.predict(X)

    def predict_proba(self, X):
        return self.clf.predict_proba(X)


class NaiveBayesModel(_BaseModel):
    """Complement Naive Bayes (better than MultinomialNB for imbalanced text)."""
    name = "naive_bayes"

    def __init__(self, **kwargs):
        params = {**NB_PARAMS, **kwargs}
        # CNB requires non-negative features — TF-IDF values are always ≥ 0
        self.clf = CalibratedClassifierCV(
            ComplementNB(**params), method="sigmoid", cv=3
        )

    def fit(self, X, y):
        logger.info("[NB] Training …")
        self.clf.fit(X, y)
        return self

    def predict(self, X):
        return self.clf.predict(X)

    def predict_proba(self, X):
        return self.clf.predict_proba(X)


class RandomForestModel(_BaseModel):
    """Random Forest with balanced class weights."""
    name = "random_forest"

    def __init__(self, **kwargs):
        params = {**RF_PARAMS, **kwargs}
        self.clf = RandomForestClassifier(**params)

    def fit(self, X, y):
        logger.info("[RF] Training …")
        self.clf.fit(X, y)
        return self

    def predict(self, X):
        return self.clf.predict(X)

    def predict_proba(self, X):
        return self.clf.predict_proba(X)


class VotingEnsembleModel(_BaseModel):
    """Soft-voting ensemble of LR + NB + RF (requires dense TF-IDF for RF)."""
    name = "voting_ensemble"

    def __init__(self):
        self.clf = VotingClassifier(
            estimators=[
                ("lr", LogisticRegression(**LR_PARAMS)),
                ("nb", CalibratedClassifierCV(ComplementNB(**NB_PARAMS), cv=3)),
                ("rf", RandomForestClassifier(**RF_PARAMS)),
            ],
            voting="soft",
            n_jobs=-1,
        )

    def fit(self, X, y):
        # VotingClassifier needs dense for RF
        if hasattr(X, "toarray"):
            X = X.toarray()
        logger.info("[Ensemble] Training …")
        self.clf.fit(X, y)
        return self

    def predict(self, X):
        if hasattr(X, "toarray"):
            X = X.toarray()
        return self.clf.predict(X)

    def predict_proba(self, X):
        if hasattr(X, "toarray"):
            X = X.toarray()
        return self.clf.predict_proba(X)


# ══════════════════════════════════════════════════════════════════════════════
# Deep Learning — Bidirectional LSTM (PyTorch)
# ══════════════════════════════════════════════════════════════════════════════

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    from torch.optim import Adam
    from torch.optim.lr_scheduler import ReduceLROnPlateau
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
    logger.warning("PyTorch not installed — LSTM/BERT models unavailable.")


if _TORCH_AVAILABLE:

    class TextDataset(Dataset):
        def __init__(self, sequences: np.ndarray, labels: np.ndarray):
            self.X = torch.tensor(sequences, dtype=torch.long)
            self.y = torch.tensor(labels,    dtype=torch.long)

        def __len__(self):
            return len(self.X)

        def __getitem__(self, idx):
            return self.X[idx], self.y[idx]

    # ── LSTM architecture ─────────────────────────────────────────────────────

    class _LSTMNet(nn.Module):
        def __init__(self, vocab_size, embed_dim, hidden_dim,
                     num_layers, dropout, bidirectional, num_classes=2):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
            self.lstm = nn.LSTM(
                embed_dim, hidden_dim,
                num_layers     = num_layers,
                batch_first    = True,
                dropout        = dropout if num_layers > 1 else 0.0,
                bidirectional  = bidirectional,
            )
            d = hidden_dim * (2 if bidirectional else 1)
            self.dropout = nn.Dropout(dropout)
            self.fc      = nn.Linear(d, num_classes)

        def forward(self, x):
            emb = self.embedding(x)                 # (B, T, E)
            out, (h, _) = self.lstm(emb)            # (B, T, 2H)
            # Concat final fwd & bwd hidden states
            if self.lstm.bidirectional:
                h = torch.cat((h[-2], h[-1]), dim=1)
            else:
                h = h[-1]
            return self.fc(self.dropout(h))         # (B, 2)

    class LSTMClassifier(_BaseModel):
        """
        Bidirectional LSTM text classifier.

        Parameters are read from config.LSTM_CONFIG but can be overridden.
        """
        name = "lstm"

        def __init__(self, vocab_size=None, **kwargs):
            cfg = {**LSTM_CONFIG, **kwargs}
            if vocab_size:
                cfg["vocab_size"] = vocab_size
            self.cfg      = cfg
            self.device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model    = _LSTMNet(
                vocab_size    = cfg["vocab_size"],
                embed_dim     = cfg["embed_dim"],
                hidden_dim    = cfg["hidden_dim"],
                num_layers    = cfg["num_layers"],
                dropout       = cfg["dropout"],
                bidirectional = cfg["bidirectional"],
            ).to(self.device)
            self.tokenizer = None   # set externally after vocabulary is built
            self._trained  = False

        def _make_loader(self, X, y, shuffle=True):
            ds = TextDataset(X, y)
            return DataLoader(ds, batch_size=self.cfg["batch_size"], shuffle=shuffle)

        def fit(self, X_seq: np.ndarray, y: np.ndarray,
                X_val: Optional[np.ndarray] = None,
                y_val: Optional[np.ndarray] = None):
            """
            X_seq : integer-encoded padded sequences (n, max_seq_len)
            y     : binary labels
            """
            train_loader = self._make_loader(X_seq, y)
            val_loader   = self._make_loader(X_val, y_val, shuffle=False) \
                           if X_val is not None else None

            optimizer = Adam(self.model.parameters(), lr=self.cfg["lr"])
            scheduler = ReduceLROnPlateau(optimizer, patience=2, factor=0.5)
            criterion = nn.CrossEntropyLoss()

            best_val_loss = float("inf")
            best_state    = None

            for epoch in range(1, self.cfg["epochs"] + 1):
                self.model.train()
                total_loss, correct, total = 0.0, 0, 0
                for xb, yb in train_loader:
                    xb, yb = xb.to(self.device), yb.to(self.device)
                    optimizer.zero_grad()
                    logits = self.model(xb)
                    loss   = criterion(logits, yb)
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg["clip_grad"])
                    optimizer.step()
                    total_loss += loss.item() * len(yb)
                    correct    += (logits.argmax(1) == yb).sum().item()
                    total      += len(yb)
                train_acc  = correct / total
                train_loss = total_loss / total

                val_info = ""
                if val_loader:
                    val_loss, val_acc = self._evaluate_loader(val_loader, criterion)
                    scheduler.step(val_loss)
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                    val_info = f"  |  val_loss={val_loss:.4f}  val_acc={val_acc:.4f}"

                logger.info("Epoch %2d/%d  loss=%.4f  acc=%.4f%s",
                            epoch, self.cfg["epochs"], train_loss, train_acc, val_info)

            if best_state:
                self.model.load_state_dict(best_state)
            self._trained = True
            return self

        @torch.no_grad()
        def _evaluate_loader(self, loader, criterion):
            self.model.eval()
            total_loss, correct, total = 0.0, 0, 0
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                logits  = self.model(xb)
                loss    = criterion(logits, yb)
                total_loss += loss.item() * len(yb)
                correct    += (logits.argmax(1) == yb).sum().item()
                total      += len(yb)
            return total_loss / total, correct / total

        @torch.no_grad()
        def predict_proba(self, X_seq: np.ndarray) -> np.ndarray:
            self.model.eval()
            loader = self._make_loader(X_seq, np.zeros(len(X_seq), dtype=int), shuffle=False)
            probs  = []
            for xb, _ in loader:
                xb = xb.to(self.device)
                p  = torch.softmax(self.model(xb), dim=1).cpu().numpy()
                probs.append(p)
            return np.concatenate(probs, axis=0)

        def predict(self, X_seq: np.ndarray) -> np.ndarray:
            return self.predict_proba(X_seq).argmax(axis=1)


# ══════════════════════════════════════════════════════════════════════════════
# Deep Learning — DistilBERT fine-tuning (HuggingFace)
# ══════════════════════════════════════════════════════════════════════════════

try:
    from transformers import (
        AutoTokenizer, AutoModelForSequenceClassification,
        get_linear_schedule_with_warmup,
    )
    from torch.utils.data import Dataset as TorchDataset
    _HF_AVAILABLE = _TORCH_AVAILABLE
except ImportError:
    _HF_AVAILABLE = False


if _HF_AVAILABLE:

    class BERTTextDataset(TorchDataset):
        def __init__(self, texts, labels, tokenizer, max_len):
            self.texts     = texts
            self.labels    = labels
            self.tokenizer = tokenizer
            self.max_len   = max_len

        def __len__(self):
            return len(self.texts)

        def __getitem__(self, idx):
            enc = self.tokenizer(
                self.texts[idx],
                max_length      = self.max_len,
                padding         = "max_length",
                truncation      = True,
                return_tensors  = "pt",
            )
            return {
                "input_ids":      enc["input_ids"].squeeze(0),
                "attention_mask": enc["attention_mask"].squeeze(0),
                "label":          torch.tensor(self.labels[idx], dtype=torch.long),
            }

    class BERTClassifier(_BaseModel):
        """
        DistilBERT fine-tuned for binary fake-news classification.
        Lighter and ~60% faster than full BERT, with ~97% of its accuracy.
        """
        name = "bert"

        def __init__(self, model_name: str = BERT_MODEL_NAME, **kwargs):
            cfg = {**BERT_CONFIG, **kwargs}
            self.cfg        = cfg
            self.model_name = model_name
            self.device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.tokenizer  = AutoTokenizer.from_pretrained(model_name)
            self.model      = AutoModelForSequenceClassification.from_pretrained(
                model_name, num_labels=2
            ).to(self.device)

        def _make_loader(self, texts, labels, shuffle=True):
            ds = BERTTextDataset(texts, labels, self.tokenizer, self.cfg["max_len"])
            return DataLoader(ds, batch_size=self.cfg["batch_size"], shuffle=shuffle)

        def fit(self, texts: List[str], y: np.ndarray,
                texts_val=None, y_val=None):
            train_loader = self._make_loader(texts, y)
            val_loader   = self._make_loader(texts_val, y_val, shuffle=False) \
                           if texts_val is not None else None

            optimizer   = torch.optim.AdamW(
                self.model.parameters(),
                lr           = self.cfg["lr"],
                weight_decay = self.cfg["weight_decay"],
            )
            total_steps = len(train_loader) * self.cfg["epochs"]
            warmup      = int(total_steps * self.cfg["warmup_ratio"])
            scheduler   = get_linear_schedule_with_warmup(
                optimizer, warmup, total_steps
            )

            for epoch in range(1, self.cfg["epochs"] + 1):
                self.model.train()
                total_loss, correct, total = 0.0, 0, 0
                for batch in train_loader:
                    ids   = batch["input_ids"].to(self.device)
                    mask  = batch["attention_mask"].to(self.device)
                    labels = batch["label"].to(self.device)
                    optimizer.zero_grad()
                    out  = self.model(ids, attention_mask=mask, labels=labels)
                    loss = out.loss
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()
                    total_loss += loss.item() * len(labels)
                    correct    += (out.logits.argmax(1) == labels).sum().item()
                    total      += len(labels)
                logger.info("BERT Epoch %d/%d  loss=%.4f  acc=%.4f",
                            epoch, self.cfg["epochs"],
                            total_loss / total, correct / total)

            return self

        @torch.no_grad()
        def predict_proba(self, texts: List[str]) -> np.ndarray:
            self.model.eval()
            loader = self._make_loader(texts, [0] * len(texts), shuffle=False)
            probs  = []
            for batch in loader:
                ids  = batch["input_ids"].to(self.device)
                mask = batch["attention_mask"].to(self.device)
                out  = self.model(ids, attention_mask=mask)
                p    = torch.softmax(out.logits, dim=1).cpu().numpy()
                probs.append(p)
            return np.concatenate(probs, axis=0)

        def predict(self, texts: List[str]) -> np.ndarray:
            return self.predict_proba(texts).argmax(axis=1)

        def save(self, path: Optional[Path] = None):
            path = path or (MODEL_DIR / "bert_model")
            path.mkdir(parents=True, exist_ok=True)
            self.model.save_pretrained(path)
            self.tokenizer.save_pretrained(path)
            logger.info("[BERT] Saved → %s", path)

        @classmethod
        def load(cls, path: Optional[Path] = None):
            path = path or (MODEL_DIR / "bert_model")
            obj = cls.__new__(cls)
            obj.device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            obj.tokenizer = AutoTokenizer.from_pretrained(str(path))
            obj.model     = AutoModelForSequenceClassification.from_pretrained(
                str(path)
            ).to(obj.device)
            obj.cfg = BERT_CONFIG
            logger.info("[BERT] Loaded ← %s", path)
            return obj


# Registry ─────────────────────────────────────────────────────────────────────

def get_all_classical_models():
    """Return a dict of {name: model_instance} for all traditional ML models."""
    return {
        "Logistic Regression": LogisticRegressionModel(),
        "Naive Bayes":         NaiveBayesModel(),
        "Random Forest":       RandomForestModel(),
    }
