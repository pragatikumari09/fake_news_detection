"""
src/interpretability.py
------------------------
Model interpretability & explanation tools.

Techniques implemented
  • LIME (Local Interpretable Model-agnostic Explanations)
      — per-prediction word-importance scores
      — works with any sklearn-compatible model
  • SHAP (SHapley Additive exPlanations)
      — global feature importance via LinearExplainer / TreeExplainer
      — beeswarm & bar plots
  • Top-words analysis
      — for Logistic Regression: extract highest-weight TF-IDF terms
        per class (no black-box needed)
  • Word-cloud generation (FAKE vs REAL most discriminative words)
"""

import sys
import logging
import warnings
from pathlib import Path
from typing import Optional, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import VIZ_DIR, RANDOM_SEED

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")


# ── LIME explanations ─────────────────────────────────────────────────────────

def explain_with_lime(
    model,
    feature_pipeline,
    text: str,
    num_features: int = 20,
    num_samples:  int = 1000,
    class_names:  List[str] = ["FAKE", "REAL"],
):
    """
    Generate a LIME explanation for a single news article.

    Parameters
    ----------
    model            : Trained sklearn-compatible model (predict_proba required).
    feature_pipeline : FeaturePipeline (fitted TF-IDF + optional meta).
    text             : Cleaned article text to explain.
    num_features     : Number of top words to highlight.

    Returns
    -------
    lime.explanation.Explanation  (can call .as_html(), .as_list(), .show_in_notebook())
    """
    try:
        from lime.lime_text import LimeTextExplainer
    except ImportError:
        logger.error("lime not installed. Run: pip install lime")
        return None

    explainer = LimeTextExplainer(
        class_names  = class_names,
        random_state = RANDOM_SEED,
    )

    def predict_fn(texts):
        """LIME passes a list of strings; we vectorise and predict using full pipeline."""
        import pandas as pd
        from src.preprocessing import compute_meta_features
        from scipy.sparse import hstack, csr_matrix
        texts_arr = np.array(texts)
        X_tfidf   = feature_pipeline.tfidf.transform(texts_arr)
        if feature_pipeline.use_meta and feature_pipeline.meta:
            meta_rows = pd.DataFrame([compute_meta_features(t) for t in texts])
            X_meta    = feature_pipeline.meta.transform(meta_rows)
            X_combined = hstack([X_tfidf, csr_matrix(X_meta)], format="csr")
        else:
            X_combined = X_tfidf
        return model.predict_proba(X_combined)

    explanation = explainer.explain_instance(
        text,
        predict_fn,
        num_features = num_features,
        num_samples  = num_samples,
    )
    return explanation


def plot_lime_explanation(explanation, title: str = "LIME Explanation", save: bool = True):
    """
    Convert a LIME explanation to a horizontal bar chart.
    Positive weights → REAL; Negative weights → FAKE.
    """
    word_weights = explanation.as_list()
    words  = [w for w, _ in word_weights]
    scores = [s for _, s in word_weights]

    colours = ["#4361EE" if s > 0 else "#F72585" for s in scores]

    fig, ax = plt.subplots(figsize=(8, max(4, len(words) * 0.35)))
    bars = ax.barh(words[::-1], scores[::-1], color=colours[::-1], edgecolor="white")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("LIME weight  (← FAKE  |  REAL →)", fontsize=11)
    ax.set_title(title, fontsize=12, pad=10)

    fake_patch = mpatches.Patch(color="#F72585", label="Pushes → FAKE")
    real_patch = mpatches.Patch(color="#4361EE", label="Pushes → REAL")
    ax.legend(handles=[real_patch, fake_patch], fontsize=9, loc="lower right")
    plt.tight_layout()

    if save:
        path = VIZ_DIR / "lime_explanation.png"
        fig.savefig(path, bbox_inches="tight")
        logger.info("LIME plot saved → %s", path)
    return fig


# ── SHAP explanations ─────────────────────────────────────────────────────────

def explain_with_shap(
    model,
    X_train,
    X_test,
    feature_names: Optional[List[str]] = None,
    max_display: int = 25,
    save: bool = True,
):
    """
    Compute and plot SHAP values for a trained LR / RF model.
    Uses LinearExplainer for LR, TreeExplainer for RF.

    Returns shap_values array.
    """
    try:
        import shap
    except ImportError:
        logger.error("shap not installed. Run: pip install shap")
        return None

    clf = model.clf  # underlying sklearn estimator

    # Dense for tree-based models
    if hasattr(X_train, "toarray"):
        X_tr_dense = X_train.toarray()
        X_te_dense = X_test.toarray()
    else:
        X_tr_dense = X_train
        X_te_dense = X_test

    model_type = type(clf).__name__
    try:
        if "LogisticRegression" in model_type:
            explainer   = shap.LinearExplainer(clf, X_tr_dense, feature_perturbation="interventional")
            shap_values = explainer.shap_values(X_te_dense[:200])
        elif "RandomForest" in model_type:
            explainer   = shap.TreeExplainer(clf)
            shap_values = explainer.shap_values(X_te_dense[:200])
            if isinstance(shap_values, list):
                shap_values = shap_values[1]   # positive class
        else:
            logger.warning("SHAP not configured for %s; skipping.", model_type)
            return None
    except Exception as exc:
        logger.error("SHAP computation failed: %s", exc)
        return None

    # Bar plot — global importance
    fig_bar, ax = plt.subplots(figsize=(9, 6))
    shap.summary_plot(
        shap_values, X_te_dense[:200],
        feature_names = feature_names,
        max_display   = max_display,
        plot_type     = "bar",
        show          = False,
    )
    plt.title("SHAP Feature Importance (bar)", fontsize=12)
    plt.tight_layout()
    if save:
        fig_bar.savefig(VIZ_DIR / "shap_bar.png", bbox_inches="tight")

    # Beeswarm (dot) plot
    fig_dot, ax = plt.subplots(figsize=(9, 6))
    shap.summary_plot(
        shap_values, X_te_dense[:200],
        feature_names = feature_names,
        max_display   = max_display,
        show          = False,
    )
    plt.title("SHAP Summary (beeswarm)", fontsize=12)
    plt.tight_layout()
    if save:
        fig_dot.savefig(VIZ_DIR / "shap_beeswarm.png", bbox_inches="tight")

    return shap_values


# ── Logistic Regression top-word analysis ─────────────────────────────────────

def lr_top_words(
    model,
    feature_pipeline,
    n_top: int = 30,
    save:  bool = True,
) -> pd.DataFrame:
    """
    For a trained Logistic Regression model, extract the n_top words
    that most strongly indicate FAKE vs REAL.
    """
    clf = model.clf
    if not hasattr(clf, "coef_"):
        logger.warning("Model has no .coef_ attribute; skipping top-word analysis.")
        return pd.DataFrame()

    feature_names = feature_pipeline.tfidf.get_feature_names()
    coef = clf.coef_[0]

    # Coefficients may include meta-feature dims appended after TF-IDF dims
    # Slice to TF-IDF vocab size only
    n_vocab = len(feature_names)
    coef    = coef[:n_vocab]
    n       = min(n_top, n_vocab)
    top_fake_idx = np.argsort(coef)[:n]
    top_real_idx = np.argsort(coef)[-n:][::-1]

    df_fake = pd.DataFrame({
        "word":   feature_names[top_fake_idx],
        "weight": coef[top_fake_idx],
        "class":  "FAKE",
    })
    df_real = pd.DataFrame({
        "word":   feature_names[top_real_idx],
        "weight": coef[top_real_idx],
        "class":  "REAL",
    })
    df_top = pd.concat([df_fake, df_real], ignore_index=True)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, (cls, colour, df_cls) in zip(axes, [
        ("FAKE", "#F72585", df_fake),
        ("REAL", "#4361EE", df_real),
    ]):
        ax.barh(df_cls["word"][::-1], df_cls["weight"].abs()[::-1],
                color=colour, edgecolor="white")
        ax.set_title(f"Top {n} words — {cls}", fontsize=12)
        ax.set_xlabel("Absolute coefficient weight")
    plt.suptitle("Logistic Regression — Most Discriminative Words", fontsize=13, y=1.01)
    plt.tight_layout()
    if save:
        path = VIZ_DIR / "lr_top_words.png"
        fig.savefig(path, bbox_inches="tight")
        logger.info("Top-words plot saved → %s", path)

    return df_top


# ── Word Cloud ─────────────────────────────────────────────────────────────────

def generate_wordcloud(
    texts: np.ndarray,
    labels: np.ndarray,
    target_label: int = 0,
    max_words: int = 150,
    save: bool = True,
):
    """
    Generate a word-cloud for FAKE (label=0) or REAL (label=1) articles.
    """
    try:
        from wordcloud import WordCloud, STOPWORDS
    except ImportError:
        logger.warning("wordcloud not installed — skipping word-cloud generation.")
        return None

    mask_texts = texts[labels == target_label]
    combined   = " ".join(mask_texts)
    label_name = "FAKE" if target_label == 0 else "REAL"
    color_fn   = "#F72585" if target_label == 0 else "#4361EE"

    wc = WordCloud(
        width           = 900,
        height          = 500,
        max_words       = max_words,
        background_color= "white",
        colormap        = "RdPu" if target_label == 0 else "Blues",
        stopwords       = STOPWORDS,
        random_state    = RANDOM_SEED,
    ).generate(combined)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    ax.set_title(f"Most Common Words in {label_name} News", fontsize=14, pad=10)
    plt.tight_layout()

    if save:
        path = VIZ_DIR / f"wordcloud_{label_name.lower()}.png"
        fig.savefig(path, bbox_inches="tight", dpi=150)
        logger.info("Word-cloud saved → %s", path)
    return fig
