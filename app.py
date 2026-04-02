"""
app.py — Streamlit Web Application
====================================
Fake News Detection System — Interactive UI

Features
  • Paste or type any news article for instant classification
  • Confidence score with colour-coded verdict badge
  • LIME word-level explanation visualisation
  • Comparison across multiple pre-trained models
  • Demo examples (one fake, one real)
  • Basic text analytics: sentiment, word count, sensational words

Run with:
  streamlit run app.py
"""

import sys
import logging
import pickle
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Project imports ────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from config import MODEL_DIR, APP_TITLE, APP_FAVICON, DEFAULT_MODEL, SENSATIONAL_WORDS
from src.preprocessing     import full_pipeline, compute_meta_features
from src.feature_engineering import FeaturePipeline
from src.interpretability  import explain_with_lime, plot_lime_explanation

logger = logging.getLogger(__name__)

# ── Page configuration ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title = APP_TITLE,
    page_icon  = APP_FAVICON,
    layout     = "wide",
    initial_sidebar_state = "expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Global font */
html, body, [class*="css"] { font-family: 'Segoe UI', sans-serif; }

/* Verdict badges */
.badge-fake {
    display: inline-block;
    background: linear-gradient(135deg, #F72585, #B5179E);
    color: white; font-size: 2rem; font-weight: 800;
    padding: 0.4rem 1.8rem; border-radius: 40px;
    letter-spacing: 3px; text-align: center;
}
.badge-real {
    display: inline-block;
    background: linear-gradient(135deg, #4361EE, #3A0CA3);
    color: white; font-size: 2rem; font-weight: 800;
    padding: 0.4rem 1.8rem; border-radius: 40px;
    letter-spacing: 3px; text-align: center;
}
.badge-unknown {
    display: inline-block;
    background: #6c757d; color: white;
    font-size: 2rem; font-weight: 800;
    padding: 0.4rem 1.8rem; border-radius: 40px;
    letter-spacing: 3px; text-align: center;
}

/* Metric cards */
.metric-card {
    background: #f8f9fa; border-radius: 12px;
    padding: 1rem 1.4rem; text-align: center;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
}
.metric-val { font-size: 1.6rem; font-weight: 700; color: #4361EE; }
.metric-lbl { font-size: 0.8rem; color: #6c757d; margin-top: 2px; }

/* Confidence bar */
.conf-bar-wrap { background: #e9ecef; border-radius: 20px; height: 18px; width: 100%; }
.conf-bar-fill {
    height: 18px; border-radius: 20px;
    transition: width 0.6s ease;
}

/* Section headers */
.section-hdr {
    font-size: 1.1rem; font-weight: 700;
    color: #343a40; margin: 1.2rem 0 0.5rem;
    border-left: 4px solid #4361EE; padding-left: 10px;
}
</style>
""", unsafe_allow_html=True)


# ── Cached resource loaders ────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading models …")
def load_all_models():
    """Load pre-trained feature pipeline and all classical models from disk."""
    models = {}
    fp = None

    fp_path = MODEL_DIR / "feature_pipeline.pkl"
    if fp_path.exists():
        with open(fp_path, "rb") as f:
            fp = pickle.load(f)

    for mfile in MODEL_DIR.glob("*.pkl"):
        stem = mfile.stem
        if stem == "feature_pipeline" or stem == "tfidf_vectorizer" or stem == "meta_scaler":
            continue
        try:
            with open(mfile, "rb") as f:
                obj = pickle.load(f)
            # Friendly name mapping
            name_map = {
                "logistic_regression": "Logistic Regression",
                "naive_bayes":         "Naive Bayes",
                "random_forest":       "Random Forest",
                "voting_ensemble":     "Voting Ensemble",
            }
            display_name = name_map.get(stem, stem.replace("_", " ").title())
            models[display_name] = obj
        except Exception as exc:
            logger.warning("Could not load %s: %s", mfile, exc)

    return fp, models


def models_available(fp, models) -> bool:
    return fp is not None and len(models) > 0


# ── Prediction helper ──────────────────────────────────────────────────────────

def predict_single(text: str, model, fp: FeaturePipeline) -> dict:
    cleaned  = full_pipeline(text)
    meta_row = pd.DataFrame([compute_meta_features(text)])

    X_tfidf = fp.tfidf.transform(np.array([cleaned]))

    # Naive Bayes was trained on TF-IDF only (requires non-negative features)
    tfidf_only = getattr(model, "_tfidf_only", False)
    if fp.use_meta and fp.meta and not tfidf_only:
        from scipy.sparse import hstack, csr_matrix
        X_meta = fp.meta.transform(meta_row)
        X = hstack([X_tfidf, csr_matrix(X_meta)], format="csr")
    else:
        X = X_tfidf

    proba   = model.predict_proba(X)[0]
    label   = int(np.argmax(proba))
    verdict = "REAL" if label == 1 else "FAKE"
    conf    = float(proba[label])

    return {
        "label":      label,
        "verdict":    verdict,
        "confidence": conf,
        "proba_fake": float(proba[0]),
        "proba_real": float(proba[1]),
        "cleaned":    cleaned,
    }

# ── Sidebar ────────────────────────────────────────────────────────────────────

def render_sidebar(models: dict) -> tuple:
    st.sidebar.image(
        "https://img.icons8.com/fluency/96/news.png",
        width=64,
    )
    st.sidebar.title("⚙️  Settings")

    model_name = st.sidebar.selectbox(
        "Select model",
        options=list(models.keys()) if models else [DEFAULT_MODEL],
        index=0,
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### ℹ️  About")
    st.sidebar.info(
        "This system uses NLP and Machine Learning to classify news articles "
        "as **FAKE** or **REAL**.\n\n"
        "Models are trained on the Kaggle Fake/Real News dataset using "
        "TF-IDF features + hand-crafted NLP meta-features."
    )
    st.sidebar.markdown("---")
    show_lime = st.sidebar.checkbox("Show LIME explanation", value=True)
    show_meta = st.sidebar.checkbox("Show text analytics",  value=True)

    return model_name, show_lime, show_meta


# ── Demo articles ──────────────────────────────────────────────────────────────

DEMO_FAKE = (
    "SHOCKING BOMBSHELL: Scientists BANNED by Big Pharma for revealing secret "
    "cancer cure that THEY don't want you to know! Anonymous insider leaks "
    "documents proving government conspiracy to suppress miracle treatment. "
    "Share this before it gets DELETED!!! The mainstream media won't tell you the truth!!!"
)

DEMO_REAL = (
    "The Federal Reserve held its benchmark interest rate steady on Wednesday, "
    "pausing its rate-hiking campaign as policymakers assess the impact of "
    "previous increases on inflation and economic growth. Fed Chair Jerome Powell "
    "said in a press conference that the central bank would take a data-dependent "
    "approach to future decisions, noting that inflation has declined but remains "
    "above the 2 percent target."
)


# ── Confidence gauge ───────────────────────────────────────────────────────────

def confidence_gauge(proba_fake: float, proba_real: float):
    fig, ax = plt.subplots(figsize=(5, 0.6))
    fig.patch.set_alpha(0)
    ax.set_axis_off()
    total_w = 1.0
    ax.barh(0, proba_fake, color="#F72585", height=0.5, left=0)
    ax.barh(0, proba_real, color="#4361EE", height=0.5, left=proba_fake)
    ax.set_xlim(0, 1)
    ax.text(proba_fake / 2, 0, f"FAKE {proba_fake:.0%}",
            ha="center", va="center", color="white", fontsize=9, fontweight="bold")
    ax.text(proba_fake + proba_real / 2, 0, f"REAL {proba_real:.0%}",
            ha="center", va="center", color="white", fontsize=9, fontweight="bold")
    return fig


# ── Main application ───────────────────────────────────────────────────────────

def main():
    fp, models = load_all_models()
    trained    = models_available(fp, models)

    model_name, show_lime, show_meta = render_sidebar(models)

    # ── Header ────────────────────────────────────────────────────────────────
    st.markdown(
        "<h1 style='text-align:center; color:#4361EE;'>🔍 Fake News Detection System</h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='text-align:center; color:#6c757d; font-size:1.05rem;'>"
        "Paste a news article below and get an instant AI-powered verdict with explanation.</p>",
        unsafe_allow_html=True,
    )

    # ── Demo buttons ──────────────────────────────────────────────────────────
    col_d1, col_d2, col_spacer = st.columns([1, 1, 3])
    load_fake = col_d1.button("📋 Load FAKE example")
    load_real = col_d2.button("📋 Load REAL example")

    # ── Text input ────────────────────────────────────────────────────────────
    default_text = ""
    if load_fake:
        default_text = DEMO_FAKE
    elif load_real:
        default_text = DEMO_REAL

    user_text = st.text_area(
        "📰 Paste your news article here:",
        value     = default_text,
        height    = 200,
        max_chars = 10_000,
        placeholder = "Paste or type a news article …",
    )

    analyse_btn = st.button("🔎  Analyse Article", type="primary", use_container_width=True)

    # ── Result display ────────────────────────────────────────────────────────
    if analyse_btn:
        if not user_text.strip():
            st.warning("⚠️  Please enter some text before analysing.")
            st.stop()

        if not trained:
            st.error(
                "❌  No trained models found in `models/`.\n\n"
                "Run the training pipeline first:\n```\npython main.py\n```"
            )
            st.stop()

        model = models[model_name]

        with st.spinner("Analysing article …"):
            t0     = time.time()
            result = predict_single(user_text, model, fp)
            latency = (time.time() - t0) * 1000

        # ── Verdict badge ─────────────────────────────────────────────────────
        st.markdown("---")
        vcol, scol = st.columns([1, 2])

        badge_cls = "badge-fake" if result["verdict"] == "FAKE" else "badge-real"
        icon      = "🚫" if result["verdict"] == "FAKE" else "✅"
        vcol.markdown(
            f"<div style='text-align:center; margin-top:20px;'>"
            f"<div class='{badge_cls}'>{icon}  {result['verdict']}</div>"
            f"<p style='color:#6c757d; margin-top:12px; font-size:0.9rem;'>"
            f"Confidence: <b>{result['confidence']:.1%}</b>&nbsp;&nbsp;|&nbsp;&nbsp;"
            f"Model: <b>{model_name}</b>&nbsp;&nbsp;|&nbsp;&nbsp;"
            f"Latency: <b>{latency:.0f} ms</b></p></div>",
            unsafe_allow_html=True,
        )

        # Probability bar
        scol.markdown("<div class='section-hdr'>Probability breakdown</div>",
                      unsafe_allow_html=True)
        gauge = confidence_gauge(result["proba_fake"], result["proba_real"])
        scol.pyplot(gauge, use_container_width=True)
        plt.close()

        m1, m2, m3, m4 = scol.columns(4)
        m1.metric("FAKE %",   f"{result['proba_fake']:.1%}")
        m2.metric("REAL %",   f"{result['proba_real']:.1%}")
        word_count = len(user_text.split())
        m3.metric("Words",    word_count)
        sens_count = sum(1 for w in SENSATIONAL_WORDS if w in user_text.lower())
        m4.metric("Sensational words", sens_count)

        # ── Multi-model comparison ─────────────────────────────────────────────
        if len(models) > 1:
            with st.expander("📊  Compare all models", expanded=False):
                rows = []
                for mn, mo in models.items():
                    try:
                        r = predict_single(user_text, mo, fp)
                        rows.append({
                            "Model":      mn,
                            "Verdict":    r["verdict"],
                            "Confidence": f"{r['confidence']:.1%}",
                            "FAKE %":     f"{r['proba_fake']:.1%}",
                            "REAL %":     f"{r['proba_real']:.1%}",
                        })
                    except Exception:
                        pass
                st.dataframe(pd.DataFrame(rows), use_container_width=True)

        # ── Text analytics ────────────────────────────────────────────────────
        if show_meta:
            with st.expander("📈  Text analytics", expanded=True):
                from textblob import TextBlob
                blob   = TextBlob(user_text)
                polarity    = blob.sentiment.polarity
                subjectivity= blob.sentiment.subjectivity
                sens_words  = [w for w in SENSATIONAL_WORDS if w in user_text.lower()]
                cap_ratio   = sum(1 for c in user_text if c.isupper()) / max(len(user_text), 1)

                ac1, ac2, ac3, ac4 = st.columns(4)
                ac1.metric("Sentiment polarity",     f"{polarity:+.2f}")
                ac2.metric("Subjectivity",            f"{subjectivity:.2f}")
                ac3.metric("CAPS ratio",              f"{cap_ratio:.1%}")
                ac4.metric("Exclamation marks",       user_text.count("!"))

                if sens_words:
                    st.markdown(
                        "**⚠️  Sensational words detected:** " +
                        ", ".join(f"`{w}`" for w in sens_words[:10])
                    )
                else:
                    st.success("✅  No sensational / clickbait words detected.")

                # Sentiment interpretation
                if polarity < -0.1:
                    st.info("😠  Tone: Negative sentiment detected.")
                elif polarity > 0.1:
                    st.info("😊  Tone: Positive sentiment detected.")
                else:
                    st.info("😐  Tone: Neutral sentiment detected.")

        # ── LIME explanation ──────────────────────────────────────────────────
        if show_lime:
            with st.expander("🔬  LIME word-importance explanation", expanded=True):
                with st.spinner("Computing LIME explanation (may take a few seconds) …"):
                    try:
                        explanation = explain_with_lime(model, fp, result["cleaned"],
                                                        num_features=15, num_samples=500)
                        if explanation:
                            lime_fig = plot_lime_explanation(
                                explanation,
                                title=f"LIME — {model_name}  →  predicted {result['verdict']}",
                                save=False,
                            )
                            st.pyplot(lime_fig, use_container_width=True)
                            plt.close()

                            # Table of word weights
                            word_weights = explanation.as_list()
                            df_lime = pd.DataFrame(word_weights, columns=["Word / phrase", "LIME weight"])
                            df_lime["Direction"] = df_lime["LIME weight"].apply(
                                lambda w: "→ REAL" if w > 0 else "→ FAKE"
                            )
                            st.dataframe(
                                df_lime.style.background_gradient(
                                    subset=["LIME weight"], cmap="coolwarm", vmin=-0.5, vmax=0.5
                                ),
                                use_container_width=True,
                            )
                        else:
                            st.warning("LIME explanation unavailable (install `lime`).")
                    except Exception as exc:
                        st.error(f"LIME error: {exc}")

    # ── Footer ────────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown(
        "<p style='text-align:center; color:#adb5bd; font-size:0.8rem;'>"
        "Fake News Detection System · Built with scikit-learn, NLTK, LIME & Streamlit"
        "</p>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
