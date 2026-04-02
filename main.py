"""
main.py
-------
End-to-end pipeline runner for the Fake News Detection System.

Execution order
  1. Load / generate dataset
  2. Preprocess text + compute meta-features
  3. Split → train / val / test
  4. Build feature pipeline (TF-IDF + meta)
  5. Train all classical models  (+ optionally LSTM, BERT)
  6. Evaluate on test set & generate report + visualisations
  7. Interpretability: LIME, top-words, word-clouds
  8. Save all artefacts

Usage
-----
  python main.py                              # default: classical models only
  python main.py --lstm                       # also train BiLSTM
  python main.py --bert                       # also fine-tune DistilBERT
  python main.py --synthetic 20000            # generate 20 k synthetic rows
  python main.py --skip-interp               # skip interpretability (faster)
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np

# ── Logging setup ──────────────────────────────────────────────────────────────
logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt = "%H:%M:%S",
)
logger = logging.getLogger("main")

# ── Project imports ────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from config          import RANDOM_SEED, MODEL_DIR
from data.data_loader  import load_dataset
from src.preprocessing import preprocess_dataframe
from src.trainer       import Trainer
from src.evaluator     import Evaluator
from src.interpretability import (
    explain_with_lime, plot_lime_explanation,
    lr_top_words, generate_wordcloud,
)

np.random.seed(RANDOM_SEED)


# ── CLI arguments ──────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Fake News Detection — Training Pipeline")
    p.add_argument("--lstm",         action="store_true", help="Train BiLSTM model")
    p.add_argument("--bert",         action="store_true", help="Fine-tune DistilBERT")
    p.add_argument("--smote",        action="store_true", help="Apply SMOTE over-sampling")
    p.add_argument("--no-meta",      action="store_true", help="Disable meta-features")
    p.add_argument("--skip-interp",  action="store_true", help="Skip interpretability step")
    p.add_argument("--synthetic",    type=int, default=0,
                   help="Generate N synthetic rows (0 = try Kaggle files first)")
    return p.parse_args()


# ── Pipeline steps ─────────────────────────────────────────────────────────────

def step_load(args) -> object:
    logger.info("━━━ STEP 1 — Data Loading ━━━")
    if args.synthetic > 0:
        from data.data_loader import generate_synthetic_dataset
        half = args.synthetic // 2
        df = generate_synthetic_dataset(n_fake=half, n_real=half)
    else:
        df = load_dataset(synthetic_fallback=True, n_synthetic=10_000)
    logger.info("Dataset shape: %s  |  label distribution: %s",
                df.shape, df["label"].value_counts().to_dict())
    return df


def step_preprocess(df) -> tuple:
    logger.info("━━━ STEP 2 — Preprocessing ━━━")
    df_proc, X_text, y = preprocess_dataframe(df)
    logger.info("Preprocessing complete. Cleaned-text samples:\n%s",
                "\n".join(f"  [{y[i]}] {X_text[i][:80]}…" for i in range(min(3, len(X_text)))))
    return df_proc, X_text, y


def step_train(df_proc, X_text, y, args) -> Trainer:
    logger.info("━━━ STEPS 3-5 — Feature Engineering + Training ━━━")
    trainer = Trainer(
        use_meta   = not args.no_meta,
        use_smote  = args.smote,
        train_lstm = args.lstm,
        train_bert = args.bert,
    )
    trainer.fit(df_proc, X_text, y)
    return trainer


def step_evaluate(trainer: Trainer) -> Evaluator:
    logger.info("━━━ STEP 6 — Evaluation ━━━")
    evaluator = Evaluator(y_true=trainer.splits["y_test"])
    preds = trainer.get_test_predictions()

    for mname, pred_dict in preds.items():
        evaluator.add_model(
            name    = mname,
            y_pred  = pred_dict["y_pred"],
            y_proba = pred_dict.get("y_proba"),
        )

    report_df = evaluator.generate_report()
    logger.info("\n%s\n", report_df.to_string(index=False))
    logger.info("Best model (by F1-weighted): %s", evaluator.best_model_name())
    return evaluator


def step_interpretability(trainer: Trainer):
    logger.info("━━━ STEP 7 — Interpretability ━━━")
    fp = trainer.feature_pipeline
    lr_model = trainer.trained_models.get("Logistic Regression")

    # Top words (LR coefficient inspection — free, no sampling)
    if lr_model:
        try:
            top_words_df = lr_top_words(lr_model, fp)
            logger.info("Top discriminative words saved.")
        except Exception as exc:
            logger.warning("Top-words analysis failed: %s", exc)

    # Word-clouds
    try:
        generate_wordcloud(trainer.splits["X_train"], trainer.splits["y_train"], target_label=0)
        generate_wordcloud(trainer.splits["X_train"], trainer.splits["y_train"], target_label=1)
        logger.info("Word-clouds generated.")
    except Exception as exc:
        logger.warning("Word-cloud generation failed: %s", exc)

    # LIME — explain a few test examples
    if lr_model:
        try:
            X_test = trainer.splits["X_test"]
            y_test = trainer.splits["y_test"]
            for i in range(min(2, len(X_test))):
                explanation = explain_with_lime(lr_model, fp, X_test[i])
                if explanation:
                    label_name = "FAKE" if y_test[i] == 0 else "REAL"
                    plot_lime_explanation(
                        explanation,
                        title=f"LIME — Test sample #{i} (True: {label_name})",
                        save=True,
                    )
            logger.info("LIME explanations generated.")
        except Exception as exc:
            logger.warning("LIME explanations failed: %s", exc)


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    t0   = time.time()

    df           = step_load(args)
    df_proc, X_text, y = step_preprocess(df)
    trainer      = step_train(df_proc, X_text, y, args)
    evaluator    = step_evaluate(trainer)

    if not args.skip_interp:
        step_interpretability(trainer)

    elapsed = time.time() - t0
    logger.info("━━━ Pipeline complete in %.1f s ━━━", elapsed)
    logger.info("Artefacts saved to:")
    logger.info("  Models     → %s", MODEL_DIR)
    logger.info("  Reports    → reports/evaluation_report.csv")
    logger.info("  Visuals    → visualizations/")
    logger.info("  Launch app → streamlit run app.py")


if __name__ == "__main__":
    main()
