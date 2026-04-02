"""
src/evaluator.py
-----------------
Comprehensive model evaluation and visualisation.

Metrics computed
  • Accuracy, Precision, Recall, F1 (macro + weighted)
  • ROC-AUC
  • Confusion matrix
  • Classification report (per-class breakdown)

Visualisations generated (saved to visualizations/)
  • Confusion matrix heatmap (per model)
  • ROC curves (all models on same chart)
  • Precision-Recall curves
  • Model comparison bar chart
  • Training-history plot (for LSTM / BERT)
"""

import sys
import logging
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, precision_recall_curve,
    confusion_matrix, classification_report,
)

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import VIZ_DIR, REPORT_DIR, RANDOM_SEED

logger = logging.getLogger(__name__)

plt.rcParams.update({
    "figure.dpi":    130,
    "font.family":   "DejaVu Sans",
    "axes.spines.top":   False,
    "axes.spines.right": False,
})
PALETTE = ["#4361EE", "#F72585", "#7209B7", "#3A0CA3", "#4CC9F0"]


# ── Single-model metrics ───────────────────────────────────────────────────────

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                    y_proba: Optional[np.ndarray] = None) -> dict:
    metrics = {
        "accuracy":           accuracy_score(y_true, y_pred),
        "precision_macro":    precision_score(y_true, y_pred, average="macro",    zero_division=0),
        "recall_macro":       recall_score(y_true, y_pred,    average="macro",    zero_division=0),
        "f1_macro":           f1_score(y_true, y_pred,        average="macro",    zero_division=0),
        "precision_weighted": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall_weighted":    recall_score(y_true, y_pred,    average="weighted", zero_division=0),
        "f1_weighted":        f1_score(y_true, y_pred,        average="weighted", zero_division=0),
    }
    if y_proba is not None:
        try:
            metrics["roc_auc"] = roc_auc_score(y_true, y_proba)
        except ValueError:
            metrics["roc_auc"] = float("nan")
    return metrics


# ── Visualisations ────────────────────────────────────────────────────────────

def plot_confusion_matrix(y_true, y_pred, model_name: str, save: bool = True):
    cm  = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["FAKE", "REAL"],
        yticklabels=["FAKE", "REAL"],
        ax=ax, linewidths=0.5,
        annot_kws={"size": 14, "weight": "bold"},
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual",    fontsize=12)
    ax.set_title(f"Confusion Matrix — {model_name}", fontsize=13, pad=12)
    plt.tight_layout()
    if save:
        path = VIZ_DIR / f"confusion_matrix_{model_name.replace(' ', '_').lower()}.png"
        fig.savefig(path, bbox_inches="tight")
        logger.info("Saved confusion matrix → %s", path)
    return fig


def plot_roc_curves(
    y_true: np.ndarray,
    proba_dict: Dict[str, np.ndarray],
    save: bool = True,
):
    """
    proba_dict: {model_name: y_proba_positive_class}
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random (AUC=0.50)")
    for (mname, y_proba), colour in zip(proba_dict.items(), PALETTE):
        try:
            fpr, tpr, _ = roc_curve(y_true, y_proba)
            auc = roc_auc_score(y_true, y_proba)
            ax.plot(fpr, tpr, lw=2, color=colour, label=f"{mname} (AUC={auc:.3f})")
        except Exception:
            pass
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate",  fontsize=11)
    ax.set_title("ROC Curves — All Models", fontsize=13, pad=12)
    ax.legend(fontsize=9, loc="lower right")
    plt.tight_layout()
    if save:
        path = VIZ_DIR / "roc_curves.png"
        fig.savefig(path, bbox_inches="tight")
        logger.info("Saved ROC curves → %s", path)
    return fig


def plot_precision_recall_curves(
    y_true: np.ndarray,
    proba_dict: Dict[str, np.ndarray],
    save: bool = True,
):
    fig, ax = plt.subplots(figsize=(7, 5))
    for (mname, y_proba), colour in zip(proba_dict.items(), PALETTE):
        try:
            prec, rec, _ = precision_recall_curve(y_true, y_proba)
            ax.plot(rec, prec, lw=2, color=colour, label=mname)
        except Exception:
            pass
    ax.set_xlabel("Recall",    fontsize=11)
    ax.set_ylabel("Precision", fontsize=11)
    ax.set_title("Precision-Recall Curves — All Models", fontsize=13, pad=12)
    ax.legend(fontsize=9)
    plt.tight_layout()
    if save:
        path = VIZ_DIR / "precision_recall_curves.png"
        fig.savefig(path, bbox_inches="tight")
        logger.info("Saved P-R curves → %s", path)
    return fig


def plot_model_comparison(metrics_df: pd.DataFrame, save: bool = True):
    """
    metrics_df: columns = ["Model", "Accuracy", "F1 (weighted)", "AUC-ROC", ...]
    """
    plot_cols = ["Accuracy", "F1 (weighted)", "Precision (weighted)", "Recall (weighted)"]
    avail = [c for c in plot_cols if c in metrics_df.columns]
    melted = metrics_df.melt(id_vars="Model", value_vars=avail,
                              var_name="Metric", value_name="Score")

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=melted, x="Metric", y="Score", hue="Model",
                palette=PALETTE[:len(metrics_df)], ax=ax)
    ax.set_ylim(0, 1.05)
    ax.set_title("Model Comparison — Test-Set Metrics", fontsize=13, pad=12)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax.legend(loc="lower right", fontsize=9)
    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", fontsize=7, padding=2)
    plt.tight_layout()
    if save:
        path = VIZ_DIR / "model_comparison.png"
        fig.savefig(path, bbox_inches="tight")
        logger.info("Saved model comparison → %s", path)
    return fig


def plot_training_history(history: dict, model_name: str, save: bool = True):
    """history: {"train_loss": [...], "val_loss": [...], "train_acc": [...], "val_acc": [...]}"""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    epochs = range(1, len(history.get("train_loss", [])) + 1)

    axes[0].plot(epochs, history["train_loss"], "b-o", label="Train")
    if "val_loss" in history:
        axes[0].plot(epochs, history["val_loss"], "r-o", label="Val")
    axes[0].set_title(f"{model_name} — Loss");  axes[0].legend()

    if "train_acc" in history:
        axes[1].plot(epochs, history["train_acc"], "b-o", label="Train")
        if "val_acc" in history:
            axes[1].plot(epochs, history["val_acc"], "r-o", label="Val")
        axes[1].set_title(f"{model_name} — Accuracy"); axes[1].legend()

    plt.tight_layout()
    if save:
        path = VIZ_DIR / f"training_history_{model_name.lower()}.png"
        fig.savefig(path, bbox_inches="tight")
    return fig


# ── Evaluation orchestrator ────────────────────────────────────────────────────

class Evaluator:
    """
    Runs evaluation for all trained models and generates
    the full report + visualisations.
    """

    def __init__(self, y_true: np.ndarray):
        self.y_true       = y_true
        self.all_metrics  : Dict[str, dict] = {}
        self.all_probas   : Dict[str, np.ndarray] = {}

    def add_model(
        self,
        name: str,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None,
    ):
        """Register one model's predictions."""
        m = compute_metrics(self.y_true, y_pred, y_proba)
        self.all_metrics[name] = m
        if y_proba is not None:
            self.all_probas[name] = y_proba

        logger.info(
            "[%s] acc=%.4f  f1=%.4f  auc=%.4f",
            name, m["accuracy"], m["f1_weighted"],
            m.get("roc_auc", float("nan")),
        )
        print(f"\n{'─'*55}")
        print(f"  Model: {name}")
        print(f"{'─'*55}")
        print(classification_report(self.y_true, y_pred, target_names=["FAKE", "REAL"]))
        plot_confusion_matrix(self.y_true, y_pred, name)

    def generate_report(self) -> pd.DataFrame:
        """Build a summary DataFrame and save as CSV + visualisations."""
        rows = []
        for name, m in self.all_metrics.items():
            rows.append({
                "Model":                name,
                "Accuracy":             m["accuracy"],
                "Precision (weighted)": m["precision_weighted"],
                "Recall (weighted)":    m["recall_weighted"],
                "F1 (weighted)":        m["f1_weighted"],
                "AUC-ROC":              m.get("roc_auc", float("nan")),
            })
        df = pd.DataFrame(rows).sort_values("F1 (weighted)", ascending=False)
        df.to_csv(REPORT_DIR / "evaluation_report.csv", index=False)
        logger.info("Evaluation report saved → %s", REPORT_DIR / "evaluation_report.csv")

        # Plots
        plot_model_comparison(df)
        if self.all_probas:
            plot_roc_curves(self.y_true, self.all_probas)
            plot_precision_recall_curves(self.y_true, self.all_probas)

        return df

    def best_model_name(self) -> str:
        best = max(self.all_metrics.items(), key=lambda kv: kv[1]["f1_weighted"])
        return best[0]
