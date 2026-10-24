"""
Training + Visualisation for AI Developer Tools multi-label prediction.

Run directly:
    python train.py
    python train.py --data survey_demo.csv --out plots/ --n-estimators 300 --folds 5

What it does:
    1. Loads and preprocesses the JetBrains survey dataset
    2. Trains XGBoost MultiOutputClassifier with k-fold cross-validation
    3. Tracks all metrics with MLflow
    4. Saves 8 diagnostic plots to --out directory
    5. Prints a final CV summary

Google-style docstrings throughout.
"""

from __future__ import annotations

import argparse
import logging
import random
import sys
import time
from pathlib import Path
from typing import Optional

# ── Make sibling modules importable from any working directory ─────────────
sys.path.insert(0, str(Path(__file__).parent))

import matplotlib
matplotlib.use("Agg")   # non-interactive — works in Colab and headless terminals
import matplotlib.pyplot as plt
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import KFold
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from config import SEED, DataConfig, ModelConfig, TrainConfig
from preprocessing import (
    build_preprocessing_pipeline,
    load_raw_data,
    split_features_targets,
)

# ── Reproducibility ────────────────────────────────────────────────────────
random.seed(SEED)
np.random.seed(SEED)

logging.basicConfig(
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    level=logging.INFO,
    datefmt="%H:%M:%S",
)
logging.getLogger("mlflow").setLevel(logging.WARNING)
logging.getLogger("xgboost").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

# ── Plot theme ─────────────────────────────────────────────────────────────
PALETTE = [
    "#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3",
    "#937860", "#DA8BC3", "#8C8C8C", "#CCB974", "#64B5CD",
    "#E377C2", "#7F7F7F", "#BCBD22", "#17BECF", "#AEC7E8",
    "#FFBB78", "#98DF8A",
]
plt.rcParams.update({
    "figure.facecolor":  "#0F1117",
    "axes.facecolor":    "#1A1D27",
    "axes.edgecolor":    "#3A3D4D",
    "axes.labelcolor":   "#C9D1D9",
    "axes.titlecolor":   "#E6EDF3",
    "axes.grid":         True,
    "grid.color":        "#2D3243",
    "grid.linewidth":    0.6,
    "xtick.color":       "#8B949E",
    "ytick.color":       "#8B949E",
    "text.color":        "#C9D1D9",
    "legend.facecolor":  "#1A1D27",
    "legend.edgecolor":  "#3A3D4D",
    "legend.labelcolor": "#C9D1D9",
    "font.family":       "monospace",
})


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _short(lbl: str, n: int = 22) -> str:
    """Shorten a verbose AI-tool label for axis ticks."""
    lbl = (lbl
           .replace("openAI", "OpenAI")
           .replace(" (chatbot models)", "")
           .replace(" (all models)", "")
           .replace(" models", "")
           .replace("Gemini (Flash general purpose)", "Gemini Flash")
           .replace("Gemini (Pro Reasoning)", "Gemini Pro")
           .replace("DeepSeek (R- Reasoning)", "DeepSeek-R")
           .replace("DeepSeek (V- General purpose)", "DeepSeek-V")
           .replace("openAI Image generating", "OpenAI Image")
           .replace("Reka (Flash 3 or other Reka)", "Reka Flash"))
    return lbl[:n] + "…" if len(lbl) > n else lbl


def _save(fig: plt.Figure, out_dir: Path, name: str) -> None:
    """Save *fig* as a PNG and close it."""
    path = out_dir / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved → %s", path)


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_multilabel_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    label_names: list[str],
    prefix: str = "",
) -> dict[str, float]:
    """Compute per-label and macro-averaged multi-label classification metrics.

    Args:
        y_true: Binary ground-truth array of shape (n_samples, n_labels).
        y_pred: Binary predicted array of shape (n_samples, n_labels).
        y_prob: Predicted probability array of shape (n_samples, n_labels).
        label_names: Human-readable label strings (length == n_labels).
        prefix: Optional string prefix for metric keys (e.g. ``"val_"``).

    Returns:
        Flat dict of ``metric_name → float`` suitable for MLflow ``log_metrics``.
    """
    metrics: dict[str, float] = {}

    metrics[f"{prefix}macro_f1"]  = float(f1_score(y_true, y_pred, average="macro",   zero_division=0))
    metrics[f"{prefix}micro_f1"]  = float(f1_score(y_true, y_pred, average="micro",   zero_division=0))
    metrics[f"{prefix}sample_f1"] = float(f1_score(y_true, y_pred, average="samples", zero_division=0))

    try:
        metrics[f"{prefix}macro_auc_roc"] = float(
            roc_auc_score(y_true, y_prob, average="macro", multi_class="ovr")
        )
    except ValueError as exc:
        logger.warning("AUC-ROC failed: %s", exc)

    try:
        metrics[f"{prefix}mean_avg_precision"] = float(
            average_precision_score(y_true, y_prob, average="macro")
        )
    except ValueError as exc:
        logger.warning("MAP failed: %s", exc)

    for lbl, score in zip(label_names, f1_score(y_true, y_pred, average=None, zero_division=0)):
        safe = lbl.replace(" ", "_").replace("(", "").replace(")", "").replace(":", "")
        metrics[f"{prefix}f1__{safe}"] = float(score)

    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# Model builder
# ─────────────────────────────────────────────────────────────────────────────

def build_model(model_cfg: ModelConfig) -> MultiOutputClassifier:
    """Instantiate a MultiOutputClassifier wrapping XGBClassifier.

    Args:
        model_cfg: :class:`ModelConfig` with XGBoost hyperparameters.

    Returns:
        Unfitted ``MultiOutputClassifier`` (one XGBoost estimator per label).
    """
    base = XGBClassifier(
        n_estimators=model_cfg.n_estimators,
        max_depth=model_cfg.max_depth,
        learning_rate=model_cfg.learning_rate,
        subsample=model_cfg.subsample,
        colsample_bytree=model_cfg.colsample_bytree,
        reg_alpha=model_cfg.reg_alpha,
        reg_lambda=model_cfg.reg_lambda,
        n_jobs=model_cfg.n_jobs,
        random_state=model_cfg.random_state,
        eval_metric="logloss",
        tree_method="hist",
    )
    return MultiOutputClassifier(base, n_jobs=model_cfg.n_jobs)


# ─────────────────────────────────────────────────────────────────────────────
# Plot functions  (each returns Figure; caller saves via _save)
# ─────────────────────────────────────────────────────────────────────────────

def _plot_label_distribution(y: np.ndarray, label_names: list[str]) -> plt.Figure:
    """Horizontal bar chart — how many respondents selected each AI tool."""
    counts = y.sum(axis=0).astype(int)
    order  = np.argsort(counts)[::-1]

    fig, ax = plt.subplots(figsize=(13, 6))
    bars = ax.barh(
        [_short(label_names[i]) for i in order],
        counts[order],
        color=[PALETTE[i % len(PALETTE)] for i in range(len(order))],
        height=0.7, edgecolor="none",
    )
    ax.bar_label(bars, padding=4, fmt="%d", fontsize=8, color="#8B949E")
    ax.set_xlabel("Positive samples")
    ax.set_title("Label Distribution — AI Tool Usage in Survey", pad=14, fontsize=13)
    ax.invert_yaxis()
    ax.spines[["top", "right", "left"]].set_visible(False)
    fig.tight_layout()
    return fig


def _plot_per_label_f1(
    y_true: np.ndarray, y_pred: np.ndarray, label_names: list[str]
) -> plt.Figure:
    """Bar chart of per-label F1 with macro-average reference line."""
    per_f1   = f1_score(y_true, y_pred, average=None, zero_division=0)
    macro_f1 = per_f1.mean()
    order    = np.argsort(per_f1)[::-1]
    colors   = ["#55A868" if per_f1[i] >= macro_f1 else "#C44E52" for i in order]

    fig, ax = plt.subplots(figsize=(13, 6))
    ax.bar(range(len(order)), per_f1[order], color=colors, edgecolor="none", width=0.7)
    ax.axhline(macro_f1, color="#FFD700", linewidth=1.4,
               linestyle="--", label=f"Macro avg F1 = {macro_f1:.3f}")
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([_short(label_names[i]) for i in order],
                       rotation=40, ha="right", fontsize=8)
    ax.set_ylabel("F1 Score")
    ax.set_ylim(0, min(1.05, per_f1.max() + 0.15))
    ax.set_title("Per-Label F1 Score", pad=14, fontsize=13)
    ax.legend(loc="upper right", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def _plot_cv_fold_stability(fold_metrics: list[dict[str, float]]) -> plt.Figure:
    """Box-and-strip plot showing metric variance across CV folds."""
    keys      = ["val_macro_f1", "val_micro_f1", "val_macro_auc_roc", "val_mean_avg_precision"]
    available = [k for k in keys if any(k in fm for fm in fold_metrics)]
    data      = {k: [fm[k] for fm in fold_metrics if k in fm] for k in available}
    positions = np.arange(len(available))

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.boxplot(
        [data[k] for k in available],
        positions=positions, widths=0.4, patch_artist=True,
        medianprops=dict(color="#FFD700", linewidth=2),
        boxprops=dict(facecolor="#2D3243", edgecolor="#4C72B0"),
        whiskerprops=dict(color="#4C72B0"),
        capprops=dict(color="#4C72B0"),
        flierprops=dict(marker="o", color="#C44E52", markersize=5),
    )
    rng = np.random.default_rng(SEED)
    for i, k in enumerate(available):
        jitter = rng.uniform(-0.08, 0.08, len(data[k]))
        ax.scatter(positions[i] + jitter, data[k],
                   color=PALETTE[i % len(PALETTE)], zorder=3,
                   s=50, edgecolors="white", linewidths=0.5)
    ax.set_xticks(positions)
    ax.set_xticklabels([k.replace("val_", "").replace("_", " ") for k in available], fontsize=9)
    ax.set_ylabel("Score")
    ax.set_title("CV Fold Stability — Metric Variance Across Folds", pad=14, fontsize=13)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def _plot_roc_curves(
    y_true: np.ndarray, y_prob: np.ndarray, label_names: list[str]
) -> plt.Figure:
    """Overlaid per-label ROC curves with interpolated macro average."""
    fig, ax     = plt.subplots(figsize=(9, 8))
    all_fpr     = np.linspace(0, 1, 300)
    interp_tprs = []

    for j, lbl in enumerate(label_names):
        if len(np.unique(y_true[:, j])) < 2:
            continue
        fpr, tpr, _ = roc_curve(y_true[:, j], y_prob[:, j])
        auc          = roc_auc_score(y_true[:, j], y_prob[:, j])
        interp_tprs.append(np.interp(all_fpr, fpr, tpr))
        ax.plot(fpr, tpr, lw=0.9, alpha=0.5,
                color=PALETTE[j % len(PALETTE)],
                label=f"{_short(lbl)} ({auc:.2f})")

    if interp_tprs:
        mean_tpr = np.mean(interp_tprs, axis=0)
        ax.plot(all_fpr, mean_tpr, color="#FFD700", lw=2.5,
                label=f"Macro avg ({np.mean(mean_tpr):.2f})", zorder=5)

    ax.plot([0, 1], [0, 1], ":", color="#555", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — Per Label + Macro Average", pad=14, fontsize=13)
    ax.legend(loc="lower right", fontsize=7, ncol=2)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def _plot_pr_curves(
    y_true: np.ndarray, y_prob: np.ndarray, label_names: list[str]
) -> plt.Figure:
    """Overlaid per-label Precision-Recall curves with macro mAP annotation."""
    fig, ax = plt.subplots(figsize=(9, 8))
    aps     = []

    for j, lbl in enumerate(label_names):
        if len(np.unique(y_true[:, j])) < 2:
            continue
        prec, rec, _ = precision_recall_curve(y_true[:, j], y_prob[:, j])
        ap            = average_precision_score(y_true[:, j], y_prob[:, j])
        aps.append(ap)
        ax.plot(rec, prec, lw=0.9, alpha=0.55,
                color=PALETTE[j % len(PALETTE)],
                label=f"{_short(lbl)} (AP={ap:.2f})")

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Precision-Recall Curves  |  mAP = {np.mean(aps):.3f}", pad=14, fontsize=13)
    ax.legend(loc="upper right", fontsize=7, ncol=2)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def _plot_probability_distributions(
    y_true: np.ndarray, y_prob: np.ndarray, label_names: list[str]
) -> plt.Figure:
    """Per-label probability histograms split by true class (calibration check)."""
    n_cols = 4
    n      = len(label_names)
    n_rows = int(np.ceil(n / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 14))
    bins = np.linspace(0, 1, 25)

    for j, lbl in enumerate(label_names):
        ax  = axes.flatten()[j]
        pos = y_prob[y_true[:, j] == 1, j]
        neg = y_prob[y_true[:, j] == 0, j]
        ax.hist(neg, bins=bins, color="#C44E52", alpha=0.65,
                label="Negative", density=True, edgecolor="none")
        ax.hist(pos, bins=bins, color="#55A868", alpha=0.65,
                label="Positive", density=True, edgecolor="none")
        ax.set_title(_short(lbl, 20), fontsize=7.5, pad=4)
        ax.set_xlim(0, 1)
        ax.tick_params(labelsize=6)
        ax.spines[["top", "right"]].set_visible(False)
        if j == 0:
            ax.legend(fontsize=6, loc="upper center")

    for ax in axes.flatten()[n:]:
        ax.set_visible(False)

    fig.suptitle(
        "Predicted Probability Distributions per Label\n"
        "(green = true positives  |  red = true negatives)",
        fontsize=12, y=1.01,
    )
    fig.tight_layout()
    return fig


def _plot_feature_importance(model: MultiOutputClassifier, pipe: Pipeline,
                              top_n: int = 25) -> plt.Figure:
    """Top-N features by mean XGBoost gain, averaged across all label classifiers."""
    importances = np.mean(
        [est.feature_importances_ for est in model.estimators_], axis=0
    )

    feature_names: list[str] = []
    try:
        feature_names = list(pipe.named_steps["preprocessor"].get_feature_names_out())
    except Exception:
        feature_names = [f"feat_{i}" for i in range(len(importances))]

    def _clean(name: str) -> str:
        for pfx in ("numeric__", "categorical__", "multi_hot__", "binary__"):
            name = name.replace(pfx, "")
        return name[:40]

    feature_names = [_clean(n) for n in feature_names]
    if len(feature_names) != len(importances):
        feature_names = [f"feat_{i}" for i in range(len(importances))]

    idx        = np.argsort(importances)[-top_n:]
    top_names  = [feature_names[i] for i in idx]
    top_scores = importances[idx]
    norm       = top_scores / top_scores.max()

    fig, ax = plt.subplots(figsize=(12, 9))
    bars = ax.barh(top_names, top_scores,
                   color=plt.cm.YlOrRd(norm), edgecolor="none", height=0.7)
    ax.bar_label(bars, fmt="%.4f", padding=4, fontsize=7, color="#8B949E")
    ax.set_xlabel("Mean importance (gain)")
    ax.set_title(f"Top {top_n} Feature Importances\n(averaged across all label classifiers)",
                 pad=14, fontsize=13)
    ax.spines[["top", "right", "left"]].set_visible(False)
    fig.tight_layout()
    return fig


def _plot_label_cooccurrence(y: np.ndarray, label_names: list[str]) -> plt.Figure:
    """Heatmap of Jaccard similarity between every pair of AI tool labels."""
    short = [_short(l, 18) for l in label_names]
    n     = y.shape[1]
    cooc  = (y.T @ y).astype(float)
    denom = np.maximum(cooc.diagonal()[:, None], cooc.diagonal()[None, :])
    with np.errstate(divide="ignore", invalid="ignore"):
        cooc = np.where(denom > 0, cooc / denom, 0.0)
    np.fill_diagonal(cooc, 1.0)

    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(cooc, cmap="Blues", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(short, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(short, fontsize=7)
    for i in range(n):
        for j in range(n):
            val   = cooc[i, j]
            text  = f"{val:.2f}" if val > 0.01 else ""
            color = "white" if val > 0.6 else "#C9D1D9"
            ax.text(j, i, text, ha="center", va="center", fontsize=5.5, color=color)
    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="Jaccard similarity")
    ax.set_title("Label Co-occurrence (Jaccard similarity)", pad=14, fontsize=13)
    fig.tight_layout()
    return fig


def save_all_plots(
    y: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    label_names: list[str],
    model: MultiOutputClassifier,
    pipe: Pipeline,
    fold_metrics: list[dict[str, float]],
    out_dir: Path,
) -> None:
    """Render and save all 8 diagnostic plots to *out_dir*.

    Args:
        y: Full binary label matrix (n_samples, n_labels).
        y_pred: Binary predictions from the best-fold model.
        y_prob: Predicted probabilities from the best-fold model.
        label_names: Ordered AI tool name strings.
        model: Best-fold fitted ``MultiOutputClassifier``.
        pipe: Best-fold fitted preprocessing ``Pipeline``.
        fold_metrics: List of per-fold metric dicts collected during CV.
        out_dir: Directory where PNG files will be written.
    """
    plots = [
        ("label_distribution",        lambda: _plot_label_distribution(y, label_names)),
        ("per_label_f1",              lambda: _plot_per_label_f1(y, y_pred, label_names)),
        ("cv_fold_stability",         lambda: _plot_cv_fold_stability(fold_metrics)),
        ("roc_curves",                lambda: _plot_roc_curves(y, y_prob, label_names)),
        ("pr_curves",                 lambda: _plot_pr_curves(y, y_prob, label_names)),
        ("probability_distributions", lambda: _plot_probability_distributions(y, y_prob, label_names)),
        ("label_cooccurrence",        lambda: _plot_label_cooccurrence(y, label_names)),
        ("feature_importance",        lambda: _plot_feature_importance(model, pipe)),
    ]
    for name, fn in plots:
        logger.info("Rendering %s …", name)
        try:
            _save(fn(), out_dir, name)
        except Exception as exc:
            logger.warning("Skipped %s: %s", name, exc)


# ─────────────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────────────

def train(
    data_cfg: Optional[DataConfig] = None,
    model_cfg: Optional[ModelConfig] = None,
    train_cfg: Optional[TrainConfig] = None,
    plots_dir: Optional[Path] = None,
) -> dict[str, float]:
    """Full training loop: CV + MLflow tracking + diagnostic plots.

    Pipeline per fold:
      1. Fit preprocessing on train split, transform both splits.
      2. Fit ``MultiOutputClassifier`` on encoded train split.
      3. Evaluate on validation split; log metrics to MLflow.
      4. Track best fold by macro F1; retain its pipeline + predictions.

    After all folds:
      5. Aggregate mean / std CV metrics; log to MLflow.
      6. Save best-fold model as MLflow artifact.
      7. Render and save all 8 plots to *plots_dir*.

    Args:
        data_cfg: Data configuration (defaults used if None).
        model_cfg: Model configuration (defaults used if None).
        train_cfg: Training / MLflow configuration (defaults used if None).
        plots_dir: Directory for PNG outputs. Defaults to ``TrainConfig.output_dir / "plots"``.

    Returns:
        Dict of mean cross-validation metrics (keys prefixed ``cv_mean_``).
    """
    data_cfg  = data_cfg  or DataConfig()
    model_cfg = model_cfg or ModelConfig()
    train_cfg = train_cfg or TrainConfig()
    plots_dir = plots_dir or (train_cfg.output_dir / "plots")

    train_cfg.output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Load ───────────────────────────────────────────────────────────
    logger.info("Loading data from %s", data_cfg.data_path)
    df = load_raw_data(data_cfg.data_path)
    X, y, label_names = split_features_targets(df, data_cfg)
    logger.info("Dataset: X=%s  y=%s  labels=%d", X.shape, y.shape, len(label_names))

    # ── 2. MLflow ─────────────────────────────────────────────────────────
    mlflow.set_experiment(train_cfg.mlflow_experiment)
    run_name = train_cfg.mlflow_run_name or f"cv_{train_cfg.cv_folds}fold_{int(time.time())}"

    with mlflow.start_run(run_name=run_name) as run:
        logger.info("MLflow run: %s", run.info.run_id)
        mlflow.log_params({
            "n_splits":        train_cfg.cv_folds,
            "n_labels":        len(label_names),
            "n_samples":       len(X),
            "n_features_raw":  X.shape[1],
            "min_label_freq":  data_cfg.min_label_freq,
            "n_estimators":    model_cfg.n_estimators,
            "max_depth":       model_cfg.max_depth,
            "learning_rate":   model_cfg.learning_rate,
            "subsample":       model_cfg.subsample,
            "seed":            SEED,
        })

        # ── 3. Cross-validation ───────────────────────────────────────────
        kf = KFold(n_splits=train_cfg.cv_folds, shuffle=True, random_state=SEED)
        fold_metrics:   list[dict[str, float]] = []
        best_macro_f1   = -np.inf
        best_pipeline:  Optional[Pipeline]                = None
        best_clf:       Optional[MultiOutputClassifier]   = None
        best_preproc:   Optional[Pipeline]                = None
        best_y_val:     Optional[np.ndarray]              = None
        best_y_pred:    Optional[np.ndarray]              = None
        best_y_prob:    Optional[np.ndarray]              = None

        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X, y[:, 0])):
            t0 = time.perf_counter()
            logger.info("── Fold %d / %d ──", fold_idx + 1, train_cfg.cv_folds)

            X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_val = y[train_idx],       y[val_idx]

            preproc_pipe, _ = build_preprocessing_pipeline(X_tr, data_cfg)
            X_tr_enc  = preproc_pipe.fit_transform(X_tr)
            X_val_enc = preproc_pipe.transform(X_val)

            if hasattr(X_tr_enc, "toarray"):
                X_tr_enc  = X_tr_enc.toarray()
                X_val_enc = X_val_enc.toarray()

            logger.info("Encoded dim: %d", X_tr_enc.shape[1])

            clf = build_model(model_cfg)
            clf.fit(X_tr_enc, y_tr)

            y_pred = clf.predict(X_val_enc)
            y_prob = np.column_stack(
                [est.predict_proba(X_val_enc)[:, 1] for est in clf.estimators_]
            )

            fold_m = compute_multilabel_metrics(y_val, y_pred, y_prob, label_names, prefix="val_")
            fold_m["fold"]            = float(fold_idx + 1)
            fold_m["fold_duration_s"] = time.perf_counter() - t0
            fold_metrics.append(fold_m)

            mlflow.log_metrics({k: v for k, v in fold_m.items() if k != "fold"}, step=fold_idx)

            logger.info(
                "Fold %d — macro F1: %.4f | micro F1: %.4f | AUC-ROC: %.4f | %.1fs",
                fold_idx + 1,
                fold_m.get("val_macro_f1", 0),
                fold_m.get("val_micro_f1", 0),
                fold_m.get("val_macro_auc_roc", 0),
                fold_m["fold_duration_s"],
            )

            if fold_m.get("val_macro_f1", 0.0) > best_macro_f1:
                best_macro_f1  = fold_m["val_macro_f1"]
                best_clf       = clf
                best_preproc   = preproc_pipe
                best_pipeline  = Pipeline([("preprocessor", preproc_pipe), ("classifier", clf)])
                best_y_val     = y_val
                best_y_pred    = y_pred
                best_y_prob    = y_prob
                logger.info("New best fold %d (macro F1 = %.4f)", fold_idx + 1, best_macro_f1)

        # ── 4. Aggregate CV metrics ───────────────────────────────────────
        scalar_keys = [k for k in fold_metrics[0]
                       if k not in ("fold", "fold_duration_s") and "f1__" not in k]
        mean_metrics: dict[str, float] = {}
        for k in scalar_keys:
            vals = [fm[k] for fm in fold_metrics if k in fm]
            mean_metrics[f"cv_mean_{k}"] = float(np.mean(vals))
            mean_metrics[f"cv_std_{k}"]  = float(np.std(vals))

        mlflow.log_metrics(mean_metrics)
        logger.info("CV summary: %s", {k: f"{v:.4f}" for k, v in mean_metrics.items()})

        # ── 5. Log best model ─────────────────────────────────────────────
        if best_pipeline is not None:
            mlflow.sklearn.log_model(
                best_pipeline,
                artifact_path=train_cfg.model_artifact_name,
                input_example=X.head(3),
            )
            logger.info("Best pipeline logged as '%s'", train_cfg.model_artifact_name)

        mlflow.log_param("label_names", "|".join(label_names))
        mlflow.set_tag("framework", "xgboost + sklearn MultiOutputClassifier")
        mlflow.set_tag("task", "multilabel_classification")

        # ── 6. Plots — use full dataset predictions from best-fold model ──
        logger.info("Generating plots → %s", plots_dir)
        if best_clf is not None and best_preproc is not None:
            # Re-encode the full dataset with the best fold's preprocessor
            X_full_enc = best_preproc.transform(X)
            if hasattr(X_full_enc, "toarray"):
                X_full_enc = X_full_enc.toarray()
            y_full_pred = best_clf.predict(X_full_enc)
            y_full_prob = np.column_stack(
                [est.predict_proba(X_full_enc)[:, 1] for est in best_clf.estimators_]
            )
            save_all_plots(
                y=y,
                y_pred=y_full_pred,
                y_prob=y_full_prob,
                label_names=label_names,
                model=best_clf,
                pipe=best_preproc,
                fold_metrics=fold_metrics,
                out_dir=plots_dir,
            )

    return mean_metrics


# ─────────────────────────────────────────────────────────────────────────────
# CLI entrypoint
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train AI tools predictor and generate plots.")
    parser.add_argument("--data",         default="survey_demo.csv",
                        help="Path to survey CSV  (default: survey_demo.csv)")
    parser.add_argument("--out",          default="plots",
                        help="Directory for plot PNGs  (default: plots/)")
    parser.add_argument("--n-estimators", type=int, default=100,
                        help="XGBoost n_estimators  (default: 100)")
    parser.add_argument("--folds",        type=int, default=5,
                        help="CV folds  (default: 5)")
    args = parser.parse_args()

    results = train(
        data_cfg  = DataConfig(data_path=Path(args.data)),
        model_cfg = ModelConfig(n_estimators=args.n_estimators),
        train_cfg = TrainConfig(cv_folds=args.folds),
        plots_dir = Path(args.out),
    )

    print("\n=== Final CV Metrics ===")
    for k, v in sorted(results.items()):
        print(f"  {k}: {v:.4f}")
