"""
Preprocessing pipeline for AI Developer Tools Prediction.

Handles:
- Multi-label target construction from semicolon-delimited survey answers
- Ordinal / categorical encoding with cardinality capping
- Multi-hot encoding for multi-value survey columns
- Numeric imputation + robust scaling
- Binary flag normalisation
- Full sklearn Pipeline composition

Google-style docstrings throughout.
"""

from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MultiLabelBinarizer, OneHotEncoder, RobustScaler

from config import (
    AI_TOOL_LABELS,
    BINARY_COLS,
    CATEGORICAL_COLS,
    MULTI_HOT_COLS,
    NUMERIC_COLS,
    SEED,
    DataConfig,
)

# ── Reproducibility ────────────────────────────────────────────────────────
random.seed(SEED)
np.random.seed(SEED)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Custom transformers
# ─────────────────────────────────────────────────────────────────────────────

class MultiHotEncoder(BaseEstimator, TransformerMixin):
    """Encode semicolon-delimited multi-value survey columns as binary arrays.

    Args:
        sep: Delimiter used to split values within a cell.
        top_k: Keep only the *top_k* most frequent values across the corpus.
            Rare / unseen values map to an ``__other__`` bucket.
        handle_unknown: How to treat values not seen during ``fit``.
            ``"ignore"`` silently drops them (output column remains 0).

    Attributes:
        vocab_: Dict mapping column name → list of retained token strings.
        feature_names_out_: Flat list of output feature names.

    Example::

        enc = MultiHotEncoder(top_k=20)
        X_train_enc = enc.fit_transform(df[MULTI_HOT_COLS])
    """

    def __init__(self, sep: str = ";", top_k: int = 30,
                 handle_unknown: str = "ignore") -> None:
        self.sep = sep
        self.top_k = top_k
        self.handle_unknown = handle_unknown

    # ------------------------------------------------------------------
    def _tokenise(self, series: pd.Series) -> list[list[str]]:
        return [
            [tok.strip() for tok in str(v).split(self.sep) if tok.strip()]
            if pd.notna(v) else []
            for v in series
        ]

    def fit(self, X: pd.DataFrame, y=None) -> "MultiHotEncoder":
        """Learn vocabulary from training data.

        Args:
            X: DataFrame whose columns will be multi-hot encoded.
            y: Ignored (present for sklearn API compatibility).

        Returns:
            self
        """
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=[f"col_{i}" for i in range(X.shape[1])])

        self.vocab_: dict[str, list[str]] = {}
        self.feature_names_out_: list[str] = []

        for col in X.columns:
            tokens_per_row = self._tokenise(X[col])
            freq: dict[str, int] = {}
            for row_tokens in tokens_per_row:
                for tok in row_tokens:
                    freq[tok] = freq.get(tok, 0) + 1

            top_tokens = sorted(freq, key=lambda t: -freq[t])[: self.top_k]
            self.vocab_[col] = top_tokens
            self.feature_names_out_ += [f"{col}__{tok}" for tok in top_tokens]

        return self

    def transform(self, X: pd.DataFrame, y=None) -> np.ndarray:
        """Apply multi-hot encoding.

        Args:
            X: DataFrame with the same columns as seen during ``fit``.
            y: Ignored.

        Returns:
            Dense binary NumPy array of shape (n_samples, total_vocab_size).
        """
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=list(self.vocab_.keys()))

        rows: list[np.ndarray] = []
        for col in X.columns:
            vocab = self.vocab_.get(col, [])
            tok_idx = {tok: i for i, tok in enumerate(vocab)}
            col_arr = np.zeros((len(X), len(vocab)), dtype=np.float32)
            for row_idx, row_tokens in enumerate(self._tokenise(X[col])):
                for tok in row_tokens:
                    if tok in tok_idx:
                        col_arr[row_idx, tok_idx[tok]] = 1.0
            rows.append(col_arr)

        return np.hstack(rows) if rows else np.empty((len(X), 0), dtype=np.float32)

    def get_feature_names_out(self, input_features=None) -> list[str]:
        return self.feature_names_out_


class BinaryNormaliser(BaseEstimator, TransformerMixin):
    """Map heterogeneous binary survey responses to 0 / 1 / NaN.

    Handles: integer 0/1, "Yes"/"No", NaN.

    Args:
        fill_value: Value to impute missing entries after mapping (default 0).
    """

    POSITIVE = {"1", "yes", "true", "1.0"}
    NEGATIVE = {"0", "no", "false", "0.0", "nan"}

    def __init__(self, fill_value: float = 0.0) -> None:
        self.fill_value = fill_value

    def fit(self, X, y=None) -> "BinaryNormaliser":
        self.is_fitted_ = True  # sentinel for sklearn check_is_fitted
        return self

    def transform(self, X, y=None) -> np.ndarray:
        """Convert to float32 binary array.

        Args:
            X: Array-like of shape (n_samples, n_features).
            y: Ignored.

        Returns:
            Float32 array of shape (n_samples, n_features) with values in {0, 1}.
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        out = np.full(X.shape, self.fill_value, dtype=np.float32)
        for j in range(X.shape[1]):
            for i, val in enumerate(X[:, j]):
                s = str(val).strip().lower()
                if s in self.POSITIVE:
                    out[i, j] = 1.0
                elif s in self.NEGATIVE:
                    out[i, j] = 0.0
                else:
                    out[i, j] = self.fill_value
        return out


# ─────────────────────────────────────────────────────────────────────────────
# Target construction
# ─────────────────────────────────────────────────────────────────────────────

def build_target(
    df: pd.DataFrame,
    labels: list[str] = AI_TOOL_LABELS,
    min_freq: int = 10,
) -> tuple[np.ndarray, list[str]]:
    """Build a multi-label binary target matrix from survey AI tool columns.

    The target is the *union* of ``AIModelsHaveWorkedWith`` and
    ``AIModelsWantToWorkWith`` for each respondent, binarised over the
    canonical ``AI_TOOL_LABELS`` vocabulary.

    Args:
        df: Raw survey DataFrame.
        labels: Canonical list of AI tool names to use as label columns.
        min_freq: Drop labels whose positive count is below this threshold
            (reduces class imbalance noise on very rare tools).

    Returns:
        Tuple of:
        - ``y``: Binary float32 array of shape (n_samples, n_labels_retained).
        - ``retained_labels``: Ordered list of label strings matching ``y`` columns.

    Raises:
        ValueError: If neither ``AIModelsHaveWorkedWith`` nor
            ``AIModelsWantToWorkWith`` is found in ``df``.
    """
    have_col = "AIModelsHaveWorkedWith"
    want_col = "AIModelsWantToWorkWith"

    if have_col not in df.columns and want_col not in df.columns:
        raise ValueError(
            "DataFrame must contain at least one of "
            f"'{have_col}' or '{want_col}'."
        )

    def parse(series: pd.Series) -> list[set[str]]:
        result: list[set[str]] = []
        for val in series:
            if pd.isna(val):
                result.append(set())
            else:
                result.append(
                    {tok.strip() for tok in str(val).split(";") if tok.strip()}
                )
        return result

    have_sets = parse(df.get(have_col, pd.Series([""] * len(df))))
    want_sets = parse(df.get(want_col, pd.Series([""] * len(df))))

    union_sets: list[list[str]] = [
        sorted(h | w) for h, w in zip(have_sets, want_sets)
    ]

    mlb = MultiLabelBinarizer(classes=labels)
    y_full: np.ndarray = mlb.fit_transform(union_sets).astype(np.float32)

    # Drop labels below minimum frequency threshold
    col_sums = y_full.sum(axis=0)
    mask = col_sums >= min_freq
    retained_labels: list[str] = [lbl for lbl, keep in zip(labels, mask) if keep]
    y = y_full[:, mask]

    dropped = [lbl for lbl, keep in zip(labels, mask) if not keep]
    if dropped:
        logger.info("Dropped %d rare labels (freq < %d): %s", len(dropped), min_freq, dropped)

    logger.info(
        "Target shape: %s | labels retained: %d / %d",
        y.shape, y.shape[1], len(labels),
    )
    return y, retained_labels


# ─────────────────────────────────────────────────────────────────────────────
# Feature column selection helpers
# ─────────────────────────────────────────────────────────────────────────────

def _select_columns(df: pd.DataFrame, candidates: list[str]) -> list[str]:
    """Return only columns present in *df* from *candidates*."""
    return [c for c in candidates if c in df.columns]


def _cap_cardinality(
    df: pd.DataFrame,
    cols: list[str],
    max_cardinality: int,
) -> list[str]:
    """Remove columns whose unique-value count exceeds *max_cardinality*."""
    kept = []
    for col in cols:
        n_unique = df[col].nunique(dropna=True)
        if n_unique <= max_cardinality:
            kept.append(col)
        else:
            logger.debug("Dropping high-cardinality column '%s' (%d unique)", col, n_unique)
    return kept


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline builder
# ─────────────────────────────────────────────────────────────────────────────

def build_preprocessing_pipeline(
    df: pd.DataFrame,
    cfg: DataConfig,
) -> tuple[Pipeline, dict[str, list[str]]]:
    """Construct a fitted-ready sklearn Pipeline for the survey feature matrix.

    The pipeline consists of three parallel sub-pipelines merged by a
    ``ColumnTransformer``:

    1. **Numeric** – median imputation → RobustScaler
    2. **Categorical** – mode imputation → OneHotEncoder (sparse, top-N values)
    3. **Multi-hot** – custom :class:`MultiHotEncoder`
    4. **Binary** – custom :class:`BinaryNormaliser`

    Args:
        df: Raw training DataFrame (used *only* to select valid columns;
            fitting happens later inside a cross-validation loop).
        cfg: :class:`DataConfig` controlling cardinality caps and top-K values.

    Returns:
        Tuple of:
        - Unfitted ``sklearn.pipeline.Pipeline`` ready for ``.fit(X)`` / ``.transform(X)``.
        - ``feature_groups``: Dict mapping group name → list of selected column names.
    """
    # ── Resolve available columns ──────────────────────────────────────────
    num_cols = _select_columns(df, NUMERIC_COLS)
    cat_cols = _cap_cardinality(
        df,
        _select_columns(df, CATEGORICAL_COLS),
        cfg.max_cat_cardinality,
    )
    mhot_cols = _select_columns(df, MULTI_HOT_COLS)
    bin_cols = _select_columns(df, BINARY_COLS)

    feature_groups: dict[str, list[str]] = {
        "numeric": num_cols,
        "categorical": cat_cols,
        "multi_hot": mhot_cols,
        "binary": bin_cols,
    }

    logger.info(
        "Feature groups — numeric: %d, categorical: %d, "
        "multi_hot: %d, binary: %d",
        len(num_cols), len(cat_cols), len(mhot_cols), len(bin_cols),
    )

    # ── Sub-pipelines ──────────────────────────────────────────────────────
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", RobustScaler()),
    ])

    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        (
            "ohe",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=True,
                max_categories=cfg.max_cat_cardinality,
            ),
        ),
    ])

    multi_hot_pipe = Pipeline([
        ("mhot", MultiHotEncoder(top_k=cfg.multi_hot_top_k)),
    ])

    binary_pipe = Pipeline([
        ("binarise", BinaryNormaliser(fill_value=0.0)),
    ])

    # ── Column transformer ─────────────────────────────────────────────────
    transformers = []
    if num_cols:
        transformers.append(("numeric", numeric_pipe, num_cols))
    if cat_cols:
        transformers.append(("categorical", categorical_pipe, cat_cols))
    if mhot_cols:
        transformers.append(("multi_hot", multi_hot_pipe, mhot_cols))
    if bin_cols:
        transformers.append(("binary", binary_pipe, bin_cols))

    col_transformer = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        n_jobs=-1,
    )

    full_pipeline = Pipeline([
        ("preprocessor", col_transformer),
    ])

    return full_pipeline, feature_groups


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_raw_data(path: Path) -> pd.DataFrame:
    """Load raw survey CSV from *path*.

    Args:
        path: File path to the survey CSV.

    Returns:
        DataFrame with all 172 original columns.

    Raises:
        FileNotFoundError: If *path* does not exist.
        pd.errors.ParserError: If the CSV cannot be parsed.
    """
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at: {path}")

    df = pd.read_csv(path, low_memory=False)
    logger.info("Loaded %d rows × %d columns from '%s'", *df.shape, path)
    return df


def split_features_targets(
    df: pd.DataFrame,
    cfg: DataConfig,
) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    """Split *df* into feature matrix *X* and multi-label target *y*.

    Args:
        df: Raw survey DataFrame.
        cfg: :class:`DataConfig` specifying label frequency threshold.

    Returns:
        Tuple of (X, y, label_names) where:
        - ``X`` is the feature DataFrame (all columns except target sources).
        - ``y`` is the binary float32 label matrix (n_samples, n_labels).
        - ``label_names`` is the ordered list of retained AI tool names.
    """
    TARGET_SOURCE_COLS = [
        "AIModelsHaveWorkedWith",
        "AIModelsWantToWorkWith",
        "AIModelsAdmired",
        "AIModelsHaveEntry",
        "AIModelsWantEntry",
        "ResponseId",
    ]

    y, label_names = build_target(df, min_freq=cfg.min_label_freq)

    drop_cols = [c for c in TARGET_SOURCE_COLS if c in df.columns]
    X = df.drop(columns=drop_cols, errors="ignore")

    return X, y, label_names
