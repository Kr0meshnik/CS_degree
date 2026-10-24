"""
Unit tests for the AI Developer Tools Prediction pipeline.

Tests cover:
- Target construction shape and binary correctness
- MultiHotEncoder output shape and feature name consistency
- BinaryNormaliser value range
- Full preprocessing pipeline output dimensionality
- Model output tensor/array shapes (key requirement from spec)
- End-to-end predict → shape validation

Run with:
    python -m pytest tests_pipeline.py -v
"""

from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

import sys
sys.path.insert(0, str(Path(__file__).parent))

from config import AI_TOOL_LABELS, SEED, DataConfig
from preprocessing import (
    BinaryNormaliser,
    MultiHotEncoder,
    build_preprocessing_pipeline,
    build_target,
    split_features_targets,
)
from train import build_model, compute_multilabel_metrics


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures / factories
# ─────────────────────────────────────────────────────────────────────────────

def _make_survey_df(n: int = 200, seed: int = SEED) -> pd.DataFrame:
    """Generate a minimal synthetic survey DataFrame for testing."""
    rng = np.random.default_rng(seed)

    tools = AI_TOOL_LABELS

    def _random_tool_str(rng: np.random.Generator) -> str:
        k = rng.integers(0, 4)
        chosen = rng.choice(tools, size=k, replace=False).tolist()
        return ";".join(chosen) if chosen else np.nan

    df = pd.DataFrame({
        "ResponseId": np.arange(n),
        "MainBranch": rng.choice(["Developer", "Student", "Other"], n),
        "Age": rng.choice(["18-24", "25-34", "35-44", "45-54", "55+"], n),
        "EdLevel": rng.choice(["Bachelor", "Master", "PhD", "Other"], n),
        "Employment": rng.choice(["Full-time", "Part-time", "Freelance"], n),
        "RemoteWork": rng.choice(["Remote", "Hybrid", "In-person"], n),
        "YearsCode": rng.integers(0, 30, n).astype(float),
        "WorkExp": rng.integers(0, 20, n).astype(float),
        "ToolCountWork": rng.integers(0, 20, n).astype(float),
        "ToolCountPersonal": rng.integers(0, 10, n).astype(float),
        "AISelect": rng.choice(
            ["Yes, I use AI tools daily", "No, and I don't plan to",
             "Yes, I use AI tools weekly", "No, but I plan to soon"], n),
        "AISent": rng.choice(["Favorable", "Very favorable", "Unfavorable", "Indifferent"], n),
        "AIThreat": rng.choice(["Yes", "No", "I'm not sure", None], n),
        "TechEndorse_1": rng.integers(1, 6, n).astype(float),
        "TechEndorse_2": rng.integers(1, 6, n).astype(float),
        "AIModelsChoice": rng.choice(["Yes", "No", None], n),
        "AIToolCurrently mostly AI": rng.integers(0, 2, n).astype(float),
        "AIToolPlan to mostly use AI": rng.integers(0, 2, n).astype(float),
        "AIModelsHaveWorkedWith": [_random_tool_str(rng) for _ in range(n)],
        "AIModelsWantToWorkWith": [_random_tool_str(rng) for _ in range(n)],
        "LanguageHaveWorkedWith": [
            ";".join(rng.choice(["Python", "JavaScript", "Rust", "Go", "C++"],
                                 size=rng.integers(1, 4), replace=False).tolist())
            for _ in range(n)
        ],
        "DevEnvsHaveWorkedWith": [
            ";".join(rng.choice(["VS Code", "PyCharm", "Vim", "Emacs"],
                                 size=rng.integers(1, 3), replace=False).tolist())
            for _ in range(n)
        ],
    })
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Test classes
# ─────────────────────────────────────────────────────────────────────────────

class TestBuildTarget(unittest.TestCase):
    """Tests for multi-label target construction."""

    def setUp(self) -> None:
        self.df = _make_survey_df(n=300)

    def test_output_is_2d_binary_float32(self) -> None:
        """Target matrix must be 2-D, binary-valued, and float32."""
        y, labels = build_target(self.df, min_freq=1)

        self.assertEqual(y.ndim, 2, "y must be 2-D")
        self.assertEqual(y.dtype, np.float32, "y must be float32")
        unique_vals = np.unique(y)
        for v in unique_vals:
            self.assertIn(v, {0.0, 1.0}, f"Unexpected value {v} in target")

    def test_n_samples_matches_input(self) -> None:
        """Number of rows in y must equal number of rows in df."""
        y, _ = build_target(self.df, min_freq=1)
        self.assertEqual(y.shape[0], len(self.df))

    def test_n_labels_leq_vocabulary(self) -> None:
        """Retained labels must be a subset of AI_TOOL_LABELS."""
        y, labels = build_target(self.df, min_freq=1)
        self.assertLessEqual(len(labels), len(AI_TOOL_LABELS))

    def test_min_freq_filter_removes_rare_labels(self) -> None:
        """Higher min_freq should produce fewer or equal retained labels."""
        _, labels_low = build_target(self.df, min_freq=1)
        _, labels_high = build_target(self.df, min_freq=100)
        self.assertGreaterEqual(len(labels_low), len(labels_high))

    def test_raises_when_target_columns_missing(self) -> None:
        """ValueError raised if neither AIModels column is present."""
        df_bad = self.df.drop(
            columns=["AIModelsHaveWorkedWith", "AIModelsWantToWorkWith"],
            errors="ignore",
        )
        with self.assertRaises(ValueError):
            build_target(df_bad, min_freq=1)


class TestMultiHotEncoder(unittest.TestCase):
    """Tests for the custom MultiHotEncoder transformer."""

    def setUp(self) -> None:
        self.data = pd.DataFrame({
            "langs": ["Python;JavaScript", "Rust;Go", "Python;Go", np.nan, "Python"],
            "envs": ["VS Code;Vim", "PyCharm", "VS Code", "Emacs", np.nan],
        })

    def test_output_shape(self) -> None:
        """Output rows must match input rows; columns == sum of per-col vocab sizes."""
        enc = MultiHotEncoder(top_k=3)
        out = enc.fit_transform(self.data)

        n_rows, n_cols = out.shape
        self.assertEqual(n_rows, len(self.data))
        expected_cols = sum(len(v) for v in enc.vocab_.values())
        self.assertEqual(n_cols, expected_cols)

    def test_binary_output_values(self) -> None:
        """All output values must be 0 or 1."""
        enc = MultiHotEncoder(top_k=5)
        out = enc.fit_transform(self.data)
        self.assertTrue(np.all((out == 0) | (out == 1)))

    def test_feature_names_length_matches_columns(self) -> None:
        """feature_names_out_ length must equal number of output columns."""
        enc = MultiHotEncoder(top_k=3)
        out = enc.fit_transform(self.data)
        self.assertEqual(len(enc.get_feature_names_out()), out.shape[1])

    def test_nan_rows_produce_all_zeros(self) -> None:
        """A fully-NaN row must map to an all-zero multi-hot vector."""
        df_nan = pd.DataFrame({
            "langs": [np.nan],
            "envs": [np.nan],
        })
        enc = MultiHotEncoder(top_k=3)
        enc.fit(self.data)
        out = enc.transform(df_nan)
        self.assertTrue(np.all(out == 0))

    def test_unknown_tokens_ignored(self) -> None:
        """Tokens not seen during fit must not raise and must produce 0 columns."""
        enc = MultiHotEncoder(top_k=3, handle_unknown="ignore")
        enc.fit(self.data)
        df_new = pd.DataFrame({
            "langs": ["COBOL;APL"],
            "envs": ["Notepad"],
        })
        out = enc.transform(df_new)
        self.assertTrue(np.all(out == 0))


class TestBinaryNormaliser(unittest.TestCase):
    """Tests for BinaryNormaliser transformer."""

    def test_maps_yes_no_correctly(self) -> None:
        data = np.array([["Yes", "No"], ["1", "0"], [None, "yes"]])
        bn = BinaryNormaliser(fill_value=0.0)
        out = bn.fit_transform(data)

        self.assertEqual(out.dtype, np.float32)
        self.assertEqual(out[0, 0], 1.0)   # Yes → 1
        self.assertEqual(out[0, 1], 0.0)   # No  → 0
        self.assertEqual(out[1, 0], 1.0)   # "1" → 1
        self.assertEqual(out[2, 1], 1.0)   # "yes" → 1

    def test_output_shape_preserved(self) -> None:
        data = np.ones((50, 10), dtype=object)
        out = BinaryNormaliser().fit_transform(data)
        self.assertEqual(out.shape, (50, 10))

    def test_output_dtype_is_float32(self) -> None:
        data = pd.DataFrame({"a": ["Yes", "No"], "b": [1, 0]})
        out = BinaryNormaliser().fit_transform(data)
        self.assertEqual(out.dtype, np.float32)


class TestPreprocessingPipeline(unittest.TestCase):
    """Tests for the full sklearn preprocessing pipeline."""

    def setUp(self) -> None:
        self.df = _make_survey_df(n=200)
        self.cfg = DataConfig(min_label_freq=1)
        X, y, label_names = split_features_targets(self.df, self.cfg)
        self.X = X
        self.y = y
        self.label_names = label_names

    def test_pipeline_output_is_2d(self) -> None:
        """Preprocessed feature matrix must be 2-D."""
        pipe, _ = build_preprocessing_pipeline(self.X, self.cfg)
        X_enc = pipe.fit_transform(self.X)
        if hasattr(X_enc, "toarray"):
            X_enc = X_enc.toarray()
        self.assertEqual(X_enc.ndim, 2)

    def test_pipeline_output_rows_match_input(self) -> None:
        """Row count of transformed matrix must equal input row count."""
        pipe, _ = build_preprocessing_pipeline(self.X, self.cfg)
        X_enc = pipe.fit_transform(self.X)
        if hasattr(X_enc, "toarray"):
            X_enc = X_enc.toarray()
        self.assertEqual(X_enc.shape[0], len(self.X))

    def test_pipeline_no_nans_in_numeric_output(self) -> None:
        """After imputation, numeric columns must contain no NaN values."""
        pipe, _ = build_preprocessing_pipeline(self.X, self.cfg)
        X_enc = pipe.fit_transform(self.X)
        if hasattr(X_enc, "toarray"):
            X_enc = X_enc.toarray()
        self.assertFalse(np.any(np.isnan(X_enc)), "NaN found in encoded features")

    def test_transform_same_columns_as_fit(self) -> None:
        """Transform on unseen split must produce same feature width as fit."""
        split = len(self.X) // 2
        X_tr, X_val = self.X.iloc[:split], self.X.iloc[split:]

        pipe, _ = build_preprocessing_pipeline(X_tr, self.cfg)
        enc_tr = pipe.fit_transform(X_tr)
        enc_val = pipe.transform(X_val)

        if hasattr(enc_tr, "toarray"):
            enc_tr = enc_tr.toarray()
            enc_val = enc_val.toarray()

        self.assertEqual(
            enc_tr.shape[1], enc_val.shape[1],
            "Train and val encoded widths differ",
        )


class TestModelOutputShapes(unittest.TestCase):
    """Tests that model output tensor/array shapes are correct.

    This is the **key shape-matching test** required by the spec.
    """

    def setUp(self) -> None:
        """Build a tiny synthetic dataset and fit a quick model."""
        rng = np.random.default_rng(SEED)
        self.n_samples = 120
        self.n_features = 50
        self.n_labels = len(AI_TOOL_LABELS)

        self.X_enc = rng.random((self.n_samples, self.n_features)).astype(np.float32)
        # Ensure each label has at least a few positives for XGB to train
        self.y = (rng.random((self.n_samples, self.n_labels)) > 0.6).astype(np.float32)

        from config import ModelConfig
        self.model = build_model(ModelConfig(n_estimators=10, max_depth=3))
        self.model.fit(self.X_enc, self.y)

    # ── predict ────────────────────────────────────────────────────────────

    def test_predict_shape(self) -> None:
        """clf.predict output must be (n_samples, n_labels)."""
        y_pred = self.model.predict(self.X_enc)
        self.assertEqual(
            y_pred.shape,
            (self.n_samples, self.n_labels),
            f"predict shape mismatch: got {y_pred.shape}, "
            f"expected {(self.n_samples, self.n_labels)}",
        )

    def test_predict_dtype_is_numeric(self) -> None:
        """predict output must be int or float (not object)."""
        y_pred = self.model.predict(self.X_enc)
        self.assertTrue(
            np.issubdtype(y_pred.dtype, np.number),
            f"predict dtype is not numeric: {y_pred.dtype}",
        )

    def test_predict_values_binary(self) -> None:
        """predict output must only contain 0 and 1."""
        y_pred = self.model.predict(self.X_enc)
        unique = np.unique(y_pred)
        for v in unique:
            self.assertIn(v, {0, 1, 0.0, 1.0}, f"Unexpected value {v} in predictions")

    # ── predict_proba ──────────────────────────────────────────────────────

    def test_predict_proba_shape(self) -> None:
        """Stacked probability matrix must be (n_samples, n_labels)."""
        y_prob = np.column_stack(
            [est.predict_proba(self.X_enc)[:, 1] for est in self.model.estimators_]
        )
        self.assertEqual(
            y_prob.shape,
            (self.n_samples, self.n_labels),
            f"predict_proba shape mismatch: got {y_prob.shape}, "
            f"expected {(self.n_samples, self.n_labels)}",
        )

    def test_predict_proba_in_unit_interval(self) -> None:
        """All predicted probabilities must be in [0, 1]."""
        y_prob = np.column_stack(
            [est.predict_proba(self.X_enc)[:, 1] for est in self.model.estimators_]
        )
        self.assertTrue(np.all(y_prob >= 0.0) and np.all(y_prob <= 1.0))

    def test_predict_proba_n_estimators_matches_n_labels(self) -> None:
        """Number of estimators must equal number of labels (one per label)."""
        self.assertEqual(
            len(self.model.estimators_),
            self.n_labels,
            "Mismatch between number of estimators and number of labels",
        )

    # ── single-sample inference ────────────────────────────────────────────

    def test_single_sample_predict_shape(self) -> None:
        """Single-sample predict must return shape (1, n_labels)."""
        single = self.X_enc[[0], :]
        y_pred_single = self.model.predict(single)
        self.assertEqual(y_pred_single.shape, (1, self.n_labels))

    def test_single_sample_proba_shape(self) -> None:
        """Single-sample probability must return shape (1, n_labels)."""
        single = self.X_enc[[0], :]
        y_prob_single = np.column_stack(
            [est.predict_proba(single)[:, 1] for est in self.model.estimators_]
        )
        self.assertEqual(y_prob_single.shape, (1, self.n_labels))

    # ── batch size invariance ──────────────────────────────────────────────

    def test_batch_size_invariance(self) -> None:
        """Output row count must always match input row count for any batch size."""
        for batch_size in [1, 10, 50, self.n_samples]:
            X_batch = self.X_enc[:batch_size]
            y_pred = self.model.predict(X_batch)
            self.assertEqual(
                y_pred.shape[0], batch_size,
                f"Row mismatch for batch_size={batch_size}: got {y_pred.shape[0]}",
            )


class TestComputeMetrics(unittest.TestCase):
    """Tests for the metric computation utility."""

    def setUp(self) -> None:
        rng = np.random.default_rng(SEED)
        self.n, self.n_labels = 80, 5
        self.y_true = (rng.random((self.n, self.n_labels)) > 0.5).astype(float)
        self.y_pred = (rng.random((self.n, self.n_labels)) > 0.5).astype(float)
        self.y_prob = rng.random((self.n, self.n_labels)).astype(float)
        self.labels = [f"tool_{i}" for i in range(self.n_labels)]

    def test_metric_keys_present(self) -> None:
        """Core aggregated keys must appear in the returned dict."""
        metrics = compute_multilabel_metrics(
            self.y_true, self.y_pred, self.y_prob, self.labels
        )
        for key in ("macro_f1", "micro_f1", "sample_f1"):
            self.assertIn(key, metrics, f"Key '{key}' missing from metrics")

    def test_per_label_f1_count(self) -> None:
        """Number of per-label F1 entries must match n_labels."""
        metrics = compute_multilabel_metrics(
            self.y_true, self.y_pred, self.y_prob, self.labels
        )
        per_label = {k: v for k, v in metrics.items() if k.startswith("f1__")}
        self.assertEqual(len(per_label), self.n_labels)

    def test_f1_values_in_unit_interval(self) -> None:
        """All F1 scores must be in [0, 1]."""
        metrics = compute_multilabel_metrics(
            self.y_true, self.y_pred, self.y_prob, self.labels
        )
        for k, v in metrics.items():
            if "f1" in k:
                self.assertGreaterEqual(v, 0.0, f"{k} < 0")
                self.assertLessEqual(v, 1.0, f"{k} > 1")

    def test_prefix_applied(self) -> None:
        """All keys must be prefixed when prefix is provided."""
        metrics = compute_multilabel_metrics(
            self.y_true, self.y_pred, self.y_prob, self.labels, prefix="val_"
        )
        for k in metrics:
            self.assertTrue(k.startswith("val_"), f"Key '{k}' missing 'val_' prefix")


# ─────────────────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    unittest.main(verbosity=2)
