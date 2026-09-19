"""Tests for the ported original Keras ANN, run with a tiny epoch count on the synthetic fixture."""

import numpy as np

from ei_model.models import original_ann


def test_build_model_matches_original_architecture():
    model = original_ann.build_model(input_dim=10)

    dense_units = [layer.units for layer in model.layers if layer.__class__.__name__ == "Dense"]
    assert dense_units == [512, 256, 128, 1]
    assert model.optimizer.__class__.__name__ == "Adam"
    assert model.loss == "mse"


def test_prepare_data_imputes_scales_and_splits(synthetic_X_y):
    X, y = synthetic_X_y
    assert X.isna().sum().sum() > 0  # sanity: the fixture actually has missing values

    X_train, X_val, y_train, y_val = original_ann.prepare_data(X, y)

    assert not X_train.isna().any().any()
    assert not X_val.isna().any().any()
    n = len(X)
    expected_val = round(n * original_ann.TEST_SIZE)
    assert len(X_val) == expected_val
    assert len(X_train) == n - expected_val
    assert len(y_train) == len(X_train)
    assert len(y_val) == len(X_val)


def test_prepare_data_is_deterministic(synthetic_X_y):
    X, y = synthetic_X_y
    split_a = original_ann.prepare_data(X, y)
    split_b = original_ann.prepare_data(X, y)

    for a, b in zip(split_a, split_b, strict=True):
        pd_equal = a.equals(b) if hasattr(a, "equals") else (a == b).all()
        assert pd_equal


def test_train_and_evaluate_runs_end_to_end_with_tiny_epochs(synthetic_X_y):
    X, y = synthetic_X_y

    result = original_ann.train_and_evaluate(X, y, epochs=2, batch_size=16, verbose=0)

    assert result.model is not None
    assert len(result.y_pred) == len(result.y_val)
    assert np.isfinite(result.r2)
    assert np.isfinite(result.mse)
    assert result.mse >= 0
