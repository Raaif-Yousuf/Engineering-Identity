"""A smaller, regularized ANN, offered alongside the original as a comparison point.

Where `ei_model.models.original_ann` faithfully reproduces the owner's
512-256-128 Keras network unchanged, this module asks: does a much smaller
network with modern regularization do any better on a dataset this size
(hundreds to ~1900 rows, dozens to ~240 columns)? Differences from the
original:

- Much smaller: 64-32-1 instead of 512-256-128 (the original network has
  roughly 100x more parameters than training rows).
- L2 weight decay on every Dense layer (the original has none).
- Lighter dropout (0.2/0.1 instead of 0.3/0.3/0.2).
- Early stopping only (no ReduceLROnPlateau/ModelCheckpoint -- the smaller
  net converges fast enough that they added little in practice runs).
- Meant to be paired with quantile scaling (see
  `ei_model.experiments.preprocessing.quantile_pipeline`) rather than a
  plain z-score, and evaluated as a small-seed ensemble (fit N times with
  different initializations/shuffles, average the predictions) rather than
  a single fit, to reduce the variance a single-seed net shows on datasets
  this small.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import Callback, EarlyStopping
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.regularizers import l2

LEARNING_RATE = 0.001
EPOCHS = 300
BATCH_SIZE = 32
EARLY_STOPPING_PATIENCE = 20
VALIDATION_FRACTION = 0.15
WEIGHT_DECAY = 1e-4
DEFAULT_N_SEEDS = 3


def build_model(input_dim: int, learning_rate: float = LEARNING_RATE) -> tf.keras.Model:
    """A 64-32-1 MLP with L2 weight decay and dropout, much smaller than the original."""
    model = Sequential()
    model.add(
        Dense(
            64,
            activation="relu",
            input_dim=input_dim,
            kernel_regularizer=l2(WEIGHT_DECAY),
        )
    )
    model.add(Dropout(0.2))

    model.add(Dense(32, activation="relu", kernel_regularizer=l2(WEIGHT_DECAY)))
    model.add(Dropout(0.1))

    model.add(Dense(1, kernel_regularizer=l2(WEIGHT_DECAY)))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
        metrics=[tf.keras.metrics.RootMeanSquaredError()],
    )
    return model


def build_callbacks() -> list[Callback]:
    return [
        EarlyStopping(
            monitor="val_loss",
            patience=EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
        )
    ]


def set_random_seed(seed: int) -> None:
    tf.keras.utils.set_random_seed(seed)


def fit_one_seed(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    seed: int,
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    validation_fraction: float = VALIDATION_FRACTION,
    verbose: int = 0,
) -> np.ndarray:
    """Fit one seed of the improved ANN and return its predictions for X_test.

    `X_train`/`X_test` are expected to already be imputed and quantile-scaled
    (see `ei_model.experiments.preprocessing.quantile_pipeline`); this
    function only carves an internal validation split out of X_train for
    early stopping, mirroring how `original_ann.train_and_evaluate` uses its
    holdout split, but nested one level deeper (inside a CV training fold).
    """
    set_random_seed(seed)
    X_fit, X_val, y_fit, y_val = train_test_split(
        X_train, y_train, test_size=validation_fraction, random_state=seed
    )
    model = build_model(X_fit.shape[1])
    model.fit(
        X_fit,
        y_fit,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=build_callbacks(),
        verbose=verbose,
    )
    return model.predict(X_test, verbose=0).flatten()


@dataclass
class EnsembleResult:
    predictions: np.ndarray
    seed_predictions: list[np.ndarray]


def fit_predict_ensemble(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    n_seeds: int = DEFAULT_N_SEEDS,
    base_seed: int = 0,
    **fit_kwargs,
) -> EnsembleResult:
    """Fit `n_seeds` independent improved-ANN models and average their predictions."""
    seed_preds = [
        fit_one_seed(X_train, y_train, X_test, seed=base_seed + i, **fit_kwargs)
        for i in range(n_seeds)
    ]
    return EnsembleResult(predictions=np.mean(seed_preds, axis=0), seed_predictions=seed_preds)
