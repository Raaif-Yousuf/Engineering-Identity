"""Faithful port of the owner's original Keras ANN for engineering identity.

Ported from `Engineering Identitiy/First.py` in the owner's original
UTD-Research project (an identical copy also exists as `First.py` under
`EI - ANN/`). The random-seed pattern used here (`tf.keras.utils.set_random_seed`
plus deterministic ops) comes from the owner's later `newANN.py` in the same
project, which formalized the seeding that `First.py` only partially applied
(via `random_state=42` on the train/validation split).

The network architecture (512 -> 256 -> 128 -> 1, with batch normalization
and dropout after each hidden layer), the Adam optimizer with a 0.001
learning rate, the MSE loss with an RMSE metric, the EarlyStopping /
ReduceLROnPlateau / ModelCheckpoint callbacks, and the 90/10 holdout split
with random_state=42 are unchanged from the original. Only the data-loading
path is adapted: the original script read a fixed Excel file path directly,
while this port takes an already-loaded feature matrix and target vector
(see `ei_model.data.load_modeling_table` / `split_features_target`) so it can
be rerun against the current dataset -- or a synthetic fixture in tests -- as
the "before" baseline for later models.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import Callback, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import BatchNormalization, Dense, Dropout
from tensorflow.keras.models import Sequential

# Original hyperparameters, unchanged from First.py.
RANDOM_STATE = 42
TEST_SIZE = 0.10
LEARNING_RATE = 0.001
EPOCHS = 1000
BATCH_SIZE = 32
EARLY_STOPPING_PATIENCE = 50
REDUCE_LR_FACTOR = 0.5
REDUCE_LR_PATIENCE = 20


def set_random_seed(seed: int = RANDOM_STATE) -> None:
    """Seed Python/NumPy/TensorFlow and enable deterministic ops, as in newANN.py."""
    tf.keras.utils.set_random_seed(seed)
    tf.config.experimental.enable_op_determinism()


def prepare_data(
    X: pd.DataFrame,
    y: pd.Series,
    random_state: int = RANDOM_STATE,
    test_size: float = TEST_SIZE,
):
    """Mean-impute, standardize, and 90/10 split -- as in the original load_and_prepare_data."""
    columns = X.columns
    imputer = SimpleImputer(strategy="mean")
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=columns)

    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X_imputed), columns=columns)

    return train_test_split(X_scaled, y, test_size=test_size, random_state=random_state)


def build_model(input_dim: int, learning_rate: float = LEARNING_RATE) -> tf.keras.Model:
    """Same 512-256-128 architecture, dropout, and Adam/MSE compile as the original build_model."""
    model = Sequential()
    model.add(Dense(512, activation="relu", input_dim=input_dim))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))

    model.add(Dense(256, activation="relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))

    model.add(Dense(128, activation="relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))

    model.add(Dense(1))  # Regression output
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
        metrics=[tf.keras.metrics.RootMeanSquaredError()],
    )
    return model


def build_callbacks(checkpoint_path: str | None = None) -> list[Callback]:
    """Same callbacks as the original: EarlyStopping, ReduceLROnPlateau, ModelCheckpoint."""
    callbacks: list[Callback] = [
        EarlyStopping(
            monitor="val_loss",
            patience=EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=REDUCE_LR_FACTOR,
            patience=REDUCE_LR_PATIENCE,
        ),
    ]
    if checkpoint_path:
        callbacks.append(ModelCheckpoint(checkpoint_path, save_best_only=True))
    return callbacks


@dataclass
class OriginalAnnResult:
    """Outcome of a single original_ann train/holdout run."""

    model: tf.keras.Model
    history: tf.keras.callbacks.History
    r2: float
    mse: float
    y_val: pd.Series
    y_pred: np.ndarray


def train_and_evaluate(
    X: pd.DataFrame,
    y: pd.Series,
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    checkpoint_path: str | None = None,
    random_state: int = RANDOM_STATE,
    test_size: float = TEST_SIZE,
    verbose: int = 0,
) -> OriginalAnnResult:
    """Reproduce the original train_and_evaluate: 90/10 holdout split, fit with callbacks, R2/MSE.

    `epochs` and `batch_size` default to the original values (1000, 32) but
    are exposed as parameters so tests can run a tiny number of epochs.
    """
    set_random_seed(random_state)

    X_train, X_val, y_train, y_val = prepare_data(
        X, y, random_state=random_state, test_size=test_size
    )

    model = build_model(X_train.shape[1])
    callbacks = build_callbacks(checkpoint_path)

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=verbose,
    )

    y_pred = model.predict(X_val, verbose=0).flatten()
    r2 = r2_score(y_val, y_pred)
    mse = mean_squared_error(y_val, y_pred)

    return OriginalAnnResult(
        model=model, history=history, r2=r2, mse=mse, y_val=y_val, y_pred=y_pred
    )
