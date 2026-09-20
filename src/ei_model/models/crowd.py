"""Faithful Python port of the lab's MATLAB "wisdom of the crowd" ANN baseline.

Ported from `populateArchitectures.m`, `trainArchPop_v5_batch.m`, and
`analyzeANN_v5_batch.m` (extracted from the lab's MATLAB toolbox archive,
`UGR SP25/MATLAB FILE PAVAN.zip`; driver:
`ANN_Prediction_v2.m`/`v3.m`). The original pipeline: enumerate 189
cascade-forward architectures (<=3 hidden layers, <=15 neurons total),
train `newcf` nets with `trainlm` (Levenberg-Marquardt) 100 times per
architecture (18,900 nets total), and average all networks' predictions
("wisdom of the crowd"). Per `EI_inventory.md` section 1, the only
target-matched MATLAB crowd output found in the data dump scored R^2 of
about 0 on a curricular-only (indicator) feature set; the full-feature
("89 + x metric") crowd result was never recorded/verified.

This module reproduces the architecture enumeration exactly (see
`enumerate_architectures`, a direct translation of
`populateArchitectures.m`'s odometer loop, asserted to yield 189
architectures with default settings) and the cascade-forward topology,
`tansig`/`purelin` activations, `mapminmax` [-1, 1] scaling, and the
70/15/15 train/validation/test divide with validation-based early
stopping (`max_fail=6`, `epochs<=1000`) described in the lab's baseline.

Deviations from the MATLAB implementation (documented per project rules):

1. **Optimizer**: MATLAB's `trainlm` performs one Levenberg-Marquardt
   step per "epoch" with an adaptively tuned damping factor `mu` that
   persists across epochs. This port instead calls
   `scipy.optimize.least_squares(method="lm")` once per epoch, warm-started
   from the previous epoch's weights, with a bounded `max_nfev` budget
   standing in for "one step" (a few LM/TRF iterations, not literally
   one). `mu`-style damping-factor persistence across epoch boundaries is
   not reproduced; each call re-derives its own internal damping schedule
   from scratch. The Jacobian itself, however, *is* computed analytically
   via backprop (`_analytic_jacobian`, passed as `least_squares`'s `jac`
   argument) rather than by scipy's default finite-difference
   approximation -- matching how MATLAB's `trainlm` computes it
   internally, and avoiding a finite-difference Jacobian's O(n_params)
   function evaluations per iteration (cascade nets have `n_params`
   dominated by the raw feature count, since every layer re-consumes the
   raw input, so this is not just a performance nicety but the difference
   between minutes and hours for real feature counts).
2. **`lm` vs `trf` fallback**: MINPACK's `lm` method (used by
   `scipy.optimize.least_squares(method="lm")`) requires at least as many
   residuals as free parameters (`n_samples_train >= n_params`). Cascade
   nets concatenate the raw input to every layer, so parameter counts
   scale with the *feature* count, not just neuron count -- with the
   wide feature matrices this project actually uses (e.g. 140 features),
   `n_params` routinely exceeds `n_samples_train` even for the smallest
   architectures. MATLAB's `trainlm` has no such restriction. When
   `n_samples_train < n_params` this port automatically falls back to
   `method="trf"` (trust-region reflective), which solves the same
   nonlinear least-squares problem without MINPACK's shape restriction.
   This is a necessary, documented substitution, not a faithful
   reproduction of MATLAB's solver in that regime.
3. **Weight initialization**: MATLAB's `newcf` initializes each layer
   with Nguyen-Widrow initialization. This port uses small-scale uniform
   random initialization (`U(-0.5, 0.5)` scaled by fan-in) instead. Exact
   Nguyen-Widrow initialization is not reproduced.
4. **`dividerand` RNG**: both implementations randomly assign 70/15/15
   train/validation/test membership per net, but MATLAB's RNG stream and
   this port's `numpy.random.Generator` stream are different algorithms,
   so the exact sample assignment (and therefore exact trained weights)
   cannot match bit-for-bit even with "the same" seed. The *mechanism*
   (uniform random assignment, validation-only early stopping, held-out
   test partition unused during training) is matched.
5. **`mapminmax` on constant columns**: MATLAB's `mapminmax` assumes
   `removeconstantrows` preprocessing has already dropped zero-range
   columns; if given one it divides by zero. This port instead maps a
   zero-range column's values to the bottom of the output range
   (`-1`) rather than raising or emitting `NaN`/`Inf`.
6. **"Epoch" granularity and budget**: because of (1), an "epoch" here is
   one bounded `least_squares` call (`max_nfev=30`, cheap because the
   Jacobian is analytic -- see (1)), not one literal gradient/Jacobian
   step. A call that converges on its own (uses fewer evaluations than its
   budget, i.e. hits `ftol`/`xtol`/`gtol` before exhausting `max_nfev`)
   ends training immediately, since a warm-started re-call from the same
   point cannot improve further. `max_fail` and `epochs` still count in
   these units and default to MATLAB's `trainlm` defaults (`max_fail=6`,
   `epochs=1000`), but they are not numerically equivalent units, and in
   practice most nets stop in well under 1000 epochs.
7. **No `showWindow`/MATLAB-GUI-only behavior** is applicable (this is a
   headless port).

**Performance note (measured, not estimated):** on a synthetic 900-row x
140-feature matrix, 189 architectures x 1 replication (189 nets total,
`n_jobs=-1` on a 16-logical-core Windows dev machine) took 341 s (~5.7
minutes) wall-clock, with nets stopping after 2.2 epochs on average. The
task-specified full benchmark (189 architectures x 5 replications = 945
nets, on a 1,800-row x 140-feature matrix) was attempted first and did
not finish within a practical wall-clock budget (over 90 minutes, still
running, on the same machine): even with the analytic Jacobian in (1),
`scipy.optimize.least_squares(method="trf")`'s own internal linear
algebra (solving the trust-region subproblem each iteration) costs
`O(n_params^2 * n_train_samples)`-ish per call, and cascade nets at 140
features have `n_params` in the ~300-2,350 range across the 189
architectures regardless of how neurons are split across layers (every
layer re-consumes the raw 140-column input). This is a genuine
scalability limitation of using MINPACK/TRF-style dense solvers for
cascade nets at realistic EI feature counts, not just a slow benchmark
machine -- it should inform whether/how this module is run against the
real ~1,800 x 144 dataset (e.g. fewer replications, a feature-reduced
input, or a future analytic-Hessian/streaming solver). Reduce
`n_replications` (and, for quick checks, pass a short `architectures`
list) to control wall-clock cost; both are constructor parameters.

None of the above are expected to reproduce the MATLAB baseline's exact
numeric predictions -- they reproduce its *architecture, topology, and
training protocol* closely enough to test whether the "wisdom of the
crowd" design itself has merit in Python, independent of MATLAB.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from joblib import Parallel, delayed
from scipy.optimize import least_squares
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

# --- populateArchitectures.m defaults ---------------------------------
MAX_NEURONS = 15
MAX_LAYERS = 3
N_EXPECTED_ARCHITECTURES = 189

# --- trainArchPop_v5_batch.m / MATLAB trainlm defaults -----------------
DEFAULT_REPLICATIONS = 100
DEFAULT_MAX_EPOCHS = 1000
DEFAULT_MAX_FAIL = 6
DEFAULT_TRAIN_FRACTION = 0.70
DEFAULT_VAL_FRACTION = 0.15
DEFAULT_TEST_FRACTION = 0.15
FEATURE_RANGE = (-1.0, 1.0)


def enumerate_architectures(
    max_neurons: int = MAX_NEURONS, max_layers: int = MAX_LAYERS
) -> list[tuple[int, ...]]:
    """Direct translation of `populateArchitectures.m`'s odometer loop.

    For `layers` in `1..max_layers`, walks a `layers`-length counter
    `Si` (each digit starting at 1) forward, appending every value for
    which all digits satisfy `Si <= max_neurons / layers` (MATLAB's
    `while Si <= (max_nuerons/layers)` short-circuits on the *first*
    digit that overflows -- it never wraps the leading digit). With the
    defaults (15 neurons, <=3 layers) this yields exactly 189
    architectures (15 + 49 + 125), asserted below.
    """
    architectures: list[tuple[int, ...]] = []
    for layers in range(1, max_layers + 1):
        cap = max_neurons / layers  # float division, as in the .m file
        si = [1] * layers
        while all(s <= cap for s in si):
            architectures.append(tuple(si))
            si[-1] += 1
            # for layer = layers:-1:2 -> carry from the last digit down to
            # (but not including) the first; MATLAB 1-indexes from the
            # left, so 0-indexed this is range(layers-1, 0, -1).
            for idx in range(layers - 1, 0, -1):
                if si[idx] > cap:
                    si[idx] = 1
                    si[idx - 1] += 1
    return architectures


class MapMinMax:
    """`mapminmax`-equivalent affine scaler to a fixed output range (default [-1, 1])."""

    def __init__(self, feature_range: tuple[float, float] = FEATURE_RANGE):
        self.feature_range = feature_range

    def fit(self, X: np.ndarray) -> MapMinMax:
        X = np.atleast_2d(X)
        self.data_min_ = X.min(axis=0)
        self.data_max_ = X.max(axis=0)
        self.data_range_ = self.data_max_ - self.data_min_
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(X)
        lo, hi = self.feature_range
        safe_range = np.where(self.data_range_ == 0, 1.0, self.data_range_)
        scaled = (X - self.data_min_) / safe_range * (hi - lo) + lo
        # Deviation (4) in the module docstring: constant columns map to lo.
        return np.where(self.data_range_ == 0, lo, scaled)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        lo, hi = self.feature_range
        return (X - lo) / (hi - lo) * self.data_range_ + self.data_min_


def _layer_shapes(n_features: int, hidden_sizes: tuple[int, ...]) -> list[tuple[int, int]]:
    """(out_dim, in_dim) for each hidden layer, then the output layer, cascade-forward style.

    Every layer (hidden and output) receives the raw input concatenated
    with every earlier layer's output, matching `newcf`.
    """
    shapes = []
    running_in = n_features
    for h in hidden_sizes:
        shapes.append((h, running_in))
        running_in += h
    shapes.append((1, running_in))  # output layer (single regression target)
    return shapes


def _param_count(shapes: list[tuple[int, int]]) -> int:
    return sum(out_dim * in_dim + out_dim for out_dim, in_dim in shapes)


def _init_params(shapes: list[tuple[int, int]], rng: np.random.Generator) -> np.ndarray:
    parts = []
    for out_dim, in_dim in shapes:
        fan_in = max(in_dim, 1)
        scale = 1.0 / np.sqrt(fan_in)
        parts.append(rng.uniform(-scale, scale, size=out_dim * in_dim))
        parts.append(np.zeros(out_dim))
    return np.concatenate(parts)


def _unpack(
    params: np.ndarray, shapes: list[tuple[int, int]]
) -> list[tuple[np.ndarray, np.ndarray]]:
    layers = []
    offset = 0
    for out_dim, in_dim in shapes:
        n_w = out_dim * in_dim
        W = params[offset : offset + n_w].reshape(out_dim, in_dim)
        offset += n_w
        b = params[offset : offset + out_dim]
        offset += out_dim
        layers.append((W, b))
    return layers


def _forward(params: np.ndarray, X: np.ndarray, shapes: list[tuple[int, int]]) -> np.ndarray:
    """Cascade-forward pass: tansig hidden layers, purelin output."""
    layers = _unpack(params, shapes)
    cat = X
    for W, b in layers[:-1]:
        z = cat @ W.T + b
        out = np.tanh(z)  # tansig(n) == tanh(n)
        cat = np.concatenate([cat, out], axis=1)
    W_out, b_out = layers[-1]
    y = cat @ W_out.T + b_out  # purelin: identity
    return y.ravel()


def _forward_cache(params: np.ndarray, X: np.ndarray, shapes: list[tuple[int, int]]):
    """Forward pass that also returns everything the backward pass needs."""
    layers = _unpack(params, shapes)
    cats = [X]  # cats[l] = input to layer l+1 (cat_0 = X)
    outs = []  # outs[l] = tansig output of hidden layer l+1
    cat = X
    for W, b in layers[:-1]:
        z = cat @ W.T + b
        out = np.tanh(z)
        outs.append(out)
        cat = np.concatenate([cat, out], axis=1)
        cats.append(cat)
    W_out, b_out = layers[-1]
    yhat = (cat @ W_out.T + b_out).ravel()
    return layers, cats, outs, yhat


def _analytic_jacobian(
    params: np.ndarray, X: np.ndarray, y: np.ndarray, shapes: list[tuple[int, int]]
) -> np.ndarray:
    """Analytic (backprop) Jacobian of the residual w.r.t. every flat parameter.

    `residual = yhat - y` and `y` is constant w.r.t. `params`, so this is
    also the Jacobian of `yhat`. Computing it analytically (as MATLAB's own
    `trainlm` does internally) avoids the O(n_params) function evaluations
    a finite-difference Jacobian would need per LM/TRF iteration -- with
    cascade nets, `n_params` scales with the raw feature count (every layer
    re-consumes the raw input), so finite differences would dominate the
    runtime for anything but a handful of features.

    Because of the cascade topology, hidden layer l's output feeds every
    later layer (each later layer's weight matrix has a same-offset slice
    corresponding to it, since concatenation only appends columns), so the
    backward pass accumulates contributions from *all* downstream layers,
    not just the immediate next one.
    """
    del y  # unused: yhat's Jacobian does not depend on the target
    layers, cats, outs, _yhat = _forward_cache(params, X, shapes)
    n_samples = X.shape[0]
    n_features = X.shape[1]
    n_hidden = len(layers) - 1
    hidden_sizes = [W.shape[0] for W, _ in layers[:-1]]
    W_out, _b_out = layers[-1]

    offsets = []  # column offset of each hidden layer's output within any later `cat`
    running = n_features
    for h in hidden_sizes:
        offsets.append(running)
        running += h

    deltas: list[np.ndarray] = [None] * n_hidden  # deltas[layer] = d yhat / d z_{layer+1}
    for layer_idx in range(n_hidden - 1, -1, -1):
        h_l = hidden_sizes[layer_idx]
        offset = offsets[layer_idx]
        g = np.broadcast_to(W_out[0, offset : offset + h_l], (n_samples, h_l)).copy()
        for later_idx in range(layer_idx + 1, n_hidden):
            W_later, _ = layers[later_idx]
            g += deltas[later_idx] @ W_later[:, offset : offset + h_l]
        deltas[layer_idx] = g * (1.0 - outs[layer_idx] ** 2)  # d/dz tanh(z) = 1 - tanh(z)^2

    blocks = []
    for layer_idx in range(n_hidden):
        cat_prev = cats[layer_idx]
        delta_l = deltas[layer_idx]
        dW = delta_l[:, :, None] * cat_prev[:, None, :]
        blocks.append(dW.reshape(n_samples, -1))
        blocks.append(delta_l)
    blocks.append(cats[-1])  # d yhat / d W_out (single output neuron)
    blocks.append(np.ones((n_samples, 1)))  # d yhat / d b_out
    return np.concatenate(blocks, axis=1)


def _residuals(params: np.ndarray, X: np.ndarray, y: np.ndarray, shapes) -> np.ndarray:
    return _forward(params, X, shapes) - y


def _mse(params: np.ndarray, X: np.ndarray, y: np.ndarray, shapes) -> float:
    if X.shape[0] == 0:
        return 0.0
    r = _residuals(params, X, y, shapes)
    return float(np.mean(r**2))


def _dividerand(
    n: int,
    rng: np.random.Generator,
    train_fraction: float,
    val_fraction: float,
    test_fraction: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """MATLAB `dividerand`-equivalent: a random 70/15/15 index partition."""
    idx = rng.permutation(n)
    n_train = round(n * train_fraction)
    n_val = round(n * val_fraction)
    train_idx = idx[:n_train]
    val_idx = idx[n_train : n_train + n_val]
    test_idx = idx[n_train + n_val :]
    return train_idx, val_idx, test_idx


@dataclass
class _FittedNet:
    """One trained cascade-forward net: its architecture and flat weight vector."""

    hidden_sizes: tuple[int, ...]
    shapes: list[tuple[int, int]]
    params: np.ndarray
    best_val_mse: float
    epochs_run: int

    def predict_scaled(self, X_scaled: np.ndarray) -> np.ndarray:
        return _forward(self.params, X_scaled, self.shapes)


def _train_single_net(
    X: np.ndarray,
    y: np.ndarray,
    hidden_sizes: tuple[int, ...],
    seed: tuple[int, ...],
    max_epochs: int,
    max_fail: int,
    train_fraction: float,
    val_fraction: float,
    test_fraction: float,
) -> _FittedNet:
    """Train one cascade-forward net via repeated bounded LM calls with validation early stop."""
    rng = np.random.default_rng(seed)
    n_samples, n_features = X.shape
    shapes = _layer_shapes(n_features, hidden_sizes)
    n_params = _param_count(shapes)

    train_idx, val_idx, test_idx = _dividerand(
        n_samples, rng, train_fraction, val_fraction, test_fraction
    )
    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]

    params = _init_params(shapes, rng)
    # Deviation (2): MINPACK's 'lm' needs n_residuals >= n_params.
    method = "lm" if X_train.shape[0] >= n_params else "trf"
    # With the analytic Jacobian (see _analytic_jacobian), one LM/TRF
    # iteration costs one residual + one Jacobian evaluation regardless of
    # n_params, so a small, fixed per-epoch budget is enough (unlike a
    # finite-difference Jacobian, whose cost scales with n_params).
    nfev_per_epoch = 30

    best_params = params.copy()
    best_val_mse = _mse(params, X_val, y_val, shapes)
    fail_count = 0
    _epoch = 0
    for _epoch in range(1, max_epochs + 1):
        result = least_squares(
            _residuals,
            params,
            jac=_analytic_jacobian,
            method=method,
            max_nfev=nfev_per_epoch,
            args=(X_train, y_train, shapes),
        )
        params = result.x
        val_mse = _mse(params, X_val, y_val, shapes)
        if val_mse < best_val_mse - 1e-12:
            best_val_mse = val_mse
            best_params = params.copy()
            fail_count = 0
        else:
            fail_count += 1
        # result.nfev < nfev_per_epoch means LM/TRF itself converged
        # (hit its ftol/xtol/gtol) rather than exhausting the budget --
        # further epochs from this exact point cannot improve the fit.
        converged = result.nfev < nfev_per_epoch
        if fail_count >= max_fail or converged:
            break

    return _FittedNet(
        hidden_sizes=hidden_sizes,
        shapes=shapes,
        params=best_params,
        best_val_mse=best_val_mse,
        epochs_run=_epoch,
    )


class CrowdANN(BaseEstimator, RegressorMixin):
    """Sklearn-style "wisdom of the crowd" cascade-forward ANN ensemble.

    Enumerates the same 189 architectures as `populateArchitectures.m`
    (<=3 hidden layers, <=15 neurons total), trains `replications` nets
    per architecture (100 by default, matching the MATLAB baseline's
    18,900 total nets; pass a smaller value for faster/test runs), and
    predicts the mean across every trained net. See the module docstring
    for the full list of documented deviations from the MATLAB original.

    Parameters
    ----------
    n_replications : number of nets trained per architecture (default 100).
    max_neurons, max_layers : architecture-enumeration bounds (default 15, 3).
    max_epochs, max_fail : early-stopping bounds (default 1000, 6).
    train_fraction, val_fraction, test_fraction : the 70/15/15 divide.
    random_state : base seed; combined with (architecture index, replication
        index) so every net gets an independent, reproducible stream.
    n_jobs : joblib parallelism across (architecture, replication) pairs.
    architectures : override the enumerated architecture list (mainly for
        tests -- pass a short list to skip the full 189).
    """

    def __init__(
        self,
        n_replications: int = DEFAULT_REPLICATIONS,
        max_neurons: int = MAX_NEURONS,
        max_layers: int = MAX_LAYERS,
        max_epochs: int = DEFAULT_MAX_EPOCHS,
        max_fail: int = DEFAULT_MAX_FAIL,
        train_fraction: float = DEFAULT_TRAIN_FRACTION,
        val_fraction: float = DEFAULT_VAL_FRACTION,
        test_fraction: float = DEFAULT_TEST_FRACTION,
        random_state: int = 0,
        n_jobs: int = 1,
        architectures: list[tuple[int, ...]] | None = None,
    ):
        self.n_replications = n_replications
        self.max_neurons = max_neurons
        self.max_layers = max_layers
        self.max_epochs = max_epochs
        self.max_fail = max_fail
        self.train_fraction = train_fraction
        self.val_fraction = val_fraction
        self.test_fraction = test_fraction
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.architectures = architectures

    def _get_architectures(self) -> list[tuple[int, ...]]:
        if self.architectures is not None:
            return list(self.architectures)
        arches = enumerate_architectures(self.max_neurons, self.max_layers)
        if self.max_neurons == MAX_NEURONS and self.max_layers == MAX_LAYERS:
            assert len(arches) == N_EXPECTED_ARCHITECTURES, (
                f"expected {N_EXPECTED_ARCHITECTURES} architectures, got {len(arches)}"
            )
        return arches

    def fit(self, X, y) -> CrowdANN:
        X, y = check_X_y(X, y, y_numeric=True)
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        self.x_scaler_ = MapMinMax(FEATURE_RANGE).fit(X)
        self.y_scaler_ = MapMinMax(FEATURE_RANGE).fit(y.reshape(-1, 1))
        X_scaled = self.x_scaler_.transform(X)
        y_scaled = self.y_scaler_.transform(y.reshape(-1, 1)).ravel()

        architectures = self._get_architectures()
        jobs = [
            (arch_idx, arch, rep)
            for arch_idx, arch in enumerate(architectures)
            for rep in range(self.n_replications)
        ]

        base_seed = 0 if self.random_state is None else int(self.random_state)
        nets = Parallel(n_jobs=self.n_jobs)(
            delayed(_train_single_net)(
                X_scaled,
                y_scaled,
                arch,
                (base_seed, arch_idx, rep),
                self.max_epochs,
                self.max_fail,
                self.train_fraction,
                self.val_fraction,
                self.test_fraction,
            )
            for arch_idx, arch, rep in jobs
        )

        self.nets_ = nets
        self.architectures_ = architectures
        self.n_features_in_ = X.shape[1]
        return self

    def predict(self, X) -> np.ndarray:
        check_is_fitted(self, "nets_")
        X = check_array(X)
        X_scaled = self.x_scaler_.transform(np.asarray(X, dtype=float))
        preds_scaled = np.stack([net.predict_scaled(X_scaled) for net in self.nets_])
        crowd_scaled = preds_scaled.mean(axis=0)
        return self.y_scaler_.inverse_transform(crowd_scaled.reshape(-1, 1)).ravel()

    def predict_all(self, X) -> np.ndarray:
        """Return every individual net's prediction (n_nets, n_samples), unscaled."""
        check_is_fitted(self, "nets_")
        X = check_array(X)
        X_scaled = self.x_scaler_.transform(np.asarray(X, dtype=float))
        preds_scaled = np.stack([net.predict_scaled(X_scaled) for net in self.nets_])
        return np.stack(
            [self.y_scaler_.inverse_transform(p.reshape(-1, 1)).ravel() for p in preds_scaled]
        )
