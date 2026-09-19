"""Tests for the cascade-forward "wisdom of the crowd" ANN port, on synthetic data only."""

import numpy as np

from ei_model.models.crowd import (
    N_EXPECTED_ARCHITECTURES,
    CrowdANN,
    _layer_shapes,
    _param_count,
    enumerate_architectures,
)


def test_enumerate_architectures_count_matches_matlab_default():
    arches = enumerate_architectures()
    assert len(arches) == N_EXPECTED_ARCHITECTURES == 189


def test_enumerate_architectures_respects_bounds_and_layer_counts():
    arches = enumerate_architectures()
    layer_counts = {}
    for arch in arches:
        assert 1 <= len(arch) <= 3
        for neuron_count in arch:
            assert neuron_count >= 1
        layer_counts[len(arch)] = layer_counts.get(len(arch), 0) + 1

    # 15 + 49 + 125 = 189, per populateArchitectures.m's odometer with cap
    # floor(15/layers) per digit (15, 7, 5 respectively).
    assert layer_counts == {1: 15, 2: 49, 3: 125}
    assert len(arches) == len(set(arches))  # no duplicates


def test_enumerate_architectures_single_layer_is_1_through_15():
    arches = enumerate_architectures()
    single_layer = sorted(a[0] for a in arches if len(a) == 1)
    assert single_layer == list(range(1, 16))


def test_cascade_forward_layer_shapes_include_raw_input_and_priors():
    # architecture (3, 2): layer 1 sees only the 5 raw features; layer 2
    # (cascade) sees the 5 raw features + layer 1's 3 outputs; the output
    # layer sees the 5 raw features + all 5 hidden outputs (3 + 2).
    shapes = _layer_shapes(n_features=5, hidden_sizes=(3, 2))
    assert shapes == [(3, 5), (2, 8), (1, 10)]
    assert _param_count(shapes) == (3 * 5 + 3) + (2 * 8 + 2) + (1 * 10 + 1)


def _make_noisy_sine(n=200, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-3, 3, size=(n, 1))
    y = np.sin(X[:, 0]) + rng.normal(scale=0.3, size=n)
    return X, y


def test_crowd_beats_a_single_net_on_noisy_sine():
    X, y = _make_noisy_sine(n=200, seed=1)
    X_train, y_train = X[:150], y[:150]
    X_test, y_test = X[150:], y[150:]

    small_archs = [(4,), (3, 2), (2, 2, 1)]
    crowd = CrowdANN(
        n_replications=4,
        architectures=small_archs,
        max_epochs=60,
        random_state=7,
        n_jobs=1,
    ).fit(X_train, y_train)

    crowd_pred = crowd.predict(X_test)
    all_preds = crowd.predict_all(X_test)
    single_net_mse = np.mean((all_preds[0] - y_test) ** 2)
    crowd_mse = np.mean((crowd_pred - y_test) ** 2)

    # The crowd average should not be worse than a single arbitrary member net.
    assert crowd_mse <= single_net_mse * 1.5
    assert np.isfinite(crowd_pred).all()


def test_crowd_is_deterministic_with_a_seed():
    X, y = _make_noisy_sine(n=80, seed=2)
    small_archs = [(3,), (2, 1)]

    kwargs = dict(
        n_replications=2, architectures=small_archs, max_epochs=20, random_state=123, n_jobs=1
    )
    pred_a = CrowdANN(**kwargs).fit(X, y).predict(X)
    pred_b = CrowdANN(**kwargs).fit(X, y).predict(X)

    # Same seed -> same architectures, splits, and init, so predictions should
    # match closely. Multi-threaded BLAS reduction order can perturb the last
    # few float bits per LM/TRF iteration, and nonlinear optimization can
    # amplify that over iterations, so this checks near-equality (same basin
    # of convergence), not bit-identity.
    np.testing.assert_allclose(pred_a, pred_b, rtol=2e-2, atol=1e-2)


def test_fit_predict_shapes_and_finite_on_synthetic_table(synthetic_X_y):
    X, y = synthetic_X_y
    X_filled = X.fillna(X.median())

    crowd = CrowdANN(
        n_replications=1,
        architectures=[(2,), (1, 1)],
        max_epochs=10,
        random_state=5,
        n_jobs=1,
    ).fit(X_filled, y)

    preds = crowd.predict(X_filled)
    assert preds.shape == (len(y),)
    assert np.isfinite(preds).all()
    assert len(crowd.nets_) == 2  # 2 architectures x 1 replication


def test_default_architectures_assert_189_when_defaults_used():
    crowd = CrowdANN(n_replications=1, architectures=None, max_epochs=1, n_jobs=1)
    arches = crowd._get_architectures()
    assert len(arches) == 189
