"""Shared pytest fixtures backed by the synthetic (non-real) modeling table."""

import pytest

from ei_model.data import split_features_target
from tests.fixtures.synthetic import generate_synthetic_modeling_table


@pytest.fixture
def synthetic_table():
    """A ~100-row synthetic modeling table with the same column-family shape as the real data."""
    return generate_synthetic_modeling_table(n_rows=100, seed=42)


@pytest.fixture
def synthetic_X_y(synthetic_table):
    """Features and target split from the synthetic modeling table."""
    return split_features_target(synthetic_table)
