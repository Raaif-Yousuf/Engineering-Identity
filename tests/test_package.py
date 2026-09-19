"""Smoke test: the package imports cleanly."""

import ei_model


def test_version_is_set():
    assert isinstance(ei_model.__version__, str)
    assert ei_model.__version__
