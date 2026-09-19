import pytest

from ei_model.experiments.config import load_config


def test_load_config_minimal(tmp_path):
    path = tmp_path / "cfg.yaml"
    path.write_text(
        "name: my_run\nprotocol: P1\nvariants: [after]\nmodels: [mean, ridge]\n",
        encoding="utf-8",
    )
    config = load_config(path)

    assert config.name == "my_run"
    assert config.protocol == "P1"
    assert config.variants == ["after"]
    assert config.models == ["mean", "ridge"]
    assert config.cv.n_splits == 5
    assert config.cv.n_repeats == 5
    assert config.ann_cv is None


def test_load_config_with_cv_and_ann_cv_overrides(tmp_path):
    path = tmp_path / "cfg.yaml"
    path.write_text(
        "name: my_run\n"
        "protocol: P2\n"
        "cv: {n_splits: 5, n_repeats: 5}\n"
        "ann_cv: {n_splits: 5, n_repeats: 1}\n",
        encoding="utf-8",
    )
    config = load_config(path)

    assert config.cv.n_repeats == 5
    assert config.ann_cv.n_repeats == 1
    assert config.effective_cv("tree").n_repeats == 5
    assert config.effective_cv("ann").n_repeats == 1


def test_load_config_rejects_unknown_protocol(tmp_path):
    path = tmp_path / "cfg.yaml"
    path.write_text("name: bad\nprotocol: P99\n", encoding="utf-8")

    with pytest.raises(ValueError):
        load_config(path)
