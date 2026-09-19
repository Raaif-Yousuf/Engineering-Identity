from ei_model.evaluation import FoldScore
from ei_model.experiments.stats import paired_r2_comparison


def _scores(r2s):
    return [
        FoldScore(repeat=i // 5, fold=i % 5, r2=r2, rmse=1.0, mae=1.0, n_test=10)
        for i, r2 in enumerate(r2s)
    ]


def test_challenger_always_better_gives_fraction_one():
    baseline = _scores([0.1, 0.1, 0.1, 0.1, 0.1])
    challenger = _scores([0.3, 0.3, 0.3, 0.3, 0.3])

    result = paired_r2_comparison(baseline, challenger, "baseline", "challenger")

    assert result.fraction_improved == 1.0
    assert result.n_matched_folds == 5


def test_challenger_always_worse_gives_fraction_zero():
    baseline = _scores([0.3, 0.3, 0.3, 0.3, 0.3])
    challenger = _scores([0.1, 0.1, 0.1, 0.1, 0.1])

    result = paired_r2_comparison(baseline, challenger, "baseline", "challenger")

    assert result.fraction_improved == 0.0


def test_matches_only_overlapping_repeat_fold_keys():
    baseline = _scores([0.1] * 25)  # 5 repeats x 5 folds
    challenger = _scores([0.2] * 5)  # only repeat 0

    result = paired_r2_comparison(baseline, challenger, "baseline", "challenger")

    assert result.n_matched_folds == 5
    assert result.fraction_improved == 1.0


def test_no_overlap_returns_nan_fraction():
    baseline = [FoldScore(repeat=0, fold=0, r2=0.1, rmse=1.0, mae=1.0, n_test=10)]
    challenger = [FoldScore(repeat=1, fold=0, r2=0.2, rmse=1.0, mae=1.0, n_test=10)]

    result = paired_r2_comparison(baseline, challenger, "baseline", "challenger")

    assert result.n_matched_folds == 0
    assert result.wilcoxon_p is None
