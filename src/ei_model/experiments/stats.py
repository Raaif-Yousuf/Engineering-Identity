"""Paired fold-by-fold comparisons against a baseline model (default: the original ANN)."""

from dataclasses import dataclass

from scipy.stats import wilcoxon

from ei_model.evaluation import FoldScore


@dataclass
class PairedComparison:
    baseline_key: str
    challenger_key: str
    n_matched_folds: int
    fraction_improved: float
    wilcoxon_statistic: float | None
    wilcoxon_p: float | None
    note: str = ""


def _by_repeat_fold(scores: list[FoldScore]) -> dict[tuple[int, int], FoldScore]:
    return {(s.repeat, s.fold): s for s in scores}


def paired_r2_comparison(
    baseline_scores: list[FoldScore],
    challenger_scores: list[FoldScore],
    baseline_key: str = "original_ann",
    challenger_key: str = "challenger",
) -> PairedComparison:
    """Fraction of matched folds where `challenger` beats `baseline` on R2, plus a Wilcoxon test.

    Folds are matched by (repeat, fold) index, not position, so this is safe
    even when one model was scored on fewer repeats than the other (e.g. the
    ANN family run at a reduced repeat count -- see docs/experiments.md);
    only the overlapping folds are compared.
    """
    base_by_key = _by_repeat_fold(baseline_scores)
    chal_by_key = _by_repeat_fold(challenger_scores)
    matched_keys = sorted(set(base_by_key) & set(chal_by_key))

    if not matched_keys:
        return PairedComparison(
            baseline_key, challenger_key, 0, float("nan"), None, None, note="no overlapping folds"
        )

    base_r2 = [base_by_key[k].r2 for k in matched_keys]
    chal_r2 = [chal_by_key[k].r2 for k in matched_keys]
    n_improved = sum(c > b for c, b in zip(chal_r2, base_r2, strict=True))
    fraction_improved = n_improved / len(matched_keys)

    if len(matched_keys) < 2 or all(c == b for c, b in zip(chal_r2, base_r2, strict=True)):
        return PairedComparison(
            baseline_key,
            challenger_key,
            len(matched_keys),
            fraction_improved,
            None,
            None,
            note="not enough variation for a Wilcoxon test",
        )

    try:
        stat, p = wilcoxon(chal_r2, base_r2)
    except ValueError as exc:  # all-zero differences etc.
        return PairedComparison(
            baseline_key,
            challenger_key,
            len(matched_keys),
            fraction_improved,
            None,
            None,
            note=str(exc),
        )

    return PairedComparison(
        baseline_key, challenger_key, len(matched_keys), fraction_improved, float(stat), float(p)
    )
