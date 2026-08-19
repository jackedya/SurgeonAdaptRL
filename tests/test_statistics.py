from __future__ import annotations

import numpy as np
import pytest

from surgeonadaptrl.statistics import (
    StatisticalError,
    bonferroni,
    bonferroni_all,
    bootstrap_interval,
    exact_binomial_two_sided,
    familywise_decisions,
    mcnemar_test,
    mean_standard_deviation,
    paired_bootstrap_interval,
    paired_t_test,
    rank_methods,
    relative_improvement,
    seed_aggregate,
    standardized_mean_difference,
    summarize_paired,
)


def test_bonferroni_correction() -> None:
    assert bonferroni(0.01, 4) == 0.04
    assert bonferroni(0.8, 4) == 1.0
    assert np.allclose(bonferroni_all((0.01, 0.2, 0.9)), np.array((0.03, 0.6, 1.0)))


def test_paired_t_test_detects_large_consistent_change() -> None:
    first = (4.0, 4.2, 3.9, 4.1, 4.3)
    second = (2.9, 3.0, 2.8, 2.9, 3.1)
    result = paired_t_test(first, second, comparisons=2)
    assert result.sample_size == 5
    assert result.significant
    assert result.adjusted_p_value >= result.p_value


def test_exact_binomial_symmetry() -> None:
    assert np.isclose(exact_binomial_two_sided(1, 10), exact_binomial_two_sided(9, 10))
    assert exact_binomial_two_sided(5, 10) == 1.0


def test_mcnemar_uses_discordant_pairs() -> None:
    first = (True,) * 10 + (False,) * 2
    second = (False,) * 10 + (True,) * 2
    result = mcnemar_test(first, second)
    assert result.sample_size == 12
    assert result.statistic > 0
    assert 0 <= result.p_value <= 1


def test_bootstrap_is_seed_deterministic() -> None:
    first = bootstrap_interval((1.0, 2.0, 3.0, 4.0), resamples=500, seed=7)
    second = bootstrap_interval((1.0, 2.0, 3.0, 4.0), resamples=500, seed=7)
    assert first == second
    assert first.lower <= first.estimate <= first.upper


def test_paired_bootstrap_uses_aligned_differences() -> None:
    interval = paired_bootstrap_interval((2, 3, 4, 5), (1, 2, 3, 4), resamples=100)
    assert interval.estimate == 1.0
    assert interval.lower == 1.0
    assert interval.upper == 1.0


def test_paired_summary_contains_test_and_interval() -> None:
    summary = summarize_paired((3.0, 3.2, 3.1, 2.9, 3.0), (3.8, 3.9, 4.0, 3.7, 3.8), resamples=100)
    assert summary.first_mean < summary.second_mean
    assert summary.difference.upper < 0
    assert summary.test.significant


def test_effect_size_and_aggregates() -> None:
    mean, deviation = mean_standard_deviation((1.0, 2.0, 3.0))
    assert mean == 2.0
    assert deviation == 1.0
    assert standardized_mean_difference((2.0, 3.0, 5.0), (1.0, 1.0, 1.0)) > 0
    records = ({"nte": 0.03}, {"nte": 0.04}, {"nte": 0.05})
    assert seed_aggregate(records, "nte") == pytest.approx((0.04, 0.01))


def test_relative_improvement_definitions() -> None:
    assert relative_improvement(2.89, 3.31, True) == pytest.approx(12.6888, rel=1e-4)
    assert relative_improvement(81.4, 74.6, False) == pytest.approx(9.1153, rel=1e-4)


def test_ranking_and_familywise_decisions() -> None:
    values = {"a": 3.0, "b": 1.0, "c": 2.0}
    assert rank_methods(values, lower_is_better=True) == ["b", "c", "a"]
    decisions = familywise_decisions({"one": 0.001, "two": 0.04})
    assert decisions == {"one": True, "two": False}


@pytest.mark.parametrize(
    "operation",
    [
        lambda: bonferroni(-0.1, 1),
        lambda: paired_t_test((1.0,), (1.0,)),
        lambda: relative_improvement(1.0, 0.0, True),
        lambda: rank_methods({}, True),
    ],
)
def test_invalid_statistical_inputs(operation: object) -> None:
    with pytest.raises(StatisticalError):
        operation()
