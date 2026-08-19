"""Statistical tests and interval estimates used in Section 3.9."""

from __future__ import annotations

from dataclasses import dataclass
from math import comb
from typing import Callable, Sequence

import numpy as np
from scipy import stats


class StatisticalError(ValueError):
    """Statistical inputs do not satisfy the paired evaluation protocol."""


@dataclass(frozen=True)
class TestResult:
    statistic: float
    p_value: float
    adjusted_p_value: float
    significant: bool
    sample_size: int


@dataclass(frozen=True)
class Interval:
    estimate: float
    lower: float
    upper: float
    confidence: float


@dataclass(frozen=True)
class PairedSummary:
    first_mean: float
    second_mean: float
    difference: Interval
    test: TestResult


def _one_dimensional(values: Sequence[float] | np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0:
        raise StatisticalError(f"{name} must be a nonempty one-dimensional sequence")
    if not np.isfinite(array).all():
        raise StatisticalError(f"{name} must contain finite values")
    return array


def bonferroni(p_value: float, comparisons: int) -> float:
    if not 0 <= p_value <= 1 or comparisons <= 0:
        raise StatisticalError("invalid probability or comparison count")
    return min(1.0, p_value * comparisons)


def bonferroni_all(p_values: Sequence[float]) -> np.ndarray:
    values = _one_dimensional(p_values, "p_values")
    if ((values < 0) | (values > 1)).any():
        raise StatisticalError("probabilities must lie in [0, 1]")
    return np.minimum(1.0, values * values.size)


def paired_t_test(
    first: Sequence[float],
    second: Sequence[float],
    comparisons: int = 1,
    alpha: float = 0.05,
) -> TestResult:
    left = _one_dimensional(first, "first")
    right = _one_dimensional(second, "second")
    if left.shape != right.shape or left.size < 2:
        raise StatisticalError("paired samples must have equal length of at least two")
    result = stats.ttest_rel(left, right)
    adjusted = bonferroni(float(result.pvalue), comparisons)
    return TestResult(float(result.statistic), float(result.pvalue), adjusted, adjusted < alpha, left.size)


def exact_binomial_two_sided(successes: int, trials: int, probability: float = 0.5) -> float:
    if trials < 0 or not 0 <= successes <= trials or not 0 <= probability <= 1:
        raise StatisticalError("invalid binomial parameters")
    observed = comb(trials, successes) * probability**successes * (1 - probability) ** (trials - successes)
    total = 0.0
    for value in range(trials + 1):
        mass = comb(trials, value) * probability**value * (1 - probability) ** (trials - value)
        if mass <= observed + 1e-15:
            total += mass
    return min(1.0, total)


def mcnemar_test(
    first_correct: Sequence[bool],
    second_correct: Sequence[bool],
    comparisons: int = 1,
    alpha: float = 0.05,
) -> TestResult:
    first = np.asarray(first_correct, dtype=bool)
    second = np.asarray(second_correct, dtype=bool)
    if first.ndim != 1 or first.shape != second.shape or first.size == 0:
        raise StatisticalError("paired categorical outcomes must be aligned and nonempty")
    first_only = int(np.sum(first & ~second))
    second_only = int(np.sum(~first & second))
    discordant = first_only + second_only
    p_value = 1.0 if discordant == 0 else exact_binomial_two_sided(min(first_only, second_only), discordant)
    statistic = 0.0 if discordant == 0 else (abs(first_only - second_only) - 1) ** 2 / discordant
    adjusted = bonferroni(p_value, comparisons)
    return TestResult(float(statistic), p_value, adjusted, adjusted < alpha, first.size)


def percentile_interval(values: Sequence[float], confidence: float = 0.95) -> Interval:
    array = _one_dimensional(values, "values")
    if not 0 < confidence < 1:
        raise StatisticalError("confidence must lie strictly between zero and one")
    tail = (1 - confidence) / 2
    return Interval(float(array.mean()), float(np.quantile(array, tail)), float(np.quantile(array, 1 - tail)), confidence)


def bootstrap_interval(
    values: Sequence[float],
    statistic: Callable[[np.ndarray], float] = np.mean,
    confidence: float = 0.95,
    resamples: int = 10000,
    seed: int = 13,
) -> Interval:
    array = _one_dimensional(values, "values")
    if resamples <= 0 or not 0 < confidence < 1:
        raise StatisticalError("invalid bootstrap parameters")
    generator = np.random.default_rng(seed)
    estimates = np.empty(resamples, dtype=np.float64)
    for index in range(resamples):
        sample = generator.choice(array, size=array.size, replace=True)
        estimates[index] = statistic(sample)
    tail = (1 - confidence) / 2
    return Interval(float(statistic(array)), float(np.quantile(estimates, tail)), float(np.quantile(estimates, 1 - tail)), confidence)


def paired_bootstrap_interval(
    first: Sequence[float],
    second: Sequence[float],
    confidence: float = 0.95,
    resamples: int = 10000,
    seed: int = 13,
) -> Interval:
    left = _one_dimensional(first, "first")
    right = _one_dimensional(second, "second")
    if left.shape != right.shape:
        raise StatisticalError("paired samples must have equal length")
    return bootstrap_interval(left - right, np.mean, confidence, resamples, seed)


def summarize_paired(
    first: Sequence[float],
    second: Sequence[float],
    comparisons: int = 1,
    confidence: float = 0.95,
    resamples: int = 10000,
    seed: int = 13,
) -> PairedSummary:
    left = _one_dimensional(first, "first")
    right = _one_dimensional(second, "second")
    difference = paired_bootstrap_interval(left, right, confidence, resamples, seed)
    test = paired_t_test(left, right, comparisons, 1 - confidence)
    return PairedSummary(float(left.mean()), float(right.mean()), difference, test)


def mean_standard_deviation(values: Sequence[float]) -> tuple[float, float]:
    array = _one_dimensional(values, "values")
    if array.size < 2:
        raise StatisticalError("standard deviation requires at least two values")
    return float(array.mean()), float(array.std(ddof=1))


def standardized_mean_difference(first: Sequence[float], second: Sequence[float]) -> float:
    left = _one_dimensional(first, "first")
    right = _one_dimensional(second, "second")
    if left.shape != right.shape or left.size < 2:
        raise StatisticalError("paired effect size requires aligned samples")
    differences = left - right
    deviation = differences.std(ddof=1)
    return 0.0 if deviation == 0 else float(differences.mean() / deviation)


def relative_improvement(candidate: float, reference: float, lower_is_better: bool) -> float:
    if reference == 0:
        raise StatisticalError("reference value cannot be zero")
    change = reference - candidate if lower_is_better else candidate - reference
    return 100.0 * change / abs(reference)


def seed_aggregate(records: Sequence[dict[str, float]], metric: str) -> tuple[float, float]:
    if not records or any(metric not in record for record in records):
        raise StatisticalError("each seed record must contain the requested metric")
    return mean_standard_deviation([record[metric] for record in records])


def rank_methods(values: dict[str, float], lower_is_better: bool) -> list[str]:
    if not values or not all(np.isfinite(value) for value in values.values()):
        raise StatisticalError("method values must be finite and nonempty")
    return [name for name, _ in sorted(values.items(), key=lambda item: item[1], reverse=not lower_is_better)]


def familywise_decisions(p_values: dict[str, float], alpha: float = 0.05) -> dict[str, bool]:
    if not p_values:
        raise StatisticalError("at least one test is required")
    adjusted = bonferroni_all(list(p_values.values()))
    return {name: bool(value < alpha) for name, value in zip(p_values, adjusted)}
