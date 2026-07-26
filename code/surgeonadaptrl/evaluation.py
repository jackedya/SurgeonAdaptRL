from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import binomtest, ttest_rel

from .style import bootstrap_mean


@dataclass(frozen=True)
class ContinuousComparison:
    mean: float
    standard_deviation: float
    confidence_lower: float
    confidence_upper: float
    p_value: float
    corrected_p_value: float


@dataclass(frozen=True)
class CategoricalComparison:
    discordant_reference: int
    discordant_candidate: int
    p_value: float
    corrected_p_value: float


def paired_continuous(
    candidate: np.ndarray,
    reference: np.ndarray,
    comparisons: int,
    seed: int = 13,
) -> ContinuousComparison:
    mean, lower, upper = bootstrap_mean(candidate, seed=seed)
    result = ttest_rel(candidate, reference)
    p_value = float(result.pvalue)
    return ContinuousComparison(
        mean,
        float(candidate.std(ddof=1)),
        lower,
        upper,
        p_value,
        min(p_value * comparisons, 1.0),
    )


def mcnemar(
    candidate_correct: np.ndarray,
    reference_correct: np.ndarray,
    comparisons: int,
) -> CategoricalComparison:
    reference_only = int(np.logical_and(reference_correct, np.logical_not(candidate_correct)).sum())
    candidate_only = int(np.logical_and(candidate_correct, np.logical_not(reference_correct)).sum())
    discordant = reference_only + candidate_only
    p_value = float(binomtest(min(reference_only, candidate_only), discordant, 0.5).pvalue) if discordant else 1.0
    return CategoricalComparison(
        reference_only,
        candidate_only,
        p_value,
        min(p_value * comparisons, 1.0),
    )


def phase_wise(values: np.ndarray, phases: np.ndarray, phase_count: int = 10) -> dict[int, tuple[float, float]]:
    output: dict[int, tuple[float, float]] = {}
    for phase in range(phase_count):
        selected = values[phases == phase]
        if len(selected):
            output[phase] = (float(selected.mean()), float(selected.std(ddof=1)) if len(selected) > 1 else 0.0)
    return output


def summarize_seeds(values: dict[int, dict[str, float]]) -> dict[str, tuple[float, float]]:
    metrics = sorted({metric for result in values.values() for metric in result})
    return {
        metric: (
            float(np.mean([result[metric] for result in values.values()])),
            float(np.std([result[metric] for result in values.values()], ddof=1)),
        )
        for metric in metrics
    }
