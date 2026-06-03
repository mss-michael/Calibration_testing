"""Concept calibration test with confidence-matched null samples.

This file is intentionally written as a small standalone module.  The goal is
to make the statistical idea visible in the code, so a student can continue
working on it without having to reverse-engineer too many abstractions.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def _validate_probability_inputs(
    probs: np.ndarray,
    y_true: np.ndarray,
    n_bins: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate inputs needed for calibration-error calculations."""
    probs = np.asarray(probs, dtype=float)
    y_true = np.asarray(y_true)

    if probs.ndim != 2:
        raise ValueError("probs must be a 2D array with shape (n_datapoints, n_classes).")
    if probs.shape[0] == 0:
        raise ValueError("probs must contain at least one datapoint.")
    if probs.shape[1] < 2:
        raise ValueError("probs must contain probabilities for at least two classes.")
    if len(y_true) != probs.shape[0]:
        raise ValueError("y_true must have the same length as probs.")
    if not np.all(np.isfinite(probs)):
        raise ValueError("probs must contain only finite values.")
    if np.any(probs < 0.0) or np.any(probs > 1.0):
        raise ValueError("probs should contain probabilities between 0 and 1.")
    if not np.issubdtype(y_true.dtype, np.integer):
        raise ValueError("y_true must contain integer class labels.")
    if np.any(y_true < 0) or np.any(y_true >= probs.shape[1]):
        raise ValueError("y_true contains class labels outside the valid range.")
    if n_bins < 1:
        raise ValueError("n_bins must be at least 1.")

    return probs, y_true.astype(int, copy=False)


def _validate_inputs(
    concept_indices: np.ndarray,
    probs: np.ndarray,
    y_true: np.ndarray,
    m: int,
    epsilon: float,
    max_concept_sample_size: int,
    n_bins: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Validate inputs for the full concept calibration test."""
    probs, y_true = _validate_probability_inputs(probs, y_true, n_bins)

    concept_indices = np.asarray(concept_indices)
    if concept_indices.ndim != 1:
        raise ValueError("concept_indices must be a 1D sequence of datapoint indices.")
    if len(concept_indices) == 0:
        raise ValueError("concept_indices must contain at least one datapoint.")
    if not np.issubdtype(concept_indices.dtype, np.integer):
        raise ValueError("concept_indices must contain integer indices.")

    concept_indices = concept_indices.astype(int, copy=False)
    if np.any(concept_indices < 0) or np.any(concept_indices >= probs.shape[0]):
        raise ValueError("concept_indices contains indices outside the dataset.")
    if len(np.unique(concept_indices)) != len(concept_indices):
        raise ValueError("concept_indices contains duplicate indices.")
    if len(concept_indices) == probs.shape[0]:
        raise ValueError("At least one datapoint must remain outside the concept.")

    if not isinstance(m, int) or m < 1:
        raise ValueError("m must be a positive integer.")
    if epsilon < 0.0:
        raise ValueError("epsilon must be non-negative.")
    if not isinstance(max_concept_sample_size, int) or max_concept_sample_size < 1:
        raise ValueError("max_concept_sample_size must be a positive integer.")

    return concept_indices, probs, y_true


def _make_nonconcept_pool(concept_indices: np.ndarray, n_datapoints: int) -> np.ndarray:
    """Return indices for all datapoints that do not belong to the concept."""
    is_concept = np.zeros(n_datapoints, dtype=bool)
    is_concept[concept_indices] = True
    return np.flatnonzero(~is_concept)


def debiased_equal_mass_ece(
    probs: np.ndarray,
    y_true: np.ndarray,
    n_bins: int = 15,
) -> float:
    """Compute debiased equal-mass expected calibration error.

    ECE compares confidence with empirical accuracy, so probabilities alone are
    not enough: we also need y_true to know whether each prediction was correct.

    Equal-mass bins mean that we first sort datapoints by confidence and then
    split them into bins with roughly the same number of datapoints.  This is
    different from equal-width bins, which split the confidence interval [0, 1]
    into fixed-width intervals.
    """
    probs, y_true = _validate_probability_inputs(probs, y_true, n_bins)

    n_datapoints = probs.shape[0]
    if n_datapoints < 2:
        raise ValueError("At least two datapoints are needed for debiased ECE.")

    # Confidence is the model's probability for the class it predicts.
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    correct = predictions == y_true

    order = np.argsort(confidences)
    bins = np.array_split(order, min(n_bins, n_datapoints))

    ece_squared = 0.0
    for bin_indices in bins:
        bin_size = len(bin_indices)
        if bin_size == 0:
            continue

        bin_accuracy = float(np.mean(correct[bin_indices]))
        bin_confidence = float(np.mean(confidences[bin_indices]))

        # The correction below removes an estimate of finite-sample noise from
        # the squared calibration gap.  We clip at zero because the correction
        # can be larger than the observed squared gap in small/noisy bins.
        squared_gap = (bin_accuracy - bin_confidence) ** 2
        bias_estimate = bin_accuracy * (1.0 - bin_accuracy) / max(bin_size - 1, 1)
        debiased_squared_gap = max(squared_gap - bias_estimate, 0.0)

        ece_squared += (bin_size / n_datapoints) * debiased_squared_gap

    return float(np.sqrt(ece_squared))


def _confidence_match_sample(
    concept_confidences: np.ndarray,
    sorted_nonconcept_confidences: np.ndarray,
    sorted_nonconcept_indices: np.ndarray,
    epsilon: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample one non-concept set matched to the concept by confidence.

    The non-concept confidences are sorted once before this function is called.
    For a concept confidence p, np.searchsorted quickly finds the slice of
    non-concept datapoints with confidence in [p - epsilon, p + epsilon].
    """
    matched_indices: list[int] = []
    used_indices: set[int] = set()

    for confidence in concept_confidences:
        lower = confidence - epsilon
        upper = confidence + epsilon

        left = np.searchsorted(sorted_nonconcept_confidences, lower, side="left")
        right = np.searchsorted(sorted_nonconcept_confidences, upper, side="right")
        candidates = sorted_nonconcept_indices[left:right]

        if len(candidates) == 0:
            raise ValueError(
                "Confidence matching failed: a concept datapoint has no "
                f"non-concept candidates within epsilon={epsilon}."
            )

        # Sampling is without replacement inside this null sample.  Usually a
        # random candidate has not been used yet, so we try a few direct random
        # draws before building the slower filtered list of unused candidates.
        chosen = None
        for _ in range(20):
            candidate = int(rng.choice(candidates))
            if candidate not in used_indices:
                chosen = candidate
                break

        if chosen is None:
            unused_candidates = [idx for idx in candidates if int(idx) not in used_indices]
        else:
            unused_candidates = []

        if chosen is None and len(unused_candidates) == 0:
            raise ValueError(
                "Confidence matching failed: not enough unused non-concept "
                f"candidates within epsilon={epsilon}."
            )

        if chosen is None:
            chosen = int(rng.choice(unused_candidates))

        matched_indices.append(chosen)
        used_indices.add(chosen)

    return np.asarray(matched_indices, dtype=int)


def concept_calibration_test(
    concept_indices: np.ndarray,
    probs: np.ndarray,
    y_true: np.ndarray,
    m: int = 500,
    epsilon: float = 0.05,
    max_concept_sample_size: int = 500,
    n_bins: int = 15,
    adaptive_bins: bool = True,
    random_state: int | None = None,
    return_null_distribution: bool = True,
) -> dict[str, Any]:
    """Test whether a concept's calibration differs from the matched null.

    The concept ECE is the test statistic.  The null distribution is built from
    m non-concept samples that have similar confidence values to the sampled
    concept datapoints.
    """
    concept_indices, probs, y_true = _validate_inputs(
        concept_indices=concept_indices,
        probs=probs,
        y_true=y_true,
        m=m,
        epsilon=epsilon,
        max_concept_sample_size=max_concept_sample_size,
        n_bins=n_bins,
    )

    rng = np.random.default_rng(random_state)

    # The algorithm uses at most 500 concept datapoints to keep the test cheap.
    concept_sample_size = min(len(concept_indices), max_concept_sample_size)
    sampled_concept_indices = rng.choice(
        concept_indices,
        size=concept_sample_size,
        replace=False,
    )

    if adaptive_bins:
        n_bins_used = min(20, max(3, concept_sample_size // 15))
    else:
        n_bins_used = n_bins

    concept_ece = debiased_equal_mass_ece(
        probs[sampled_concept_indices],
        y_true[sampled_concept_indices],
        n_bins=n_bins_used,
    )

    # Compute confidence once for every datapoint.  Matching is based on these
    # confidence values, not on the true labels.
    all_confidences = np.max(probs, axis=1)
    concept_confidences = all_confidences[sampled_concept_indices]

    nonconcept_indices = _make_nonconcept_pool(concept_indices, probs.shape[0])
    nonconcept_confidences = all_confidences[nonconcept_indices]

    # Sort non-concept confidences once.  Then each epsilon lookup is logarithmic
    # instead of scanning all non-concept datapoints repeatedly.
    sort_order = np.argsort(nonconcept_confidences)
    sorted_nonconcept_confidences = nonconcept_confidences[sort_order]
    sorted_nonconcept_indices = nonconcept_indices[sort_order]

    null_distribution = np.empty(m, dtype=float)
    for j in range(m):
        matched_indices = _confidence_match_sample(
            concept_confidences=concept_confidences,
            sorted_nonconcept_confidences=sorted_nonconcept_confidences,
            sorted_nonconcept_indices=sorted_nonconcept_indices,
            epsilon=epsilon,
            rng=rng,
        )

        null_distribution[j] = debiased_equal_mass_ece(
            probs[matched_indices],
            y_true[matched_indices],
            n_bins=n_bins_used,
        )

    percentile_values = np.percentile(null_distribution, [5, 25, 50, 75, 95])
    null_percentiles = {
        percentile: float(value)
        for percentile, value in zip([5, 25, 50, 75, 95], percentile_values)
    }

    # Two-sided empirical p-value: H1 says the concept ECE differs from the
    # general model calibration, not only that it is larger.  The +1 correction
    # avoids exactly-zero p-values when m is finite.
    count_lower = int(np.sum(null_distribution <= concept_ece))
    count_upper = int(np.sum(null_distribution >= concept_ece))
    lower_tail = (count_lower + 1) / (m + 1)
    upper_tail = (count_upper + 1) / (m + 1)
    p_value = min(1.0, 2.0 * min(lower_tail, upper_tail))

    result: dict[str, Any] = {
        "concept_ece": float(concept_ece),
        "null_percentiles": null_percentiles,
        "p_value": float(p_value),
    }
    if return_null_distribution:
        result["null_distribution"] = null_distribution

    return result
