"""Example usage for the concept calibration test.

Run this file directly:

    python example_usage.py

It prints the concept ECE, null-distribution percentiles, and p-value.  It also
saves a histogram image of the null distribution as null_distribution.png.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from concept_calibration_test import concept_calibration_test


def softmax(raw_scores: np.ndarray) -> np.ndarray:
    """Convert arbitrary class scores into class probabilities."""
    # Subtracting the row maximum makes the exponential calculation more stable.
    shifted_scores = raw_scores - np.max(raw_scores, axis=1, keepdims=True)
    exp_scores = np.exp(shifted_scores)
    return exp_scores / exp_scores.sum(axis=1, keepdims=True)


def plot_null_distribution(
    null_distribution: np.ndarray,
    concept_ece: float,
    output_path: Path,
) -> None:
    """Save a histogram of the null ECE values and mark the concept ECE."""
    plt.figure(figsize=(8, 5))

    plt.hist(
        null_distribution,
        bins=30,
        color="#4C78A8",
        edgecolor="white",
        alpha=0.85,
    )
    plt.axvline(
        concept_ece,
        color="#E45756",
        linewidth=2,
        label=f"Concept ECE = {concept_ece:.4f}",
    )

    plt.title("Confidence-Matched Null Distribution")
    plt.xlabel("ECE")
    plt.ylabel("Number of null samples")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def main() -> None:
    rng = np.random.default_rng(42)

    n_datapoints = 2000
    n_classes = 3

    # Create synthetic probabilities with shape (n_datapoints, n_classes).
    # In real use, this would come from model.predict_proba(...) or similar.
    raw_scores = rng.normal(size=(n_datapoints, n_classes))
    probs = softmax(raw_scores)

    # In real use, y_true should be the real label for each datapoint.
    y_true = rng.integers(0, n_classes, size=n_datapoints)

    # Example concept: datapoints with indices 100, 101, ..., 349.
    concept_indices = np.arange(100, 350)

    result = concept_calibration_test(
        concept_indices=concept_indices,
        probs=probs,
        y_true=y_true,
        m=300,
        epsilon=0.05,
        random_state=42,
    )

    print("Concept ECE:", result["concept_ece"])
    print("Null percentiles:", result["null_percentiles"])
    print("Uncorrected p-value:", result["p_value"])

    image_path = Path(__file__).with_name("null_distribution.png")
    plot_null_distribution(
        null_distribution=result["null_distribution"],
        concept_ece=result["concept_ece"],
        output_path=image_path,
    )
    print("Saved null-distribution image to:", image_path)


if __name__ == "__main__":
    main()
