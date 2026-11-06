
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REVISION_DIR = PROJECT_ROOT / "results/scientific_reports_revision"

INPUT_PATH = REVISION_DIR / "heldout_query_metrics.csv"
OUTPUT_PATH = REVISION_DIR / "heldout_paired_comparisons.csv"

N_RESAMPLES = 10000
SEED = 42

METRICS = [
    "Recall@3",
    "Recall@5",
    "Recall@10",
    "Recall@20",
    "MRR",
    "nDCG@5",
    "nDCG@10",
    "MAP",
]

COMPARISONS = [
    ("Hybrid", "MLP"),
    ("Hybrid", "VQC"),
    ("MLP", "VQC"),
]


def main():
    df = pd.read_csv(INPUT_PATH)

    assert len(df) == 180
    assert df["query_id"].nunique() == 60
    assert not df.duplicated(["query_id", "method"]).any()

    rng = np.random.default_rng(SEED)
    results = []

    for metric in METRICS:
        pivot = df.pivot(
            index="query_id",
            columns="method",
            values=metric,
        ).sort_index()

        assert pivot.shape == (60, 3)
        assert pivot.notna().all().all()

        for method_a, method_b in COMPARISONS:
            differences = (
                pivot[method_a] - pivot[method_b]
            ).to_numpy()

            observed_difference = float(
                differences.mean()
            )

            n_queries = len(differences)

            # Paired bootstrap confidence interval.
            indices = rng.integers(
                0,
                n_queries,
                size=(N_RESAMPLES, n_queries),
            )

            bootstrap_differences = (
                differences[indices].mean(axis=1)
            )

            ci_lower, ci_upper = np.percentile(
                bootstrap_differences,
                [2.5, 97.5],
            )

            # Two-sided paired sign-flip permutation test.
            signs = rng.choice(
                [-1, 1],
                size=(N_RESAMPLES, n_queries),
            )

            null_differences = (
                signs * differences
            ).mean(axis=1)

            extreme_count = np.count_nonzero(
                np.abs(null_differences)
                >= abs(observed_difference) - 1e-12
            )

            p_value = (
                extreme_count + 1
            ) / (N_RESAMPLES + 1)

            results.append({
                "metric": metric,
                "method_a": method_a,
                "method_b": method_b,
                "mean_difference": observed_difference,
                "ci_lower": ci_lower,
                "ci_upper": ci_upper,
                "permutation_p_value": p_value,
                "n_queries": n_queries,
                "resamples": N_RESAMPLES,
                "seed": SEED,
            })

    output = pd.DataFrame(results)
    output.to_csv(OUTPUT_PATH, index=False)

    print("Paired ranking comparisons completed.")
    print("Queries:", 60)
    print("Resamples:", N_RESAMPLES)
    print("Comparisons:", len(output))
    print("Saved to:", OUTPUT_PATH)

    print("\nPaired comparisons for nDCG@10:")
    print(
        output.loc[
            output["metric"] == "nDCG@10",
            [
                "method_a",
                "method_b",
                "mean_difference",
                "ci_lower",
                "ci_upper",
                "permutation_p_value",
            ],
        ].round(5).to_string(index=False)
    )


if __name__ == "__main__":
    main()
