
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REVISION_DIR = PROJECT_ROOT / "results/scientific_reports_revision"

INPUT_PATH = REVISION_DIR / "heldout_query_metrics.csv"
OUTPUT_PATH = REVISION_DIR / "heldout_bootstrap_ci.csv"

N_BOOTSTRAP = 10000
SEED = 42

METHODS = ["Hybrid", "MLP", "VQC"]
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


def main():
    df = pd.read_csv(INPUT_PATH)

    assert df["query_id"].nunique() == 60
    assert len(df) == 180
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

        values = pivot[METHODS].to_numpy()
        n_queries = len(values)

        # Identical bootstrap samples for all methods.
        indices = rng.integers(
            0,
            n_queries,
            size=(N_BOOTSTRAP, n_queries),
        )

        bootstrap_means = values[indices].mean(axis=1)

        for method_index, method in enumerate(METHODS):
            observed_mean = values[:, method_index].mean()

            lower, upper = np.percentile(
                bootstrap_means[:, method_index],
                [2.5, 97.5],
            )

            results.append({
                "metric": metric,
                "method": method,
                "mean": observed_mean,
                "ci_lower": lower,
                "ci_upper": upper,
                "confidence_level": 0.95,
                "bootstrap_repetitions": N_BOOTSTRAP,
                "seed": SEED,
            })

    output = pd.DataFrame(results)
    output.to_csv(OUTPUT_PATH, index=False)

    print("Bootstrap confidence intervals calculated.")
    print("Queries:", 60)
    print("Bootstrap repetitions:", N_BOOTSTRAP)
    print("Results saved to:", OUTPUT_PATH)

    print("\n95% confidence intervals:")
    print(
        output[
            output["metric"].isin(["MRR", "nDCG@10", "MAP"])
        ][["metric", "method", "mean", "ci_lower", "ci_upper"]]
        .round(4)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
