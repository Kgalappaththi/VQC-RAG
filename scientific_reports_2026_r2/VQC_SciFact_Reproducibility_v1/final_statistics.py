from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("results")
OUT = ROOT / "final_statistical_analysis"
OUT.mkdir(exist_ok=True)

SEEDS = [7, 21, 42]
METRIC = "nDCG@10"
N_BOOT = 20000
N_PERM = 100000
RANDOM_SEED = 20261111

rng = np.random.default_rng(RANDOM_SEED)

classical = pd.read_csv(
    ROOT / "classical/per_query_classical_metrics.csv"
)

def get_series(df, method):
    selected = df[
        (df["partition"] == "test") &
        (df["method"] == method)
    ][["query_id", METRIC]].copy()

    if selected["query_id"].duplicated().any():
        raise ValueError(f"Duplicate queries: {method}")

    return selected.set_index("query_id")[METRIC].sort_index()

hybrid = get_series(classical, "Hybrid")
mlp = []

for seed in SEEDS:
    mlp.append(
        get_series(classical, f"Hybrid+MLP_seed{seed}")
    )

vqc = []
rrf = []
vqc_alone = []

for seed in SEEDS:
    folder = (
        "audit_seed42_recomputed"
        if seed == 42 else f"audit_seed{seed}_final"
    )

    df = pd.read_csv(
        ROOT / folder / "vqc_per_query_metrics.csv"
    )

    vqc.append(
        get_series(df, "Hybrid+VQC_weighted_0.8")
    )
    rrf.append(
        get_series(df, "Hybrid+VQC_RRF_k60")
    )
    vqc_alone.append(
        get_series(df, "VQC")
    )

all_series = [hybrid] + mlp + vqc + rrf + vqc_alone
query_ids = hybrid.index

if len(query_ids) != 60:
    raise ValueError("Expected exactly 60 test queries")

for s in all_series:
    if not s.index.equals(query_ids):
        raise ValueError("Query IDs do not match")

def mean_seeds(series_list):
    return np.column_stack(
        [s.to_numpy() for s in series_list]
    ).mean(axis=1)

scores = {
    "Hybrid": hybrid.to_numpy(),
    "Hybrid+MLP": mean_seeds(mlp),
    "Hybrid+VQC": mean_seeds(vqc),
    "Hybrid+VQC_RRF": mean_seeds(rrf),
    "VQC_alone": mean_seeds(vqc_alone),
}

per_query = pd.DataFrame({
    "query_id": query_ids,
    **scores
})

per_query.to_csv(
    OUT / "matched_per_query_test_scores.csv",
    index=False
)

contrasts = [
    ("Hybrid+MLP", "Hybrid"),
    ("Hybrid+VQC", "Hybrid"),
    ("Hybrid+VQC_RRF", "Hybrid"),
    ("VQC_alone", "Hybrid"),
    ("Hybrid+VQC", "Hybrid+MLP"),
]

rows = []

for a, b in contrasts:
    delta = scores[a] - scores[b]
    n = len(delta)

    indices = rng.integers(
        0, n, size=(N_BOOT, n)
    )
    bootstrap = delta[indices].mean(axis=1)

    low, high = np.percentile(
        bootstrap, [2.5, 97.5]
    )

    observed = abs(delta.mean())
    extreme = 0

    # Batched paired sign-flip randomisation.
    for _ in range(N_PERM // 1000):
        signs = rng.choice(
            [-1, 1], size=(1000, n)
        )
        simulated = abs(
            (signs * delta).mean(axis=1)
        )
        extreme += np.count_nonzero(
            simulated >= observed - 1e-12
        )

    p = (extreme + 1) / (N_PERM + 1)

    rows.append({
        "comparison": f"{a} vs {b}",
        "mean_delta": delta.mean(),
        "ci95_low": low,
        "ci95_high": high,
        "p_unadjusted": p,
    })

stats = pd.DataFrame(rows)

# Holm correction across all five contrasts.
pvals = stats["p_unadjusted"].to_numpy()
order = np.argsort(pvals)
adjusted = np.empty(len(pvals))
running_max = 0.0

for rank, index in enumerate(order):
    value = min(
        1.0, (len(pvals) - rank) * pvals[index]
    )
    running_max = max(running_max, value)
    adjusted[index] = running_max

stats["p_holm"] = adjusted

stats.to_csv(
    OUT / "paired_statistical_comparisons.csv",
    index=False
)

print("\nMEAN TEST nDCG@10")
for name, values in scores.items():
    print(f"{name:22s}: {values.mean():.6f}")

print("\nPAIRED STATISTICAL COMPARISONS")
print(stats.to_string(index=False))

print("\nSaved results in:", OUT.resolve())
