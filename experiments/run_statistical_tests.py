import pandas as pd
from scipy.stats import wilcoxon, ttest_rel


HYBRID_FILE = "results/hybrid_results.csv"
VQC_FILE = "results/vqc_reranking_results.csv"

METRICS = [
    "recall_at_3",
    "recall_at_5",
    "recall_at_10",
    "mrr",
    "ndcg_at_5",
    "ndcg_at_10",
    "map",
]


def main():
    hybrid = pd.read_csv(HYBRID_FILE)
    vqc = pd.read_csv(VQC_FILE)

    rows = []

    for metric in METRICS:
        h = hybrid[metric]
        q = vqc[metric]

        wilcoxon_stat, wilcoxon_p = wilcoxon(h, q)
        t_stat, t_p = ttest_rel(h, q)

        rows.append({
            "metric": metric,
            "hybrid_mean": h.mean(),
            "vqc_weighted_mean": q.mean(),
            "difference": q.mean() - h.mean(),
            "wilcoxon_stat": wilcoxon_stat,
            "wilcoxon_p": wilcoxon_p,
            "paired_t_stat": t_stat,
            "paired_t_p": t_p,
        })

    results = pd.DataFrame(rows)
    results.to_csv("results/statistics/hybrid_vs_vqc_weighted_tests.csv", index=False)

    print(results.to_string(index=False))


if __name__ == "__main__":
    main()