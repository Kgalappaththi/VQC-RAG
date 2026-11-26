from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


FIG_DIR = Path("paper/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)


def save_figure(name):
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"{name}.png", dpi=300, bbox_inches="tight")
    plt.savefig(FIG_DIR / f"{name}.pdf", bbox_inches="tight")
    plt.close()


def figure_3_training_loss():
    # Actual final losses from completed VQC training experiments
    data = pd.DataFrame({
        "Configuration": ["4Q-D2", "6Q-D4", "8Q-D4"],
        "Initial Loss": [0.133325, 0.127939, 0.117987],
        "Final Loss": [0.093986, 0.081628, 0.084614],
    })

    x = range(len(data))

    plt.figure(figsize=(6, 4))
    plt.plot(x, data["Initial Loss"], marker="o", label="Initial Loss")
    plt.plot(x, data["Final Loss"], marker="o", label="Final Loss")

    plt.xticks(x, data["Configuration"])
    plt.xlabel("VQC Configuration")
    plt.ylabel("Pairwise Ranking Loss")
    plt.title("VQC Training Loss Across Configurations")
    plt.legend()
    plt.grid(True, linestyle="--", linewidth=0.5)

    save_figure("fig3_vqc_training_loss")


def figure_4_quantum_ablation():
    df = pd.read_csv("results/statistics/vqc_ablation_results.csv")

    labels = df["experiment_id"]
    accuracy = df["pairwise_accuracy"]

    plt.figure(figsize=(7, 4))
    plt.bar(labels, accuracy)

    plt.xlabel("VQC Configuration")
    plt.ylabel("Pairwise Accuracy")
    plt.title("Quantum Capacity Ablation")
    plt.xticks(rotation=30, ha="right")
    plt.ylim(0, 0.75)
    plt.grid(axis="y", linestyle="--", linewidth=0.5)

    save_figure("fig4_quantum_ablation")


def figure_5_fusion_comparison():
    df = pd.read_csv("results/statistics/final_reranking_comparison.csv")

    metrics = ["Recall@3", "Recall@5", "Recall@10", "MRR", "nDCG@10", "MAP"]

    plot_df = df.set_index("Method")[metrics]

    plt.figure(figsize=(8, 5))
    plot_df.plot(kind="bar", figsize=(8, 5))

    plt.xlabel("Method")
    plt.ylabel("Score")
    plt.title("Fusion Strategy Comparison")
    plt.xticks(rotation=20, ha="right")
    plt.ylim(0, 0.85)
    plt.legend(loc="lower right", fontsize=8)
    plt.grid(axis="y", linestyle="--", linewidth=0.5)

    save_figure("fig5_fusion_comparison")


def main():
    figure_3_training_loss()
    figure_4_quantum_ablation()
    figure_5_fusion_comparison()

    print("Publication figures saved in paper/figures:")
    print("- fig3_vqc_training_loss.png / .pdf")
    print("- fig4_quantum_ablation.png / .pdf")
    print("- fig5_fusion_comparison.png / .pdf")


if __name__ == "__main__":
    main()