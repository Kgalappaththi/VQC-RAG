"""Run additional VQC seeds for Scientific Reports revision."""

from experiments.scientific_reports_revision.train_vqc_with_validation import (
    train_vqc_with_validation,
    save_best_model,
)


def main():
    seeds = [7, 21]

    for seed in seeds:
        print(f"\nStarting VQC experiment with seed {seed}")

        result = train_vqc_with_validation(
            n_qubits=4,
            depth=4,
            epochs=100,
            learning_rate=0.01,
            margin=0.1,
            seed=seed,
        )

        model_path = save_best_model(
            result=result,
            n_qubits=4,
            depth=4,
            seed=seed,
        )

        print(f"Completed seed {seed}")
        print(f"Saved checkpoint: {model_path}")

    print("\nAll additional VQC experiments completed.")


if __name__ == "__main__":
    main()