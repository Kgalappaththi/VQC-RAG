
"""Evaluate additional VQC seeds on the held-out test candidates."""

import pickle
import numpy as np

from experiments.scientific_reports_revision.evaluate_heldout_ranking import (
    MODELS_DIR,
    REVISION_DIR,
    load_evaluation_data,
    build_candidate_features,
    compute_vqc_scores,
    build_trainable_vqc,
)


def main():
    (
        test_candidates,
        test_qrels,
        doc_embeddings,
        query_embeddings,
        doc_id_to_index,
        query_id_to_index,
    ) = load_evaluation_data()

    candidate_features = build_candidate_features(
        test_candidates,
        doc_embeddings,
        query_embeddings,
        doc_id_to_index,
        query_id_to_index,
    )

    for seed in [7, 21]:
        print(f"\nEvaluating VQC seed {seed}")

        model_path = MODELS_DIR / f"vqc_4q_d4_seed{seed}.pkl"

        with open(model_path, "rb") as file:
            checkpoint = pickle.load(file)

        config = checkpoint["config"]
        assert config["n_qubits"] == 4
        assert config["depth"] == 4
        assert config["seed"] == seed

        circuit = build_trainable_vqc(
            n_qubits=config["n_qubits"],
            depth=config["depth"],
        )

        scores = compute_vqc_scores(
            candidate_features,
            checkpoint,
            circuit,
        )

        assert len(scores) == len(test_candidates)
        assert np.isfinite(scores).all()

        output = test_candidates.copy()
        output["vqc_score"] = scores
        output["seed"] = seed

        output_path = (
            REVISION_DIR / f"heldout_vqc_seed{seed}_scores.csv"
        )

        assert not output_path.exists(), (
            f"Output already exists: {output_path}"
        )

        output.to_csv(output_path, index=False)

        print(f"Saved seed-{seed} scores to: {output_path}")

    print("\nAdditional VQC evaluations completed.")


if __name__ == "__main__":
    main()
