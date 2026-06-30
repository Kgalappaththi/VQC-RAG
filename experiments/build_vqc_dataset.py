import sys

from src.quantum.dataset_builder import build_vqc_training_dataset
from src.config.vqc_experiments import VQC_EXPERIMENTS


def main():
    if len(sys.argv) < 2:
        raise ValueError("Please provide experiment ID, e.g. E004_6q_d4")

    experiment_id = sys.argv[1]
    config = VQC_EXPERIMENTS[experiment_id]

    build_vqc_training_dataset(
        candidates_csv="results/hybrid_top20_candidates.csv",
        qrels_csv="data/qrels/qrels.csv",
        document_embeddings_npy="results/document_embeddings.npy",
        document_metadata_csv="results/document_metadata.csv",
        query_embeddings_npy="results/query_embeddings.npy",
        query_metadata_csv="results/query_metadata.csv",
        output_npz=config["dataset_path"],
        output_preprocessor=config["preprocessor_path"],
        n_components=config["n_qubits"],
    )


if __name__ == "__main__":
    main()