from src.quantum.dataset_builder import build_vqc_training_dataset


def main():
    build_vqc_training_dataset(
        candidates_csv="results/hybrid_top20_candidates.csv",
        qrels_csv="data/qrels/qrels.csv",
        document_embeddings_npy="results/document_embeddings.npy",
        document_metadata_csv="results/document_metadata.csv",
        query_embeddings_npy="results/query_embeddings.npy",
        query_metadata_csv="results/query_metadata.csv",
        output_npz="results/vqc/vqc_training_dataset_4q.npz",
        output_preprocessor="results/vqc/vqc_preprocessor_4q.pkl",
        n_components=4,
    )


if __name__ == "__main__":
    main()