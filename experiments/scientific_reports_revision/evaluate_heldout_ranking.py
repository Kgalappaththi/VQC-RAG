from pathlib import Path
import pickle

import numpy as np
import pandas as pd

from src.quantum.features import (
    create_pair_feature,
    transform_features,
)

from src.quantum.trainer import (
    build_trainable_vqc,
    quantum_relevance_score,
)


# Repository and input paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]

RESULTS_DIR = PROJECT_ROOT / "results"
REVISION_DIR = RESULTS_DIR / "scientific_reports_revision"
MODELS_DIR = REVISION_DIR / "models"

CANDIDATES_PATH = RESULTS_DIR / "hybrid_top20_candidates.csv"
QRELS_PATH = PROJECT_ROOT / "data/qrels/qrels.csv"
SPLITS_PATH = REVISION_DIR / "query_splits.csv"

DOC_EMBEDDINGS_PATH = RESULTS_DIR / "document_embeddings.npy"
QUERY_EMBEDDINGS_PATH = RESULTS_DIR / "query_embeddings.npy"

DOC_METADATA_PATH = RESULTS_DIR / "document_metadata.csv"
QUERY_METADATA_PATH = RESULTS_DIR / "query_metadata.csv"

MLP_MODEL_PATH = MODELS_DIR / "mlp_4components_seed42.pkl"
VQC_MODEL_PATH = MODELS_DIR / "vqc_4q_d4_seed42.pkl"

def load_evaluation_data():
    """Load datasets and select the held-out test queries."""

    candidates = pd.read_csv(CANDIDATES_PATH)
    qrels = pd.read_csv(QRELS_PATH)
    splits = pd.read_csv(SPLITS_PATH)

    doc_embeddings = np.load(DOC_EMBEDDINGS_PATH)
    query_embeddings = np.load(QUERY_EMBEDDINGS_PATH)

    doc_meta = pd.read_csv(DOC_METADATA_PATH)
    query_meta = pd.read_csv(QUERY_METADATA_PATH)

    # Select only the held-out test queries.
    test_ids = set(
        splits.loc[splits["split"] == "test", "query_id"]
    )

    test_candidates = candidates[
        candidates["query_id"].isin(test_ids)
    ].copy()

    test_qrels = qrels[
        qrels["query_id"].isin(test_ids)
    ].copy()

    # Verify the held-out evaluation structure.
    assert len(test_ids) == 60
    assert test_candidates["query_id"].nunique() == 60
    assert len(test_candidates) == 1200

    # Verify embedding metadata alignment.
    assert len(doc_meta) == len(doc_embeddings)
    assert len(query_meta) == len(query_embeddings)

    # Create ID-to-embedding-row mappings.
    doc_id_to_index = {
        str(doc_id): idx
        for idx, doc_id in enumerate(doc_meta["document_id"])
    }

    query_id_to_index = {
        str(query_id): idx
        for idx, query_id in enumerate(query_meta["query_id"])
    }

    # Ensure every candidate has the required embeddings.
    assert set(test_candidates["document_id"].astype(str)).issubset(
        doc_id_to_index
    )

    assert set(test_candidates["query_id"].astype(str)).issubset(
        query_id_to_index
    )

    print("Held-out evaluation data loaded successfully.")
    print("Test queries:", len(test_ids))
    print("Candidate documents:", len(test_candidates))

    return (
        test_candidates,
        test_qrels,
        doc_embeddings,
        query_embeddings,
        doc_id_to_index,
        query_id_to_index,
    )

def load_evaluation_models():
    """Load the trained MLP and VQC checkpoints."""

    with open(MLP_MODEL_PATH, "rb") as file:
        mlp_checkpoint = pickle.load(file)

    with open(VQC_MODEL_PATH, "rb") as file:
        vqc_checkpoint = pickle.load(file)

    # Verify compatible preprocessing.
    assert np.allclose(
        mlp_checkpoint["scaler"].mean_,
        vqc_checkpoint["scaler"].mean_
    )

    assert np.allclose(
        mlp_checkpoint["scaler"].scale_,
        vqc_checkpoint["scaler"].scale_
    )

    assert np.allclose(
        mlp_checkpoint["pca"].components_,
        vqc_checkpoint["pca"].components_
    )

    assert np.allclose(
        mlp_checkpoint["pca"].mean_,
        vqc_checkpoint["pca"].mean_
    )

    # Reconstruct the trained quantum circuit.
    config = vqc_checkpoint["config"]

    circuit = build_trainable_vqc(
        n_qubits=config["n_qubits"],
        depth=config["depth"]
    )

    print("MLP and VQC checkpoints loaded successfully.")
    print("VQC qubits:", config["n_qubits"])
    print("VQC depth:", config["depth"])
    print("MLP input dimensions:",
          mlp_checkpoint["model"].n_features_in_)

    return mlp_checkpoint, vqc_checkpoint, circuit


def build_candidate_features(
    test_candidates,
    doc_embeddings,
    query_embeddings,
    doc_id_to_index,
    query_id_to_index,
):
    """Build query-document features for all held-out candidates."""

    features = []

    for row in test_candidates.itertuples(index=False):
        query_id = str(row.query_id)
        document_id = str(row.document_id)

        query_embedding = query_embeddings[
            query_id_to_index[query_id]
        ]

        document_embedding = doc_embeddings[
            doc_id_to_index[document_id]
        ]

        feature = create_pair_feature(
            query_embedding,
            document_embedding
        )

        features.append(feature)

    features = np.asarray(features)

    assert features.shape == (1200, 1536)
    assert np.isfinite(features).all()

    print("Candidate features constructed successfully.")
    print("Feature matrix shape:", features.shape)

    return features

def compute_mlp_scores(candidate_features, mlp_checkpoint):
    """Calculate MLP relevance scores using saved preprocessing."""

    scaler = mlp_checkpoint["scaler"]
    pca = mlp_checkpoint["pca"]
    model = mlp_checkpoint["model"]

    # Apply training-fitted preprocessing.
    reduced_features = transform_features(
        candidate_features,
        scaler,
        pca
    )

    assert reduced_features.shape == (1200, 4)

    # Probability of the positive relevance class.
    positive_class_index = list(model.classes_).index(1.0)

    mlp_scores = model.predict_proba(
        reduced_features
    )[:, positive_class_index]

    assert len(mlp_scores) == 1200
    assert np.isfinite(mlp_scores).all()

    print("MLP relevance scores calculated successfully.")
    print("MLP scores shape:", mlp_scores.shape)
    print("MLP score range:",
          float(mlp_scores.min()),
          float(mlp_scores.max()))

    return mlp_scores
def compute_vqc_scores(
    candidate_features,
    vqc_checkpoint,
    circuit,
):
    """Calculate VQC scores using the trained quantum circuit."""

    scaler = vqc_checkpoint["scaler"]
    pca = vqc_checkpoint["pca"]
    weights = vqc_checkpoint["weights"]

    # Apply the training-fitted preprocessing.
    reduced_features = transform_features(
        candidate_features,
        scaler,
        pca
    )

    assert reduced_features.shape == (1200, 4)

    vqc_scores = []

    for index, feature in enumerate(reduced_features):

        score = quantum_relevance_score(
            circuit,
            feature,
            weights
        )

        vqc_scores.append(float(score))

        if (index + 1) % 100 == 0:
            print(
                f"VQC scored {index + 1}/1200 candidates"
            )

    vqc_scores = np.asarray(vqc_scores)

    assert vqc_scores.shape == (1200,)
    assert np.isfinite(vqc_scores).all()

    print("VQC relevance scores calculated successfully.")
    print("VQC scores shape:", vqc_scores.shape)
    print(
        "VQC score range:",
        float(vqc_scores.min()),
        float(vqc_scores.max())
    )

    return vqc_scores

def attach_model_scores(test_candidates, mlp_scores, vqc_scores):
    """Attach model scores while preserving candidate alignment."""

    assert len(test_candidates) == len(mlp_scores) == len(vqc_scores)

    scored_candidates = test_candidates.copy().reset_index(drop=True)

    scored_candidates["mlp_score"] = mlp_scores
    scored_candidates["vqc_score"] = vqc_scores

    assert scored_candidates[
        ["hybrid_score", "mlp_score", "vqc_score"]
    ].notna().all().all()

    assert not scored_candidates.duplicated(
        ["query_id", "document_id"]
    ).any()

    print("Model scores attached successfully.")
    print("Scored candidate records:", len(scored_candidates))
    print(
        "Scoring methods:",
        ["hybrid_score", "mlp_score", "vqc_score"]
    )

    return scored_candidates


if __name__ == "__main__":

    (
        test_candidates,
        test_qrels,
        doc_embeddings,
        query_embeddings,
        doc_id_to_index,
        query_id_to_index,
    ) = load_evaluation_data()

    mlp_checkpoint, vqc_checkpoint, circuit = load_evaluation_models()

    candidate_features = build_candidate_features(
        test_candidates,
        doc_embeddings,
        query_embeddings,
        doc_id_to_index,
        query_id_to_index,
    )

    mlp_scores = compute_mlp_scores(
        candidate_features,
        mlp_checkpoint
    )
    vqc_scores = compute_vqc_scores(
        candidate_features,
        vqc_checkpoint,
        circuit
    )

    scored_candidates = attach_model_scores(
        test_candidates,
        mlp_scores,
        vqc_scores
    )

    output_path = REVISION_DIR / "heldout_scored_candidates.csv"

    scored_candidates.to_csv(output_path, index=False)

    print("Scored candidates saved to:", output_path)  
