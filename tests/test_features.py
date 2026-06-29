import numpy as np

from src.quantum.features import (
    create_pair_feature,
    fit_feature_reducer,
    transform_features
)


def test_create_pair_feature():
    query_embedding = np.array([1.0, 2.0, 3.0])
    doc_embedding = np.array([2.0, 4.0, 6.0])

    feature = create_pair_feature(query_embedding, doc_embedding)

    print("Pair feature:", feature)
    print("Feature shape:", feature.shape)

    assert feature.shape == (12,)


def test_feature_reducer():
    np.random.seed(42)

    features = np.random.rand(20, 12)

    reduced, scaler, pca = fit_feature_reducer(features, n_components=4)

    print("Reduced feature shape:", reduced.shape)

    assert reduced.shape == (20, 4)

    new_features = np.random.rand(5, 12)
    transformed = transform_features(new_features, scaler, pca)

    print("Transformed feature shape:", transformed.shape)

    assert transformed.shape == (5, 4)


if __name__ == "__main__":
    test_create_pair_feature()
    test_feature_reducer()
    print("Feature tests passed.")