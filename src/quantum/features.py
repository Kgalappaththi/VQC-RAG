import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def create_pair_feature(query_embedding, doc_embedding):
    return np.concatenate([
        query_embedding,
        doc_embedding,
        np.abs(query_embedding - doc_embedding),
        query_embedding * doc_embedding
    ])


def fit_feature_reducer(features, n_components=4):
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)

    pca = PCA(n_components=n_components, random_state=42)
    reduced = pca.fit_transform(scaled)

    return reduced, scaler, pca


def transform_features(features, scaler, pca):
    scaled = scaler.transform(features)
    return pca.transform(scaled)