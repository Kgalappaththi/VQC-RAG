import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def fit_transform_pca(features, n_components=8):
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    pca = PCA(n_components=n_components)
    features_reduced = pca.fit_transform(features_scaled)

    return features_reduced, scaler, pca

def transform_pca(features, scaler, pca):
    features_scaled = scaler.transform(features)
    return pca.transform(features_scaled)