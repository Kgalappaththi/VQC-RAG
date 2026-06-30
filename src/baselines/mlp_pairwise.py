import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, roc_auc_score


def build_pairwise_classification_dataset(pos_features, neg_features):
    X = np.vstack([pos_features, neg_features])

    y = np.concatenate([
        np.ones(len(pos_features)),
        np.zeros(len(neg_features))
    ])

    return X, y


def train_mlp_classifier(X, y, seed=42):
    model = MLPClassifier(
        hidden_layer_sizes=(16,),
        activation="relu",
        solver="adam",
        max_iter=500,
        random_state=seed
    )

    model.fit(X, y)

    return model


def evaluate_mlp_classifier(model, X, y):
    probabilities = model.predict_proba(X)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    return {
        "classification_accuracy": accuracy_score(y, predictions),
        "roc_auc": roc_auc_score(y, probabilities)
    }


def pairwise_accuracy_from_classifier(model, pos_features, neg_features):
    pos_scores = model.predict_proba(pos_features)[:, 1]
    neg_scores = model.predict_proba(neg_features)[:, 1]

    return float(np.mean(pos_scores > neg_scores))