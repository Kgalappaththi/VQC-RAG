import numpy as np

def create_pair_feature(query_embedding, document_embedding):
    difference = np.abs(query_embedding - document_embedding)
    product = query_embedding * document_embedding

    feature = np.concatenate([
        query_embedding,
        document_embedding,
        difference,
        product
    ])

    return feature