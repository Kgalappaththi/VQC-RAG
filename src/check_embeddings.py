import numpy as np

# Load the embeddings
emb = np.load("results/document_embeddings.npy")

# Display information
print("Embedding Shape:", emb.shape)
print("Number of Documents:", emb.shape[0])
print("Embedding Dimension:", emb.shape[1])

# Show the first embedding vector (optional)
print("\nFirst Embedding Vector:")
print(emb[0])