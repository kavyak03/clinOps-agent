import faiss
import numpy as np

def build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # cosine similarity if embeddings normalized
    index.add(embeddings)
    return index

def search(index, query_emb: np.ndarray, k: int = 5):
    scores, idxs = index.search(query_emb, k)
    return scores[0], idxs[0]
