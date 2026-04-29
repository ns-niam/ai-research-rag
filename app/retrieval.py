import numpy as np

def retrieve(query, chunks, model, index, k=2):
    q_emb = model.encode([query])

    distances, indices = index.search(
        np.array(q_emb).astype("float32"), k
    )

    return [chunks[i] for i in indices[0]]