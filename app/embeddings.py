from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

def build_index(chunks):
    model = SentenceTransformer("all-MiniLM-L6-v2")

    embeddings = model.encode(chunks)

    dim = embeddings.shape[1]

    index = faiss.IndexFlatL2(dim)

    index.add(np.array(embeddings).astype("float32"))

    return model, index