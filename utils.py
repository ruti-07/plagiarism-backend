import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

# Load model once
model = SentenceTransformer("all-MiniLM-L6-v2")

# Load FAISS index and filenames (created from build_index.py)
index = faiss.read_index("faiss_index.index")
with open("filenames.pkl", "rb") as f:
    filenames = pickle.load(f)

def check_similarity(input_text):
    input_embedding = model.encode([input_text], convert_to_numpy=True)
    distances, indices = index.search(input_embedding, k=1)
    similarity_score = 1 - float(distances[0][0])  # Convert to native float

    return {
        "score": round(similarity_score * 100, 2),
        "plagiarized": bool(similarity_score > 0.8),
        "source": str(filenames[indices[0][0]])
    }