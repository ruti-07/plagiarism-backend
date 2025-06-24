import os
import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")
doc_dir = "documents"

documents = []
filenames = []

for fname in os.listdir(doc_dir):
    with open(os.path.join(doc_dir, fname), encoding="utf-8") as f:
        documents.append(f.read())
        filenames.append(fname)

# Convert documents to embeddings
doc_embeddings = model.encode(documents, convert_to_numpy=True)

# Create and save FAISS index
dim = doc_embeddings.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(doc_embeddings)
faiss.write_index(index, "faiss_index.index")

with open("filenames.pkl", "wb") as f:
    pickle.dump(filenames, f)

print("✅ FAISS index built and saved.")