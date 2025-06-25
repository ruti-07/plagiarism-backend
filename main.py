from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from sentence_transformers import SentenceTransformer, util

class InputText(BaseModel):
    text: str

app = FastAPI()

# Load a lightweight model (keeps memory under 512MB)
model = SentenceTransformer('all-MiniLM-L6-v2')

# CORS config for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # frontend origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Sample reference text for comparison
reference_text = "This is a sample reference for similarity checking."

@app.post("/check")
def check_text(data: InputText):
    query_embedding = model.encode(data.text, convert_to_tensor=True)
    reference_embedding = model.encode(reference_text, convert_to_tensor=True)
    score = util.pytorch_cos_sim(query_embedding, reference_embedding).item()
    return {"similarity_score": round(score, 4)}

# Ensure Render binds correctly
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=10000)