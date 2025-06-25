from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import gc

class InputText(BaseModel):
    text: str

app = FastAPI()

# CORS config for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Adjust if frontend is hosted elsewhere
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/check")
def check_text(data: InputText):
    from sentence_transformers import SentenceTransformer, util

    reference_text = "This is a sample reference for similarity checking."

    # 🔄 Lazy-load the model inside the route
    model = SentenceTransformer('all-MiniLM-L6-v2')
    query_embedding = model.encode(data.text, convert_to_tensor=True)
    reference_embedding = model.encode(reference_text, convert_to_tensor=True)
    score = util.pytorch_cos_sim(query_embedding, reference_embedding).item()

    # 🧹 Clean up memory
    del model
    gc.collect()

    return {"similarity_score": round(score, 4)}

# 🔧 Ensures Render binds to the right port
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=10000)