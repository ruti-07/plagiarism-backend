from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import gc

class InputText(BaseModel):
    text: str

app = FastAPI()

# CORS: Enable for local dev + Swagger UI
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "https://plagiarism-backend-2.onrender.com"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/check")
def check_text(data: InputText):
    from sentence_transformers import SentenceTransformer, util

    reference_text = "This is a sample reference for similarity checking."

    # ✅ Lightest available transformer model (~45MB)
    model = SentenceTransformer('paraphrase-albert-small-v2')

    query_embedding = model.encode(data.text, convert_to_tensor=True)
    reference_embedding = model.encode(reference_text, convert_to_tensor=True)

    score = util.pytorch_cos_sim(query_embedding, reference_embedding).item()

    # 🧹 Free up memory post-inference
    del model
    gc.collect()

    return {"similarity_score": round(score, 4)}

# 🚀 Render entry point
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=10000)