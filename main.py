from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from utils import check_similarity

class InputText(BaseModel):
    text: str

app = FastAPI()

# CORS config for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # frontend origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/check")
def check_text(data: InputText):
    result = check_similarity(data.text)
    return result