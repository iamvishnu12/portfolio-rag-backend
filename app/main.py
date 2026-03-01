from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.models import QueryRequest
from app.rag_service import ask_question

app = FastAPI(title="Resume RAG API")

# Allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # change later to your domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def health():
    return {"status": "RAG API running"}

@app.post("/ask")
def ask(query: QueryRequest):

    answer = ask_question(query.question)

    return {
        "question": query.question,
        "answer": answer
    }