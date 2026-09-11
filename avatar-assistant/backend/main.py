from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import os
from typing import Optional
from rag import retrieve_context, RagNotReadyError
from llm import generate_answer, LLMNotConfiguredError
from vision import generate_vision_answer, InvalidImageError

load_dotenv()

app = FastAPI(title="Personal Avatar Assistant API")

# Allow the Vite dev server, plus a configurable production frontend
# origin (set FRONTEND_URL in backend/.env once deployed).
allowed_origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]
frontend_url = os.getenv("FRONTEND_URL")
if frontend_url:
    allowed_origins.append(frontend_url)

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {"message": "Backend is running. Try /health or /docs"}


@app.get("/health")
def health():
    """Session 1 sanity check endpoint. Session 6 will add /vision."""
    return {
        "status": "ok",
        "service": "personal-avatar-assistant-backend",
        "env_loaded": os.getenv("APP_ENV", "not set"),
    }


class ChatRequest(BaseModel):
    question: str


class ChatResponse(BaseModel):
    answer: str
    sources: list[str]


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    """RAG-grounded chat endpoint: retrieve relevant chunks from ChromaDB,
    then ask Claude to answer using only that context."""
    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    try:
        chunks = retrieve_context(question)
    except RagNotReadyError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    if not chunks:
        return ChatResponse(
            answer="I don't have information about that yet — try asking about "
            "Agentix System's services, agents, or how to get in touch.",
            sources=[],
        )

    try:
        answer = generate_answer(question, chunks)
    except LLMNotConfiguredError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=502, detail=f"LLM request failed: {exc}"
        ) from exc

    sources = sorted({c["source"] for c in chunks})
    return ChatResponse(answer=answer, sources=sources)


class VisionRequest(BaseModel):
    image: str  # data URL, e.g. "data:image/jpeg;base64,...."
    question: Optional[str] = None

@app.post("/vision", response_model=ChatResponse)
def vision(req: VisionRequest):
    """Analyzes a single captured camera frame with Claude's vision
    capability. Not RAG-grounded — this is about what's visually in
    front of the camera, not company knowledge."""
    try:
        answer, sources = generate_vision_answer(req.image, req.question)
    except InvalidImageError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except LLMNotConfiguredError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=502, detail=f"Vision request failed: {exc}"
        ) from exc

    return ChatResponse(answer=answer, sources=sources)
