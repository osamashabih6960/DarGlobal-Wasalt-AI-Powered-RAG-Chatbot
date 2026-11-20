
"""FastAPI application for the DarGlobal x Wasalt RAG chatbot."""

import os
import time
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field


# Load environment variables from project root .env
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

STATIC = Path(__file__).parent / "static"

app = FastAPI(
    title="DarGlobal x Wasalt Assistant",
    description="RAG chatbot powered by Groq, LangGraph and FAISS.",
    version="1.0.0",
)


# --------------------------------------------------
# Request Schema
# --------------------------------------------------

class ChatIn(BaseModel):
    message: str = Field(min_length=1, max_length=1000)
    session_id: str = Field(default="default", max_length=64)


# --------------------------------------------------
# LangGraph Initialization
# --------------------------------------------------

@lru_cache(maxsize=1)
def get_graph():
    """Initialize the RAG graph only once."""

    from app.graph import build_graph

    return build_graph()


# --------------------------------------------------
# Health Check
# --------------------------------------------------

@app.get("/health")
def health():
    """Check whether the API server is running."""

    return {
        "status": "ok",
        "model": os.getenv("MODEL", "unset"),
        "provider": "Groq",
    }


# --------------------------------------------------
# Chat Endpoint
# --------------------------------------------------

@app.post("/chat")
def chat(body: ChatIn, bg: BackgroundTasks):
    """Process user messages through the RAG chatbot."""

    from app.graph import ask
    from app.tracking import log_chat

    start_time = time.time()

    try:
        result = ask(
            get_graph(),
            body.message,
            body.session_id,
        )

    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Model unavailable, please retry: {type(exc).__name__}",
        ) from exc

    latency = time.time() - start_time

    bg.add_task(
        log_chat,
        latency,
        result["route"] or "",
        len(result["sources"]),
    )

    return result


# --------------------------------------------------
# Frontend
# --------------------------------------------------

@app.get("/")
def index():
    """Serve the chatbot frontend."""

    return FileResponse(STATIC / "index.html")


app.mount(
    "/static",
    StaticFiles(directory=STATIC),
    name="static",
)
