import logging
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from src.agent import agent_service
from src.schemas import ChatRequest, ChatResponse, NewChatResponse
from src.session_store import build_session_id

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)

app = FastAPI(title="RAVI API", version="1.0.0")

static_dir = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=static_dir), name="static")

@app.get("/")
def root():
    return FileResponse(static_dir / "index.html")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/api/chats", response_model=NewChatResponse)
def create_chat():
    return NewChatResponse(chat_id=str(uuid4()))

@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    try:
        session_id = build_session_id(req.user_id, req.chat_id)
        answer = agent_service.chat(req.message, session_id=session_id)
        return ChatResponse(
            response=answer,
            user_id=req.user_id,
            chat_id=req.chat_id,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
