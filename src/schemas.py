from typing import Optional
from pydantic import BaseModel, Field

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)
    user_id: str = Field(..., min_length=1)
    chat_id: str = Field(..., min_length=1)

class ChatResponse(BaseModel):
    response: str
    user_id: str
    chat_id: str

class NewChatResponse(BaseModel):
    chat_id: str