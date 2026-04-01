def build_session_id(user_id: str, chat_id: str) -> str:
    return f"{user_id}::{chat_id}"