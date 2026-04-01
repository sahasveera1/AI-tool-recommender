import os

LLM_ENDPOINT_NAME = os.getenv("LLM_ENDPOINT_NAME", "databricks-gpt-oss-120b")
VECTOR_SEARCH_ENDPOINT = os.getenv("VECTOR_SEARCH_ENDPOINT", "ai-tool-vector-endpoint")
VECTOR_INDEX_NAME = os.getenv("VECTOR_INDEX_NAME", "workspace.default.ai_tool_profiles_index")
APP_ENV = os.getenv("APP_ENV", "dev")