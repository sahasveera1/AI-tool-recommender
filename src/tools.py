from databricks.vector_search.client import VectorSearchClient
from langchain_core.tools import tool

from src.config import VECTOR_SEARCH_ENDPOINT, VECTOR_INDEX_NAME

vs_client = VectorSearchClient()

@tool
def search_ai_tools(query: str, num_results: int = 5) -> str:
    """Search for relevant AI tools based on a natural language query."""
    index = vs_client.get_index(
        endpoint_name=VECTOR_SEARCH_ENDPOINT,
        index_name=VECTOR_INDEX_NAME,
    )

    results = index.similarity_search(
        query_text=query,
        columns=["tool_summary", "vendor", "tool", "product_id"],
        num_results=num_results,
    )

    hits = results.get("result", {}).get("data_array", [])
    if not hits:
        return "No matching AI tools found."

    formatted = [
        f"[{i+1}] {hit[1]} - {hit[2]} (ID: {hit[3]})\n{hit[0]}"
        for i, hit in enumerate(hits)
    ]
    return "\n\n".join(formatted)