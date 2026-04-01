import logging
import os

from databricks.vector_search.client import VectorSearchClient
from langchain_core.tools import tool

from src.config import VECTOR_SEARCH_ENDPOINT, VECTOR_INDEX_NAME

logger = logging.getLogger(__name__)
MAX_SEARCH_RESULTS = 5


def get_vs_client() -> VectorSearchClient:
    return VectorSearchClient(
        workspace_url=os.environ["DATABRICKS_HOST"],
        service_principal_client_id=os.environ["DATABRICKS_CLIENT_ID"],
        service_principal_client_secret=os.environ["DATABRICKS_CLIENT_SECRET"],
    )


@tool
def search_ai_tools(query: str, num_results: int = 5) -> str:
    """Search for relevant AI tools based on a natural language query."""
    capped_num_results = min(num_results, MAX_SEARCH_RESULTS)
    logger.info(
        "search_ai_tools called query=%r requested_num_results=%d applied_num_results=%d",
        query,
        num_results,
        capped_num_results,
    )
    vs_client = get_vs_client()

    index = vs_client.get_index(
        endpoint_name=VECTOR_SEARCH_ENDPOINT,
        index_name=VECTOR_INDEX_NAME,
    )

    results = index.similarity_search(
        query_text=query,
        columns=["tool_summary", "vendor", "tool", "product_id"],
        num_results=capped_num_results,
    )

    hits = results.get("result", {}).get("data_array", [])
    if not hits:
        return "No matching AI tools found."

    formatted = [
        f"[{i+1}] {hit[1]} - {hit[2]} (ID: {hit[3]})\n{hit[0]}"
        for i, hit in enumerate(hits)
    ]
    return "\n\n".join(formatted)
