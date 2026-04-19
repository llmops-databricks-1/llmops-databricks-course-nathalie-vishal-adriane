"""RAG tool for RCA agent — arXiv paper vector search."""

import json
from collections.abc import Callable

from loguru import logger

from ordr_bhvr_rca_agent.vector_search import VectorSearchManager


def make_arxiv_search_fn(
    vs_manager: VectorSearchManager,
) -> Callable[[str, int], str]:
    """Return an exec_fn bound to *vs_manager* whose signature matches ARXIV_TOOL_SPEC.

    Args:
        vs_manager: Initialised VectorSearchManager pointing at the arXiv index.

    Returns:
        Callable ``search_arxiv_papers(query, num_results)`` ready to pass to ToolInfo.
    """

    def search_arxiv_papers(query: str, num_results: int = 3) -> str:
        """Search the arXiv paper vector index for context relevant to the query."""
        results = vs_manager.search(query=query, num_results=num_results)
        # Databricks VS returns a manifest plus data rows under result.data_array.
        rows = (
            results.get("result", {}).get("data_array", [])
            if isinstance(results, dict)
            else []
        )
        if not rows:
            logger.warning("arxiv_search: no results for query=%r", query)
            return "No relevant papers found."
        formatted = []
        for row in rows:
            # row is [id, text, metadata] based on columns=["id", "text", "metadata"]
            text = row[1] if len(row) > 1 else ""
            formatted.append(text)
        return json.dumps(formatted, ensure_ascii=False)

    return search_arxiv_papers


# Tool specification for OpenAI function calling format
ARXIV_TOOL_SPEC = {
    "type": "function",
    "function": {
        "name": "search_arxiv_papers",
        "description": (
            "Search a vector index of arXiv AI/ML research papers for "
            "background context "
            "on methods, techniques, or anomaly patterns relevant to the analysis. "
            "Use this after forming hypotheses to find supporting or "
            "contradicting evidence."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "Semantic search query (e.g. 'demand forecasting "
                        "anomaly detection e-commerce')"
                    ),
                },
                "num_results": {
                    "type": "integer",
                    "description": "Number of paper chunks to retrieve (default: 3)",
                    "default": 3,
                },
            },
            "required": ["query"],
        },
    },
}
