# Databricks notebook source
# MAGIC %md
# MAGIC # Lecture 3.1: Custom Functions & Tools for Agents
# MAGIC
# MAGIC ## Topics Covered:
# MAGIC - What are agent tools?
# MAGIC - Creating custom functions
# MAGIC - Tool specifications (OpenAI format)
# MAGIC - Integrating tools with agents
# MAGIC - Vector search as a tool
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Understanding Agent Tools
# MAGIC
# MAGIC **Tools** are functions that agents can call to perform specific tasks.
# MAGIC
# MAGIC ### Why Tools?
# MAGIC
# MAGIC LLMs alone cannot:
# MAGIC - Access external data (databases, APIs)
# MAGIC - Perform calculations
# MAGIC - Execute code
# MAGIC - Search documents
# MAGIC
# MAGIC **Tools bridge this gap** by giving LLMs the ability to take actions.
# MAGIC
# MAGIC ### Tool Calling Flow:
# MAGIC
# MAGIC ```
# MAGIC User: "What papers discuss transformers?"
# MAGIC   ↓
# MAGIC Agent: Decides to use vector_search tool
# MAGIC   ↓
# MAGIC Tool: vector_search(query="transformers")
# MAGIC   ↓
# MAGIC Tool Result: [paper1, paper2, paper3]
# MAGIC   ↓
# MAGIC Agent: Synthesizes answer from results
# MAGIC   ↓
# MAGIC Response: "Here are papers about transformers..."
# MAGIC ```

# COMMAND ----------

# MAGIC %pip install /Workspace/Users/aschelin@gmail.com/.bundle/llmops-databricks-course-nathalie-vishal-adriane/dev/files

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

import json

from databricks.sdk import WorkspaceClient
from databricks.vector_search.client import VectorSearchClient
from loguru import logger
from pyspark.sql import SparkSession

from ordr_bhvr_rca_agent.agent import SimpleAgent
from ordr_bhvr_rca_agent.config import get_env, load_config
from ordr_bhvr_rca_agent.mcp import ToolInfo

# COMMAND ----------

spark = SparkSession.builder.getOrCreate()

# Load configuration
env = get_env(spark)
cfg = load_config("../project_config.yml", env)

w = WorkspaceClient()
vsc = VectorSearchClient(
    workspace_url=w.config.host,
    personal_access_token=w.tokens.create(lifetime_seconds=1200).token_value,
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Tool Specification Format
# MAGIC
# MAGIC Tools are defined using the **OpenAI function calling format**:
# MAGIC
# MAGIC ```json
# MAGIC {
# MAGIC   "type": "function",
# MAGIC   "function": {
# MAGIC     "name": "tool_name",
# MAGIC     "description": "What the tool does",
# MAGIC     "parameters": {
# MAGIC       "type": "object",
# MAGIC       "properties": {
# MAGIC         "param1": {
# MAGIC           "type": "string",
# MAGIC           "description": "Description of param1"
# MAGIC         }
# MAGIC       },
# MAGIC       "required": ["param1"]
# MAGIC     }
# MAGIC   }
# MAGIC }
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Creating a Simple metric_delta Tool

# COMMAND ----------


def metric_delta(baseline: float, comparison: float) -> dict:
    """Compute absolute and percentage change between two values."""
    delta = comparison - baseline
    pct_change = (delta / baseline) * 100 if baseline != 0 else float("inf")

    return {
        "baseline": baseline,
        "comparison": comparison,
        "delta": delta,
        "pct_change": pct_change,
    }


# Test the function
result = metric_delta(500, 300)
logger.info(f"Metric delta (baseline=5, comparison=3): {result}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Tool Specification for metric_delta

# COMMAND ----------

metric_delta_tool_spec = {
    "type": "function",
    "function": {
        "name": "metric_delta",
        "description": "Compute absolute and percentage change between two values. Returns baseline, comparison, delta, and pct_change.",
        "parameters": {
            "type": "object",
            "properties": {
                "baseline": {"type": "number", "description": "The baseline value"},
                "comparison": {"type": "number", "description": "The comparison value"},
            },
            "required": ["baseline", "comparison"],
        },
    },
}

logger.info("Metric Delta Tool Specification:")
logger.info(json.dumps(metric_delta_tool_spec, indent=2))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Creating a Vector Search Tool

# COMMAND ----------


# Helper function to parse vector search results
def parse_vector_search_results(results: dict) -> list[dict]:
    """Parse vector search results from array format to dict format.

    Args:
        results: Raw results from similarity_search()

    Returns:
        List of dictionaries with column names as keys
    """
    columns = [col["name"] for col in results.get("manifest", {}).get("columns", [])]
    data_array = results.get("result", {}).get("data_array", [])

    return [dict(zip(columns, row_data, strict=False)) for row_data in data_array]


# COMMAND ----------

# def search_papers(query: str, num_results: int = 5, year_filter: str = None) -> str:
#     """Search for relevant papers using vector search.

#     Args:
#         query: Search query
#         num_results: Number of results to return
#         year_filter: Optional year filter (e.g., "2024")

#     Returns:
#         JSON string with search results
#     """
#     index_name = f"{cfg.catalog}.{cfg.schema}.arxiv_index"
#     index = vsc.get_index(index_name=index_name)

#     # Build search parameters
#     search_params = {
#         "query_text": query,
#         "columns": ["text", "title", "arxiv_id", "authors", "year"],
#         "num_results": num_results,
#         "query_type": "hybrid"
#     }

#     # Add year filter if provided
#     if year_filter:
#         search_params["filters"] = {"year": year_filter}

#     # Perform search
#     results = index.similarity_search(**search_params)

#     # Format results using helper function
#     papers = []
#     for row in parse_vector_search_results(results):
#         papers.append({
#             "title": row.get("title", "N/A"),
#             "arxiv_id": row.get("arxiv_id", "N/A"),
#             "authors": str(row.get("authors", "N/A")),
#             "year": row.get("year", "N/A"),
#             "excerpt": row.get("text", "")[:200] + "..."
#         })

#     return json.dumps(papers, indent=2)

# # Test the function
# results = search_papers("machine learning", num_results=2)
# logger.info("Search Results:")
# logger.info(results)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Tool Specification for Vector Search

# COMMAND ----------

# search_papers_tool_spec = {
#     "type": "function",
#     "function": {
#         "name": "search_papers",
#         "description": "Search for academic papers using semantic search. Returns relevant papers with titles, authors, and excerpts.",
#         "parameters": {
#             "type": "object",
#             "properties": {
#                 "query": {
#                     "type": "string",
#                     "description": "The search query describing what papers to find"
#                 },
#                 "num_results": {
#                     "type": "integer",
#                     "description": "Number of results to return (default: 5)",
#                     "default": 5
#                 },
#                 "year_filter": {
#                     "type": "string",
#                     "description": "Optional year filter to limit results (e.g., '2024')"
#                 }
#             },
#             "required": ["query"]
#         }
#     }
# }

# logger.info("Search Papers Tool Specification:")
# logger.info(json.dumps(search_papers_tool_spec, indent=2))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Tool Information Class

# COMMAND ----------

# Using ToolInfo from arxiv_curator.mcp package
# This class represents a tool with name, spec, and execution function

# Create tool info objects
metric_delta_tool = ToolInfo(
    name="metric_delta", spec=metric_delta_tool_spec, exec_fn=metric_delta
)

# search_papers_tool = ToolInfo(
#     name="search_papers",
#     spec=search_papers_tool_spec,
#     exec_fn=search_papers
# )

logger.info("Available Tools:")
logger.info(f"1. {metric_delta_tool.name}")
# logger.info(f"2. {search_papers_tool.name}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Tool Registry Pattern

# COMMAND ----------


class ToolRegistry:
    """Registry for managing agent tools."""

    def __init__(self):
        self._tools: dict[str, ToolInfo] = {}

    def register(self, tool: ToolInfo) -> None:
        """Register a tool."""
        self._tools[tool.name] = tool
        logger.info(f"✓ Registered tool: {tool.name}")

    def get_tool(self, name: str) -> ToolInfo:
        """Get a tool by name."""
        if name not in self._tools:
            raise ValueError(f"Tool not found: {name}")
        return self._tools[name]

    def get_all_specs(self) -> list[dict]:
        """Get all tool specifications."""
        return [tool.spec for tool in self._tools.values()]

    def execute(self, name: str, args: dict) -> object:
        """Execute a tool with arguments."""
        tool = self.get_tool(name)
        return tool.exec_fn(**args)

    def list_tools(self) -> list[str]:
        """List all registered tool names."""
        return list(self._tools.keys())

    def get_all_tools(self) -> list[ToolInfo]:
        """Get all tools as a list."""
        return list(self._tools.values())


# Create registry and register tools
registry = ToolRegistry()
registry.register(metric_delta_tool)
# registry.register(search_papers_tool)

logger.info(f"Total tools registered: {len(registry.list_tools())}")
logger.info(f"Tools: {registry.list_tools()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Executing Tools

# COMMAND ----------

# Execute metric_delta tool
calc_result = registry.execute("metric_delta", {"baseline": 1000, "comparison": 1005})
logger.info(f"metric_delta result: {calc_result}")

# # Execute search tool
# search_result = registry.execute("search_papers", {
#     "query": "neural networks",
#     "num_results": 3
# })
# logger.info(f"Search result:\n{search_result}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Best Practices for Tool Design
# MAGIC
# MAGIC ### Do:
# MAGIC 1. **Clear descriptions**: Help the LLM understand when to use the tool
# MAGIC 2. **Type hints**: Use proper Python type hints
# MAGIC 3. **Error handling**: Handle errors gracefully
# MAGIC 4. **Return structured data**: JSON or clear text format
# MAGIC 6. **Validate inputs**: Check parameters before execution
# MAGIC 7. **Document parameters**: Clear parameter descriptions
# MAGIC
# MAGIC ### Don't:
# MAGIC 1. Create tools that are too complex
# MAGIC 2. Return unstructured or ambiguous data
# MAGIC 3. Forget error handling
# MAGIC 4. Make tools that take too long to execute
# MAGIC 5. Overlap tool functionality
# MAGIC 6. Use unclear tool names

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Tool Design Patterns

# COMMAND ----------

# MAGIC %md
# MAGIC ### Pattern 1: Data Retrieval Tool
# MAGIC ```python
# MAGIC def get_data(query: str) -> str:
# MAGIC     # Fetch data from database/API
# MAGIC     # Format and return
# MAGIC     pass
# MAGIC ```
# MAGIC
# MAGIC ### Pattern 2: Computation Tool
# MAGIC ```python
# MAGIC def calculate(formula: str, values: dict) -> float:
# MAGIC     # Perform calculation
# MAGIC     # Return result
# MAGIC     pass
# MAGIC ```
# MAGIC
# MAGIC ### Pattern 3: Action Tool
# MAGIC ```python
# MAGIC def send_notification(message: str, recipient: str) -> str:
# MAGIC     # Perform action
# MAGIC     # Return confirmation
# MAGIC     pass
# MAGIC ```
# MAGIC
# MAGIC ### Pattern 4: Analysis Tool
# MAGIC ```python
# MAGIC def analyze_data(data: list, metric: str) -> dict:
# MAGIC     # Analyze data
# MAGIC     # Return insights
# MAGIC     pass
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. Testing Tools

# COMMAND ----------


def test_tool(tool_name: str, test_cases: list[dict]) -> None:
    """Test a tool with multiple test cases."""
    logger.info(f"Testing tool: {tool_name}")
    logger.info("=" * 80)

    for i, test_case in enumerate(test_cases, 1):
        logger.info(f"Test Case {i}:")
        logger.info(f"  Input: {test_case}")

        try:
            result = registry.execute(tool_name, test_case)
            logger.info("  ✓ Success")
            logger.info(f"  Result: {str(result)[:100]}...")
        except Exception as e:
            logger.error(f"  ✗ Error: {e}")


# Test metric_delta
test_tool(
    "metric_delta",
    [
        {"baseline": 5, "comparison": 3},
        {"baseline": 100, "comparison": 150},
        {"baseline": 0, "comparison": 10},
    ],
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 11. Using Tools with an Agent
# MAGIC
# MAGIC Now let's create a simple agent that can call our tools.
# MAGIC
# MAGIC **Note:** We're using the `SimpleAgent` class from the `ordr_bhvr_rca_agent` package.
# MAGIC This agent:
# MAGIC - Takes an LLM endpoint, system prompt, and list of tools
# MAGIC - Maintains conversation context
# MAGIC - Handles the tool-calling loop automatically
# MAGIC - Executes tools and passes results back to the LLM

# COMMAND ----------

from openai import OpenAI

# The SimpleAgent class is now imported from the package:
# - Manages tool-calling loop
# - Handles conversation context
# - Executes tools and returns results
# See src/ordr_bhvr_rca_agent/agent.py for implementation

# Create agent with our tools
agent = SimpleAgent(
    llm_endpoint=cfg.llm_endpoint,
    system_prompt="You are a helpful assistant. Use the available tools to answer questions.",
    tools=[metric_delta_tool],  # , search_papers_tool
)

# agent = SimpleAgent(
# llm_endpoint=cfg.llm_endpoint,
# system_prompt="You are a helpful assistant. Use the available tools to answer questions.",
# tools=registry.get_all_tools())

logger.info("✓ Agent created with tools:")
for tool_name in agent._tools_dict:
    logger.info(f"  - {tool_name}")

# COMMAND ----------

# Test agent with metric_delta
logger.info("Testing agent with metric_delta:")
logger.info("=" * 80)

response = agent.chat(
    "- If last year's sales were 1000 and this year's are 1100, what is the delta?"
)
logger.info(f"Agent response: {response}")

# COMMAND ----------

# # Test agent with search tool
# logger.info("Testing agent with search tool:")
# logger.info("=" * 80)

# response = agent.chat("Find papers about attention mechanisms")
# logger.info(f"Agent response: {response}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 12. Real-World Example: Sales Order RCA Agent
# MAGIC
# MAGIC Now let's build a complete real-world agent: **Root Cause Analysis (RCA) for Sales Anomalies**
# MAGIC
# MAGIC This agent will:
# MAGIC 1. Query sales data using **Databricks Genie** (via REST API)
# MAGIC 2. Generate hypotheses about anomalies using an LLM
# MAGIC 3. Scrape news for external context
# MAGIC 4. Produce a comprehensive RCA report
# MAGIC
# MAGIC **Learning Goals:**
# MAGIC - Build tools that integrate external APIs (Genie REST API)
# MAGIC - Create tools that call LLMs within tools
# MAGIC - Orchestrate complex multi-step workflows
# MAGIC - See how custom tools work in production scenarios
# MAGIC
# MAGIC **Note:** In notebook 3.2, we'll see how MCP simplifies the Genie integration!

# COMMAND ----------

# DBTITLE 1,RCA Agent Configuration
import re
import time
import urllib.parse

import feedparser
import requests

# Configuration - using values from project config
RCA_LLM_ENDPOINT = cfg.llm_endpoint
GENIE_SPACE_ID = cfg.genie_space_id  # Using your existing Genie space

# Databricks auth
DATABRICKS_HOST = (
    dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().get()  # noqa: F821
)
DATABRICKS_TOKEN = (
    dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()  # noqa: F821
)

RCA_HEADERS = {
    "Authorization": f"Bearer {DATABRICKS_TOKEN}",
    "Content-Type": "application/json",
}

# Create OpenAI client for RCA (same as we use elsewhere)
rca_llm_client = OpenAI(
    api_key=DATABRICKS_TOKEN, base_url=f"{DATABRICKS_HOST}/serving-endpoints"
)

logger.info("RCA Agent configured:")
logger.info(f"  Host: {DATABRICKS_HOST}")
logger.info(f"  Genie Space: {GENIE_SPACE_ID}")
logger.info(f"  LLM Endpoint: {RCA_LLM_ENDPOINT}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Tool 1: Genie Query Tool (REST API)
# MAGIC
# MAGIC This tool queries Databricks Genie using the REST API.
# MAGIC
# MAGIC **Key Learning:** How to integrate with Databricks APIs in custom tools.

# COMMAND ----------


def genie_query(question: str, space_id: str = None, timeout: int = 120) -> str:
    """Query Databricks Genie space for sales data analysis using REST API.

    Args:
        question: Natural language question to ask Genie
        space_id: Optional Genie space ID (defaults to configured space)
        timeout: Query timeout in seconds

    Returns:
        JSON string with query results including text response, SQL, and data
    """
    space_id = space_id or GENIE_SPACE_ID
    base = f"{DATABRICKS_HOST}/api/2.0/genie/spaces/{space_id}"

    # 1. Start conversation
    resp = requests.post(
        f"{base}/start-conversation",
        headers=RCA_HEADERS,
        json={"content": question},
    )
    resp.raise_for_status()
    payload = resp.json()
    conv_id = payload["conversation_id"]
    msg_id = payload["message_id"]

    # 2. Poll until completion
    poll_url = f"{base}/conversations/{conv_id}/messages/{msg_id}"
    deadline = time.time() + timeout
    status = "SUBMITTED"
    message = {}

    while time.time() < deadline:
        r = requests.get(poll_url, headers=RCA_HEADERS)
        r.raise_for_status()
        message = r.json()
        status = message.get("status", "UNKNOWN")
        if status in ("COMPLETED", "FAILED"):
            break
        time.sleep(3)

    if status == "FAILED":
        return json.dumps(
            {
                "error": message.get("error", "Genie query failed"),
                "sql": None,
                "data": None,
                "text": None,
            }
        )

    # 3. Extract results
    attachments = message.get("attachments", []) or []
    response_text = None
    sql_query = None
    query_data = None

    for att in attachments:
        if att.get("text") and att["text"].get("content"):
            response_text = att["text"]["content"]
        if att.get("query"):
            sql_query = att["query"].get("query", "")
            att_id = att.get("attachment_id")
            if att_id:
                result_url = f"{base}/conversations/{conv_id}/messages/{msg_id}/attachments/{att_id}/query-result"
                rr = requests.get(result_url, headers=RCA_HEADERS)
                if rr.status_code == 200:
                    raw = rr.json()
                    stmt = raw.get("statement_response", {})
                    columns = (
                        stmt.get("manifest", {}).get("schema", {}).get("columns", [])
                    )
                    data_array = stmt.get("result", {}).get("data_array", [])
                    query_data = {
                        "columns": columns,
                        "data_array": data_array,
                    }

    return json.dumps(
        {"text": response_text, "sql": sql_query, "data": query_data, "error": None},
        indent=2,
    )


# Test the function
logger.info("Testing genie_query tool...")
test_result = genie_query("Show me a sample of the data")
logger.info(f"Genie query result (first 200 chars): {test_result[:200]}...")

# COMMAND ----------

# Tool specification for genie_query
genie_query_tool_spec = {
    "type": "function",
    "function": {
        "name": "genie_query",
        "description": "Query Databricks Genie space for data analysis using natural language. Returns SQL query, data results, and natural language response.",
        "parameters": {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "Natural language question about the data (e.g., 'What were sales by region in Q1?')",
                },
                "space_id": {
                    "type": "string",
                    "description": "Optional Genie space ID. Uses default if not provided.",
                },
                "timeout": {
                    "type": "integer",
                    "description": "Query timeout in seconds (default: 120)",
                    "default": 120,
                },
            },
            "required": ["question"],
        },
    },
}

# COMMAND ----------

# MAGIC %md
# MAGIC ### Tool 2: News Scraping Tool
# MAGIC
# MAGIC Scrapes Google News for external context that might explain sales anomalies.
# MAGIC
# MAGIC **Key Learning:** Tools can integrate external data sources.

# COMMAND ----------

# News categories and keywords for RCA
RCA_KEYWORDS = {
    "economic": [
        "inflation",
        "recession",
        "interest rate hike",
        "currency devaluation",
        "economic crisis",
    ],
    "supply_chain": [
        "supply chain disruption",
        "shipping delay",
        "port congestion",
        "raw material shortage",
        "logistics crisis",
    ],
    "labor": [
        "strike",
        "labor strike",
        "worker protest",
        "union strike",
        "workforce shortage",
    ],
    "regulatory": [
        "trade tariff",
        "import ban",
        "sanctions",
        "new regulation",
        "tax increase",
    ],
    "natural": ["flood", "drought", "hurricane", "earthquake", "wildfire"],
}


def scrape_news(
    search_term: str, year: int, month: int, categories: list = None, top_n: int = 2
) -> str:
    """Scrape Google News for headlines related to potential sales anomaly causes.

    Args:
        search_term: Main search term (e.g., "retail sales", "consumer behavior")
        year: Year to search
        month: Month to search (1-12)
        categories: List of RCA categories (economic, supply_chain, labor, regulatory, natural)
        top_n: Number of articles to retrieve per keyword

    Returns:
        JSON string with news articles grouped by category
    """
    start_date = f"{year}-{month:02d}-01"
    end_date = f"{year}-{month:02d}-30"
    selected = categories or list(RCA_KEYWORDS.keys())
    results = {}

    for cat in selected:
        cat_results = []
        for kw in RCA_KEYWORDS.get(cat, []):
            query = f"{search_term} {kw} after:{start_date} before:{end_date}"
            encoded = urllib.parse.quote(query)
            rss_url = (
                f"https://news.google.com/rss/search?q={encoded}&hl=en&gl=US&ceid=US:en"
            )

            try:
                feed = feedparser.parse(rss_url)
                cat_results.extend(
                    [
                        {
                            "keyword": kw,
                            "title": e.title,
                            "link": e.link,
                            "published": e.get("published", ""),
                        }
                        for e in feed.entries[:top_n]
                    ]
                )
            except Exception as e:
                logger.warning(f"Error scraping news for {kw}: {e}")

        if cat_results:
            results[cat] = cat_results

    return json.dumps(results, indent=2)


# Test the function
logger.info("Testing scrape_news tool...")
news_test = scrape_news("retail sales", 2018, 3, categories=["economic"], top_n=1)
logger.info(f"News scraping result (first 200 chars): {news_test[:200]}...")

# COMMAND ----------

# Tool specification for scrape_news
scrape_news_tool_spec = {
    "type": "function",
    "function": {
        "name": "scrape_news",
        "description": "Scrape Google News for headlines that may explain sales anomalies across economic, supply chain, labor, regulatory, and natural event categories.",
        "parameters": {
            "type": "object",
            "properties": {
                "search_term": {
                    "type": "string",
                    "description": "Main search term (e.g., 'retail sales', 'e-commerce')",
                },
                "year": {
                    "type": "integer",
                    "description": "Year to search for news (e.g., 2018)",
                },
                "month": {"type": "integer", "description": "Month to search (1-12)"},
                "categories": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional categories: economic, supply_chain, labor, regulatory, natural",
                },
                "top_n": {
                    "type": "integer",
                    "description": "Articles per keyword (default: 2)",
                    "default": 2,
                },
            },
            "required": ["search_term", "year", "month"],
        },
    },
}

# COMMAND ----------

# MAGIC %md
# MAGIC ### Tool 3: Hypothesis Generation Tool
# MAGIC
# MAGIC Uses an LLM to analyze data and generate RCA hypotheses.
# MAGIC
# MAGIC **Key Learning:** Tools can call LLMs to perform complex reasoning.

# COMMAND ----------


def generate_rca_hypotheses(question: str, data_summary: str) -> str:
    """Generate RCA hypotheses from sales data analysis.

    Args:
        question: Original user question about sales anomaly
        data_summary: Summary of data retrieved from Genie

    Returns:
        JSON string with hypotheses and suggested search terms
    """
    system_prompt = """You are a senior business analyst specializing in root cause analysis
for sales and e-commerce anomalies. Given data from a sales database, generate:
1. A list of hypotheses that could explain the observed patterns
2. For each hypothesis, suggest a search term to look for confirming news/events
3. Identify the relevant time periods (year, month) to search

Return a JSON object with this structure:
{
  "data_insights": "Brief summary of what the data shows",
  "hypotheses": [
    {
      "hypothesis": "Description of the hypothesis",
      "search_term": "keyword to search for in news",
      "year": 2018,
      "month": 5,
      "categories": ["economic", "supply_chain"]
    }
  ]
}
Return ONLY valid JSON."""

    user_prompt = (
        f"User question: {question}\n\nData from sales database:\n{data_summary}"
    )

    response = rca_llm_client.chat.completions.create(
        model=RCA_LLM_ENDPOINT,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=4096,
        temperature=0.3,
    )

    raw = response.choices[0].message.content

    # Try to extract JSON
    try:
        result = json.loads(raw)
        return json.dumps(result, indent=2)
    except json.JSONDecodeError:
        # Try to find JSON in the response
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            return match.group()
        return json.dumps({"data_insights": raw, "hypotheses": []})


# Test the function
logger.info("Testing generate_rca_hypotheses tool...")
test_data = "Sales dropped 30% in May compared to March. All product categories affected."
hyp_test = generate_rca_hypotheses("Why did sales drop?", test_data)
logger.info(f"Generated hypotheses (first 200 chars): {hyp_test[:200]}...")

# COMMAND ----------

# Tool specification for generate_rca_hypotheses
generate_rca_hypotheses_tool_spec = {
    "type": "function",
    "function": {
        "name": "generate_rca_hypotheses",
        "description": "Analyze sales data and generate root cause analysis hypotheses with suggested news search terms and time periods.",
        "parameters": {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "Original user question about the sales anomaly",
                },
                "data_summary": {
                    "type": "string",
                    "description": "Summary of sales data showing the anomaly or pattern",
                },
            },
            "required": ["question", "data_summary"],
        },
    },
}

# COMMAND ----------

# MAGIC %md
# MAGIC ### Tool 4: RCA Report Generation Tool
# MAGIC
# MAGIC Combines all evidence into a comprehensive report.
# MAGIC
# MAGIC **Key Learning:** Tools can perform synthesis and generate structured outputs.

# COMMAND ----------


def generate_rca_report(
    question: str, data_summary: str, hypotheses: str, news_summary: str
) -> str:
    """Generate a comprehensive RCA report combining data, hypotheses, and external evidence.

    Args:
        question: Original user question
        data_summary: Sales data analysis
        hypotheses: Generated hypotheses (JSON string)
        news_summary: External news evidence (JSON string)

    Returns:
        Markdown-formatted RCA report
    """
    system_prompt = """You are a senior business analyst writing a Root Cause Analysis report.
Combine the sales data analysis, hypotheses, and external news evidence into a clear,
actionable report. Structure it as:

1. **Executive Summary** - What was the question and key finding
2. **Data Analysis** - What the numbers show (with specific figures)
3. **Hypotheses & Evidence** - Each hypothesis with supporting/contradicting news
4. **Most Likely Root Causes** - Ranked by evidence strength
5. **Recommended Actions** - What the business should do next

Be specific, cite data points, and clearly distinguish between confirmed and speculative causes."""

    user_prompt = f"""## Original Question
{question}

## Sales Data Analysis
{data_summary}

## Hypotheses
{hypotheses}

## External News & Events
{news_summary}"""

    response = rca_llm_client.chat.completions.create(
        model=RCA_LLM_ENDPOINT,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=6000,
        temperature=0.3,
    )

    return response.choices[0].message.content


# Test the function
logger.info("Testing generate_rca_report tool...")
test_report = generate_rca_report(
    "Why did sales drop?",
    "Sales down 30%",
    '{"hypotheses": [{"hypothesis": "Economic downturn"}]}',
    '{"economic": [{"title": "Fed raises interest rates"}]}',
)
logger.info(f"Generated report (first 200 chars): {test_report[:200]}...")

# COMMAND ----------

# Tool specification for generate_rca_report
generate_rca_report_tool_spec = {
    "type": "function",
    "function": {
        "name": "generate_rca_report",
        "description": "Generate a comprehensive Root Cause Analysis report combining sales data, hypotheses, and external news evidence into an actionable business report.",
        "parameters": {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "Original user question about the sales anomaly",
                },
                "data_summary": {
                    "type": "string",
                    "description": "Summary of sales data analysis",
                },
                "hypotheses": {
                    "type": "string",
                    "description": "JSON string of generated hypotheses",
                },
                "news_summary": {
                    "type": "string",
                    "description": "JSON string of scraped news articles",
                },
            },
            "required": ["question", "data_summary", "hypotheses", "news_summary"],
        },
    },
}

# COMMAND ----------

# MAGIC %md
# MAGIC ### Register RCA Tools

# COMMAND ----------

# Create ToolInfo objects for RCA tools
genie_query_tool = ToolInfo(
    name="genie_query", spec=genie_query_tool_spec, exec_fn=genie_query
)

scrape_news_tool = ToolInfo(
    name="scrape_news", spec=scrape_news_tool_spec, exec_fn=scrape_news
)

generate_rca_hypotheses_tool = ToolInfo(
    name="generate_rca_hypotheses",
    spec=generate_rca_hypotheses_tool_spec,
    exec_fn=generate_rca_hypotheses,
)

generate_rca_report_tool = ToolInfo(
    name="generate_rca_report",
    spec=generate_rca_report_tool_spec,
    exec_fn=generate_rca_report,
)

# Register all RCA tools
rca_registry = ToolRegistry()
rca_registry.register(genie_query_tool)
rca_registry.register(scrape_news_tool)
rca_registry.register(generate_rca_hypotheses_tool)
rca_registry.register(generate_rca_report_tool)

logger.info(f"RCA Tools registered: {rca_registry.list_tools()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Create RCA Agent

# COMMAND ----------

# Create an agent with all RCA tools
rca_agent = SimpleAgent(
    llm_endpoint=RCA_LLM_ENDPOINT,
    system_prompt="""You are a sales analytics assistant specializing in root cause analysis.

Use the available tools to investigate sales anomalies:
1. genie_query - Query sales data
2. scrape_news - Find external events
3. generate_rca_hypotheses - Generate hypotheses from data
4. generate_rca_report - Create final report

Workflow:
1. Query Genie for relevant sales data
2. Analyze the data and generate hypotheses
3. Search for news that might confirm hypotheses
4. Generate comprehensive RCA report

Be systematic and thorough in your analysis.""",
    tools=rca_registry.get_all_tools(),
    workspace_client=w,
)

logger.info("RCA Agent created with tools:")
for tool in rca_registry.list_tools():
    logger.info(f"  - {tool}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 13. Testing the RCA Agent

# COMMAND ----------

# Test individual RCA tools
logger.info("Testing RCA Tools individually:")
logger.info("=" * 80)

# Test 1: Genie query
logger.info("\nTest 1: Genie Query")
try:
    genie_result = rca_registry.execute(
        "genie_query", {"question": "Show me a summary of the data"}
    )
    logger.info(f"Result (first 150 chars): {genie_result[:150]}...")
except Exception as e:
    logger.error(f"Error: {e}")

# COMMAND ----------

# Test 2: Hypothesis generation
logger.info("\nTest 2: Hypothesis Generation")
try:
    hyp_result = rca_registry.execute(
        "generate_rca_hypotheses",
        {
            "question": "Why might sales fluctuate?",
            "data_summary": "Sales data shows seasonal patterns and some anomalies.",
        },
    )
    logger.info(f"Result (first 150 chars): {hyp_result[:150]}...")
except Exception as e:
    logger.error(f"Error: {e}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 14. End-to-End RCA Workflow
# MAGIC
# MAGIC **Note:** For a complete demo, you would ask a specific business question.
# MAGIC The agent will orchestrate all tools to produce a full RCA report.

# COMMAND ----------

# Example RCA question (customize based on your data)
logger.info("RCA Agent Example:")
logger.info("=" * 80)
logger.info("\nFor a full demo, ask a question like:")
logger.info("'Compare sales behaviour from March 2018 to May 2018'")
logger.info("\nThe agent will:")
logger.info("1. Query your Genie space for sales data")
logger.info("2. Generate hypotheses about observed patterns")
logger.info("3. Search for relevant news/events")
logger.info("4. Produce a comprehensive RCA report")

# Uncomment to run (replace with your actual question):
# rca_question = "Compare sales behaviour from March 2018 to May 2018"
# rca_response = rca_agent.chat(rca_question, max_iterations=15)
# logger.info("\n" + "=" * 80)
# logger.info("RCA REPORT:")
# logger.info("=" * 80)
# logger.info(rca_response)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 15. Key Takeaways: RCA Agent with Custom Tools
# MAGIC
# MAGIC ### What We Built:
# MAGIC
# MAGIC 1. **4 Custom Tools**
# MAGIC    - `genie_query`: REST API integration with Databricks Genie
# MAGIC    - `scrape_news`: External data source integration
# MAGIC    - `generate_rca_hypotheses`: LLM-powered analysis
# MAGIC    - `generate_rca_report`: Multi-source synthesis
# MAGIC
# MAGIC 2. **Complex Orchestration**
# MAGIC    - Tools calling other services (Genie REST API)
# MAGIC    - Tools calling LLMs (hypothesis generation, reporting)
# MAGIC    - Tools scraping external data (Google News)
# MAGIC    - Agent coordinating multi-step workflow
# MAGIC
# MAGIC 3. **REST API Integration**
# MAGIC    - Manual API calls to Genie
# MAGIC    - Polling for async operations
# MAGIC    - Response parsing and error handling
# MAGIC    - **This required substantial code!**
# MAGIC
# MAGIC ### Challenges with REST API Approach:
# MAGIC
# MAGIC - ⚠️ Manual request/response handling
# MAGIC - ⚠️ Polling logic for async operations
# MAGIC - ⚠️ Response format parsing
# MAGIC - ⚠️ Error handling
# MAGIC - ⚠️ Authentication management
# MAGIC
# MAGIC ### Coming Up in Notebook 3.2:
# MAGIC
# MAGIC **See how MCP simplifies Genie integration!**
# MAGIC - No REST API code needed
# MAGIC - No polling logic
# MAGIC - Automatic response parsing
# MAGIC - Built-in error handling
# MAGIC - Managed authentication
# MAGIC
# MAGIC The same RCA workflow, but much simpler!
