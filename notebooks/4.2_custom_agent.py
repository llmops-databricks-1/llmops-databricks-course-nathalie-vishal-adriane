# Databricks notebook source
# MAGIC %md
# MAGIC # Lecture 4.2: Custom RCA Agent with Tracing
# MAGIC
# MAGIC ## Topics Covered:
# MAGIC - Integrating tracing into the RCA agent
# MAGIC - Tracing LLM calls for hypothesis generation
# MAGIC - Tracing tool executions (Genie, news scraping, attribution)
# MAGIC - Session and request tracking for RCA workflows
# MAGIC - End-to-end RCA agent tracing
# MAGIC - Performance analysis


# COMMAND ----------

# MAGIC %pip install /Workspace/Users/aschelin@gmail.com/.bundle/llmops-databricks-course-nathalie-vishal-adriane/dev/files/ nest-asyncio
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import asyncio
import os

import mlflow
from databricks.sdk import WorkspaceClient
from dotenv import load_dotenv
from loguru import logger
from pyspark.sql import SparkSession

from ordr_bhvr_rca_agent.agent import SimpleAgent
from ordr_bhvr_rca_agent.config import get_env, load_config
from ordr_bhvr_rca_agent.mcp import create_mcp_tools

# COMMAND ----------

# Setup MLflow tracking
if "DATABRICKS_RUNTIME_VERSION" not in os.environ:
    load_dotenv()
    profile = os.environ["PROFILE"]
    mlflow.set_tracking_uri(f"databricks://{profile}")
    mlflow.set_registry_uri(f"databricks-uc://{profile}")


spark = SparkSession.builder.getOrCreate()

# Load configuration
env = get_env(spark)
cfg = load_config("../project_config.yml", env)

# Set experiment
mlflow.set_experiment(cfg.experiment_name)

w = WorkspaceClient()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. RCA Agent with Tracing - Architecture
# MAGIC
# MAGIC ```
# MAGIC User Request (e.g., "Compare March to May 2018 orders")
# MAGIC     ↓
# MAGIC ┌──────────────────────────────────────────────┐
# MAGIC │  @mlflow.trace(AGENT)                        │
# MAGIC │  predict()                                    │
# MAGIC │    ├─ Update trace metadata                   │
# MAGIC │    │  (session_id, request_id, anomaly_period)│
# MAGIC │    │                                           │
# MAGIC │    ├─ @mlflow.trace(RETRIEVER)                │
# MAGIC │    │  load_memory(session_id)                  │
# MAGIC │    │  → prepend past RCA conversations         │
# MAGIC │    │                                           │
# MAGIC │    ├─ call_and_run_tools()                    │
# MAGIC │    │    ├─ @mlflow.trace(TOOL)                │
# MAGIC │    │    │  query_genie() → order metrics       │
# MAGIC │    │    │                                      │
# MAGIC │    │    ├─ @mlflow.trace(CHAIN)               │
# MAGIC │    │    │  compute_attribution()               │
# MAGIC │    │    │                                      │
# MAGIC │    │    ├─ @mlflow.trace(LLM)                 │
# MAGIC │    │    │  generate_hypotheses()               │
# MAGIC │    │    │                                      │
# MAGIC │    │    ├─ @mlflow.trace(TOOL)                │
# MAGIC │    │    │  scrape_news()                       │
# MAGIC │    │    │                                      │
# MAGIC │    │    ├─ @mlflow.trace(LLM)                 │
# MAGIC │    │    │  generate_report()                   │
# MAGIC │    │    │                                      │
# MAGIC │    │    └─ Loop until analysis complete        │
# MAGIC │    │                                           │
# MAGIC │    └─ @mlflow.trace(CHAIN)                    │
# MAGIC │       save_memory(session_id, rca_results)     │
# MAGIC └──────────────────────────────────────────────┘
# MAGIC     ↓
# MAGIC RCA Report + Complete Trace
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. RCA Agent with Tracing
# MAGIC
# MAGIC The `SimpleAgent` class from `ordr_bhvr_rca_agent.agent` provides:
# MAGIC - Full MLflow tracing integration
# MAGIC - MCP tool support (Vector Search, Genie, UC Functions)
# MAGIC - Session and request tracking
# MAGIC - Automatic deployment metadata

# COMMAND ----------

# MAGIC %md
# MAGIC ### Agent Architecture (from agent.py):
# MAGIC
# MAGIC ```python
# MAGIC class SimpleAgent(ResponsesAgent):
# MAGIC     def __init__(self, llm_endpoint, system_prompt, catalog, schema, ...):
# MAGIC         # Automatically creates MCP tools from Genie and UC Functions
# MAGIC         ...
# MAGIC
# MAGIC     @mlflow.trace(span_type=SpanType.TOOL)
# MAGIC     def execute_tool(self, tool_name, args):
# MAGIC         # Traced tool execution (Genie, attribution, news scraping)
# MAGIC         ...
# MAGIC
# MAGIC     @mlflow.trace(span_type=SpanType.LLM)
# MAGIC     def call_llm(self, messages):
# MAGIC         # Traced LLM calls (hypothesis generation, report writing)
# MAGIC         ...
# MAGIC
# MAGIC     @mlflow.trace(span_type=SpanType.CHAIN)
# MAGIC     def call_and_run_tools(self, messages, ...):
# MAGIC         # Traced RCA loop (query → attribute → hypothesize → scrape → report)
# MAGIC         ...
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Create RCA Agent Instance
# MAGIC
# MAGIC The agent automatically gets MCP tools for:
# MAGIC - Genie (query order metrics with natural language)
# MAGIC - Vector Search (retrieve historical RCA reports)
# MAGIC - UC Functions (attribution computation, customer flow analysis)

# COMMAND ----------
# First, create MCP tools from Genie
host = w.config.host
mcp_urls = [f"{host}/api/2.0/mcp/genie/{cfg.genie_space_id}"]

logger.info(f"Loading MCP tools from Genie space: {cfg.genie_space_id}")

# Handle async event loop - use nest_asyncio to allow nested event loops
import nest_asyncio

nest_asyncio.apply()
mcp_tools = asyncio.run(create_mcp_tools(w, mcp_urls))

logger.info(f"✓ Loaded {len(mcp_tools)} tools from MCP servers")

# Create agent with tools
agent = SimpleAgent(
    llm_endpoint=cfg.llm_endpoint,
    system_prompt=cfg.system_prompt,
    tools=mcp_tools,
)

logger.info("✓ RCA Agent created with MCP tools:")
logger.info(f"  - Genie Space: {cfg.genie_space_id}")
logger.info(f"  - Catalog: {cfg.catalog}.{cfg.schema}")
logger.info(f"  - LLM Endpoint: {cfg.llm_endpoint}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Test the Traced RCA Agent

# COMMAND ----------

# Test the RCA agent
logger.info("Testing RCA Agent:")
logger.info("=" * 80)

# Call agent using the chat method
user_query = "Compare sales behavior from March 2018 to May 2018"
response = agent.chat(user_query)

logger.info(f"User: {user_query}")
logger.info(f"Agent: {response}")
logger.info("✓ Agent response completed! Check MLflow UI for traces.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Multi-Turn RCA Conversation with Tracing

# COMMAND ----------

# Test another query
user_query2 = "What happened to order volumes in May 2018?"
response2 = agent.chat(user_query2)

logger.info(f"User: {user_query2}")
logger.info(f"Agent: {response2}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Additional RCA Queries

# COMMAND ----------

# Test more RCA queries
logger.info("Testing additional RCA queries:")
logger.info("=" * 80)

# Query 1
query1 = "Which product categories were most affected in May 2018?"
response1 = agent.chat(query1)
logger.info(f"User: {query1}")
logger.info(f"Agent: {response1}")
logger.info("")

# Query 2
query2 = "What was the trend in Electronics category from March to May 2018?"
response2 = agent.chat(query2)
logger.info(f"User: {query2}")
logger.info(f"Agent: {response2}")

logger.info("✓ Multiple RCA queries completed!")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Analyzing RCA Agent Traces

# COMMAND ----------

# Search for recent traces
recent_traces_df = mlflow.search_traces(order_by=["timestamp_ms DESC"], max_results=5)

logger.info(f"Recent RCA Traces ({len(recent_traces_df)}):")
logger.info("=" * 80)

if len(recent_traces_df) > 0:
    logger.info(f"Available columns: {list(recent_traces_df.columns)}")

    # Select only scalar columns to avoid Arrow conversion errors
    simple_cols = []
    for col in recent_traces_df.columns:
        if col not in ["request", "response", "spans", "inputs", "outputs"]:
            simple_cols.append(col)

    if simple_cols:
        display(recent_traces_df[simple_cols].head())
    else:
        logger.info(str(recent_traces_df.info()))
else:
    logger.info("No traces found.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. RCA Performance Analysis

# COMMAND ----------

# Get recent RCA agent traces (returns DataFrame)
# Note: search_traces in MLflow 3.8.1 doesn't support experiment_names, so we search all traces
recent_traces_df = mlflow.search_traces(order_by=["timestamp_ms DESC"], max_results=20)

if len(recent_traces_df) > 0:
    logger.info("RCA Agent Performance Statistics:")
    logger.info("=" * 80)
    logger.info(f"Total traces: {len(recent_traces_df)}")

    # Calculate statistics if execution_time_ms column exists
    if "execution_time_ms" in recent_traces_df.columns:
        durations = recent_traces_df["execution_time_ms"].dropna()
        if len(durations) > 0:
            logger.info(f"Avg RCA duration: {durations.mean():.2f}ms")
            logger.info(f"Min RCA duration: {durations.min():.2f}ms")
            logger.info(f"Max RCA duration: {durations.max():.2f}ms")

    # Count by status if column exists
    if "status" in recent_traces_df.columns:
        logger.info("By Status:")
        status_counts = recent_traces_df["status"].value_counts()
        for status, count in status_counts.items():
            logger.info(f"  {status}: {count}")

    # Show sample of traces (select only simple columns to avoid Arrow conversion issues)
    logger.info("Sample Traces:")
    logger.info(f"Available columns: {list(recent_traces_df.columns)}")

    # Select only scalar columns for display
    simple_cols = []
    for col in recent_traces_df.columns:
        # Skip complex object columns that cause Arrow conversion errors
        if col not in ["request", "response", "spans", "inputs", "outputs"]:
            simple_cols.append(col)

    if simple_cols:
        display(recent_traces_df[simple_cols].head())
    else:
        # Fallback: just show the info
        logger.info(str(recent_traces_df.info()))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Detailed Trace Inspection

# COMMAND ----------

if len(recent_traces_df) > 0:
    # Get the most recent trace (first row)
    trace = recent_traces_df.iloc[0]

    print("Detailed RCA Trace Inspection:")
    print("=" * 80)
    print(f"Request ID: {trace.get('request_id', 'N/A')}")
    print(f"Trace ID: {trace.get('trace_id', 'N/A')}")
    print(f"Duration: {trace.get('execution_time_ms', 'N/A')}ms")
    print(f"Status: {trace.get('status', 'N/A')}")

    # Tags
    if "tags" in trace and trace["tags"]:
        print("\nTags:")
        for key, value in trace["tags"].items():
            print(f"  {key}: {value}")

    # Metadata
    if "request_metadata" in trace and trace["request_metadata"]:
        print("\nMetadata:")
        for key, value in trace["request_metadata"].items():
            print(f"  {key}: {value}")

    # Spans
    if "spans" in trace:
        spans_count = len(trace["spans"]) if trace["spans"] else 0
        print(f"\nSpans: {spans_count}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. Trace Filtering for RCA Analysis

# COMMAND ----------

# Filter traces by anomaly analysis request type
anomaly_traces = mlflow.search_traces(
    filter_string="tags.request_type = 'anomaly_analysis'", max_results=10
)
logger.info(f"Anomaly analysis traces: {len(anomaly_traces)}")

# Filter traces by specific time period
may2018_traces = mlflow.search_traces(
    filter_string="request_metadata.anomaly_period = 'May 2018'", max_results=10
)
logger.info(f"May 2018 RCA traces: {len(may2018_traces)}")

# Filter failed traces for debugging
failed_rca_traces = mlflow.search_traces(filter_string="status = 'ERROR'", max_results=5)
logger.info(f"Failed RCA traces: {len(failed_rca_traces)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 11. Best Practices for RCA Tracing

# COMMAND ----------

# MAGIC %md
# MAGIC ### RCA Tracing Best Practices:
# MAGIC
# MAGIC 1. **Always add session_id** for multi-turn RCA conversations
# MAGIC 2. **Use unique request_id** for each anomaly analysis request
# MAGIC 3. **Include git_sha** for RCA model version tracking
# MAGIC 4. **Trace all LLM calls** (hypothesis generation, report writing)
# MAGIC 5. **Trace all tool executions** (Genie queries, news scraping, attribution)
# MAGIC 6. **Use SpanType.CHAIN** for multi-step operations (attribution → hypothesize → report)
# MAGIC 7. **Add anomaly metadata** (period, metric, categories affected)
# MAGIC 8. **Set span attributes** for debugging (SQL queries, API responses)
# MAGIC 9. **Handle errors gracefully** in traces (failed Genie queries, timeouts)
# MAGIC 10. **Search traces** for RCA quality analysis and debugging
# MAGIC 11. **Track token usage** for cost optimization
# MAGIC 12. **Monitor attribution computation time** for performance tuning
# MAGIC
# MAGIC ### RCA-Specific Metadata:
# MAGIC - `anomaly_period`: Time range being analyzed (e.g., "March-May 2018")
# MAGIC - `metric`: Primary metric under investigation (e.g., "order_volume")
# MAGIC - `categories_affected`: Product categories with largest impact
# MAGIC - `attribution_method`: Waterfall, Shapley, etc.
# MAGIC - `hypothesis_count`: Number of hypotheses generated
# MAGIC - `external_context_sources`: News categories scraped (economic, supply_chain, etc.)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC In this notebook, you learned:
# MAGIC - How to use MLflow tracing with the RCA agent
# MAGIC - Different span types for RCA workflows (Genie queries, attribution, hypothesis generation)
# MAGIC - Adding RCA-specific metadata and tags
# MAGIC - Searching and analyzing RCA traces
# MAGIC - Performance analysis for RCA workflows
# MAGIC - Best practices for production RCA tracing
# MAGIC
# MAGIC **Next**: [4.3_evaluation_theory.py](4.3_evaluation_theory.py) - Evaluating RCA agent quality
