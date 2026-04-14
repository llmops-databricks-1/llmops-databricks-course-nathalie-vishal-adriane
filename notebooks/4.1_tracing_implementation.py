# Databricks notebook source
# MAGIC %md
# MAGIC # Lecture 4.1: MLflow Tracing Implementation for RCA Agent
# MAGIC
# MAGIC ## Topics Covered:
# MAGIC - What is tracing?
# MAGIC - Why tracing matters for GenAI
# MAGIC - Using @mlflow.trace decorator
# MAGIC - Manual span creation
# MAGIC - Adding metadata and tags
# MAGIC - Searching and analyzing traces
# MAGIC - Tracing RCA agent workflows

# COMMAND ----------

# MAGIC %pip install /Workspace/Users/aschelin@gmail.com/.bundle/llmops-databricks-course-nathalie-vishal-adriane/dev/files/
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import os
import random
from datetime import datetime

import mlflow
from dotenv import load_dotenv
from loguru import logger
from mlflow.entities import SpanType
from pyspark.sql import SparkSession

from ordr_bhvr_rca_agent.config import get_env, load_config

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

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. What is Tracing?
# MAGIC
# MAGIC **Tracing** captures the execution flow of your GenAI application.
# MAGIC
# MAGIC ### Why Tracing Matters for RCA:
# MAGIC
# MAGIC - **Observability**: See how the agent analyzes anomalies
# MAGIC - **Debugging**: Find where hypothesis generation fails
# MAGIC - **Performance**: Identify slow Genie queries or news scraping
# MAGIC - **Cost**: Track token usage in LLM calls
# MAGIC - **Quality**: Analyze attribution accuracy and hypothesis quality
# MAGIC
# MAGIC ### Trace Structure for RCA Agent:
# MAGIC
# MAGIC ```
# MAGIC Trace (Root)
# MAGIC ├── Span: RCA Agent Call
# MAGIC │   ├── Span: Genie Query (order metrics)
# MAGIC │   ├── Span: Attribution Computation
# MAGIC │   ├── Span: LLM Call (hypothesis generation)
# MAGIC │   ├── Span: News Scraping (external context)
# MAGIC │   └── Span: LLM Call (report generation)
# MAGIC └── Metadata: session_id, request_id, anomaly_period, etc.
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Simple Tracing with @mlflow.trace

# COMMAND ----------

# Set experiment
mlflow.set_experiment(cfg.experiment_name)


# Simple function with tracing
@mlflow.trace
def compute_metric_delta(baseline: float, current: float) -> dict:
    """Compute delta and percent change between two metrics."""
    delta = current - baseline
    pct_change = (delta / baseline * 100) if baseline != 0 else 0
    return {"delta": delta, "pct_change": pct_change}


# Call the function
result = compute_metric_delta(100.0, 85.0)
logger.info(f"Result: {result}")

logger.info("✓ Trace created! Check MLflow UI to see the trace.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Tracing with Span Types

# COMMAND ----------


@mlflow.trace(span_type=SpanType.LLM)
def call_llm(prompt: str) -> str:
    """Simulate an LLM call for hypothesis generation."""
    return f"Response to: {prompt}"


@mlflow.trace(span_type=SpanType.TOOL)
def query_genie(question: str) -> dict:
    """Simulate a Genie query for order metrics."""
    return {
        "sql": "SELECT * FROM orders WHERE date BETWEEN '2018-03-01' AND '2018-05-31'",
        "data": [{"month": "March", "orders": 7500}, {"month": "May", "orders": 6200}],
    }


@mlflow.trace(span_type=SpanType.CHAIN)
def analyze_anomaly(user_query: str) -> str:
    """Process an RCA query with Genie and LLM."""
    # Query Genie for data
    genie_results = query_genie(user_query)

    # Call LLM with results
    prompt = (
        f"User asked: {user_query}\nData: {genie_results['data']}\nGenerate hypotheses."
    )
    response = call_llm(prompt)

    return response


# Test the chain
output = analyze_anomaly("Compare sales behavior from March 2018 to May 2018")
logger.info(f"Output: {output}")

logger.info("✓ Multi-span trace created!")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Manual Span Creation

# COMMAND ----------


def compute_attribution(baseline_data: dict, current_data: dict) -> dict:
    """Compute attribution decomposition with manual span control."""

    with mlflow.start_span("compute_attribution") as span:
        # Set inputs
        span.set_inputs({"baseline": baseline_data, "current": current_data})

        # Step 1: Compute volume effect
        with mlflow.start_span("volume_effect") as step1:
            volume_delta = current_data["volume"] - baseline_data["volume"]
            volume_contribution = volume_delta * baseline_data["avg_price"]
            step1.set_outputs({"volume_contribution": volume_contribution})

        # Step 2: Compute mix effect
        with mlflow.start_span("mix_effect") as step2:
            price_delta = current_data["avg_price"] - baseline_data["avg_price"]
            mix_contribution = price_delta * current_data["volume"]
            step2.set_outputs({"mix_contribution": mix_contribution})

        # Calculate total and attribution percentages
        total_delta = volume_contribution + mix_contribution
        attribution = {
            "volume_contribution": volume_contribution,
            "mix_contribution": mix_contribution,
            "total_delta": total_delta,
            "volume_pct": (volume_contribution / total_delta * 100)
            if total_delta != 0
            else 0,
            "mix_pct": (mix_contribution / total_delta * 100) if total_delta != 0 else 0,
        }

        # Set final outputs
        span.set_outputs(attribution)

        return attribution


# Test
baseline = {"volume": 7500, "avg_price": 120.0}
current = {"volume": 6200, "avg_price": 115.0}
result = compute_attribution(baseline, current)
logger.info(f"Attribution: {result}")

logger.info("✓ Trace with nested spans created!")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Adding Metadata and Tags

# COMMAND ----------

# Generate trace identifiers
timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
session_id = f"s-{timestamp}-{random.randint(100000, 999999)}"
request_id = f"req-{timestamp}-{random.randint(100000, 999999)}"
git_sha = "abc123def456"


@mlflow.trace
def rca_with_metadata(anomaly_period: str, metric: str) -> dict:
    """RCA function with rich metadata."""

    # Update current trace with metadata
    mlflow.update_current_trace(
        metadata={
            "mlflow.trace.session": session_id,
            "user_id": "analyst_123",
            "environment": "production",
            "anomaly_period": anomaly_period,
            "metric": metric,
        },
        tags={
            "model_serving_endpoint_name": "rca-agent-endpoint",
            "model_version": "1",
            "git_sha": git_sha,
            "request_type": "anomaly_analysis",
        },
        client_request_id=request_id,
    )

    return {
        "anomaly_period": anomaly_period,
        "metric": metric,
        "analysis": "Volume declined 17.3%, driven primarily by decreased order count",
    }


# Test
result = rca_with_metadata("March-May 2018", "order_volume")
logger.info(f"Result: {result}")
logger.info("Trace metadata:")
logger.info(f"  Session ID: {session_id}")
logger.info(f"  Request ID: {request_id}")
logger.info(f"  Git SHA: {git_sha}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Searching Traces

# COMMAND ----------

# Search traces by git_sha
traces_df = mlflow.search_traces(
    filter_string=f"tags.git_sha = '{git_sha}'", max_results=5
)

logger.info(f"Found {len(traces_df)} traces with git_sha={git_sha}")

if len(traces_df) > 0:
    logger.info(f"Available columns: {list(traces_df.columns)}")
    logger.info("Trace Details:")
    logger.info("=" * 80)

    # Display the DataFrame - select only columns that exist
    cols_to_show = []
    for col in ["request_id", "timestamp_ms", "status", "tags"]:
        if col in traces_df.columns:
            cols_to_show.append(col)

    if cols_to_show:
        display(traces_df[cols_to_show].head())
    else:
        # Just show all columns if none of the expected ones exist
        display(traces_df.head())
else:
    logger.info("No traces found. Try running some traced functions first!")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Tracing Real LLM Calls

# COMMAND ----------

from databricks.sdk import WorkspaceClient
from openai import OpenAI

w = WorkspaceClient()

# Authenticate using Databricks SDK
host = w.config.host
token = w.tokens.create(lifetime_seconds=1200).token_value

# For Databricks serving endpoints
client = OpenAI(api_key=token, base_url=f"{host.rstrip('/')}/serving-endpoints")


@mlflow.trace(span_type=SpanType.LLM)
def generate_hypothesis(data_summary: str, model: str = None) -> str:
    """Call a real LLM to generate RCA hypotheses."""

    model = model or cfg.llm_endpoint

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "You are an RCA analyst. Generate hypotheses for order volume anomalies.",
            },
            {
                "role": "user",
                "content": f"Data: {data_summary}\n\nGenerate 3 hypotheses for the observed decline.",
            },
        ],
        max_tokens=200,
        temperature=0.7,
    )

    return response.choices[0].message.content


# Test with real LLM
data = "Orders declined 17% from March to May 2018. Top declining categories: Electronics (-25%), Home & Garden (-20%)."
result = generate_hypothesis(data)
logger.info(f"Hypotheses: {result}")

logger.info("✓ Real LLM call traced!")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Tracing RCA Agent Interactions

# COMMAND ----------


@mlflow.trace(span_type=SpanType.AGENT)
def rca_agent_interaction(user_message: str) -> dict:
    """Simulate a complete RCA agent interaction."""

    # Generate identifiers
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    session_id = f"s-{timestamp}-{random.randint(100000, 999999)}"
    request_id = f"req-{timestamp}-{random.randint(100000, 999999)}"

    # Add trace metadata
    mlflow.update_current_trace(
        metadata={
            "mlflow.trace.session": session_id,
        },
        tags={"agent_type": "rca_assistant", "model_version": "1.0"},
        client_request_id=request_id,
    )

    # Step 1: Query Genie for order data
    with mlflow.start_span("genie_query", span_type=SpanType.TOOL) as span:
        span.set_inputs({"query": user_message})
        genie_result = {
            "sql": "SELECT month, COUNT(*) FROM orders GROUP BY month",
            "data": [
                {"month": "March", "orders": 7500},
                {"month": "May", "orders": 6200},
            ],
        }
        span.set_outputs(genie_result)

    # Step 2: Compute attribution
    with mlflow.start_span("compute_attribution", span_type=SpanType.CHAIN) as span:
        span.set_inputs({"data": genie_result["data"]})
        attribution = {
            "volume_contribution": -1300,
            "mix_contribution": -200,
            "volume_pct": 86.7,
            "mix_pct": 13.3,
        }
        span.set_outputs(attribution)

    # Step 3: Generate hypotheses (LLM call)
    with mlflow.start_span("generate_hypotheses", span_type=SpanType.LLM) as span:
        span.set_inputs({"user_message": user_message, "attribution": attribution})
        hypotheses = "1. Seasonal decline in demand\n2. Supply chain disruptions\n3. Competitor campaigns"
        span.set_outputs({"hypotheses": hypotheses})

    # Step 4: Scrape news for context
    with mlflow.start_span("scrape_news", span_type=SpanType.TOOL) as span:
        span.set_inputs({"period": "March-May 2018"})
        news = [
            {"title": "Brazil economic slowdown continues", "category": "economic"},
            {"title": "Shipping delays affect e-commerce", "category": "supply_chain"},
        ]
        span.set_outputs({"news": news})

    return {
        "response": f"Analysis complete. Volume contribution: {attribution['volume_pct']:.1f}%\n\nHypotheses:\n{hypotheses}",
        "session_id": session_id,
        "request_id": request_id,
    }


# Test RCA agent interaction
result = rca_agent_interaction("Compare sales behavior from March 2018 to May 2018")
logger.info(f"Agent Response: {result['response']}")
logger.info(f"Session ID: {result['session_id']}")
logger.info(f"Request ID: {result['request_id']}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Analyzing Traces

# COMMAND ----------

# Get recent traces for this experiment
# Note: search_traces in MLflow 3.8.1 doesn't support experiment_names, so we search all traces
recent_traces_df = mlflow.search_traces(order_by=["timestamp_ms DESC"], max_results=10)

logger.info(f"Recent Traces ({len(recent_traces_df)}):")
logger.info("=" * 80)

if len(recent_traces_df) > 0:
    # Display available columns first
    logger.info(f"Available columns: {list(recent_traces_df.columns)}")

    # Display the DataFrame with columns that exist
    cols_to_show = []
    for col in ["request_id", "trace_id", "timestamp_ms", "execution_time_ms", "status"]:
        if col in recent_traces_df.columns:
            cols_to_show.append(col)

    if cols_to_show:
        display(recent_traces_df[cols_to_show].head(10))
    else:
        # Just show first few columns if none of our preferred ones exist
        display(recent_traces_df.head(10))
else:
    logger.info("No traces found.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. Trace Attributes

# COMMAND ----------

if len(recent_traces_df) > 0:
    # Get first row as a Series
    trace = recent_traces_df.iloc[0]

    logger.info("Trace Attributes:")
    logger.info("=" * 80)
    logger.info(f"Request ID: {trace.get('request_id', 'N/A')}")
    logger.info(f"Trace ID: {trace.get('trace_id', 'N/A')}")
    logger.info(f"Timestamp: {trace.get('timestamp_ms', 'N/A')}")
    logger.info(f"Execution Time: {trace.get('execution_time_ms', 'N/A')}ms")
    logger.info(f"Status: {trace.get('status', 'N/A')}")

    if "tags" in trace and trace["tags"]:
        logger.info("Tags:")
        for key, value in trace["tags"].items():
            logger.info(f"  {key}: {value}")

    if "request_metadata" in trace and trace["request_metadata"]:
        logger.info("Metadata:")
        for key, value in trace["request_metadata"].items():
            logger.info(f"  {key}: {value}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 11. Best Practices for RCA Tracing

# COMMAND ----------

# MAGIC %md
# MAGIC ### Do:
# MAGIC 1. **Use appropriate span types** (LLM for hypothesis generation, TOOL for Genie/news scraping, CHAIN for attribution)
# MAGIC 2. **Add session and request IDs** for tracking multi-turn RCA conversations
# MAGIC 3. **Include git_sha** for version tracking
# MAGIC 4. **Set inputs and outputs** for each span (especially attribution data, hypotheses)
# MAGIC 5. **Add meaningful tags** for filtering (anomaly_period, metric, categories)
# MAGIC 6. **Use nested spans** for complex attribution computations
# MAGIC 7. **Trace all LLM calls** for cost tracking (hypothesis generation, report writing)
# MAGIC 8. **Include error information** in traces (failed Genie queries, news scraping timeouts)
# MAGIC
# MAGIC ### Don't:
# MAGIC 1. Trace too granularly (performance overhead on attribution loops)
# MAGIC 2. Forget to add anomaly metadata (period, metric, categories)
# MAGIC 3. Skip tracing expensive operations (Genie queries, LLM calls)
# MAGIC 4. Ignore trace search capabilities for RCA debugging
# MAGIC 5. Store sensitive customer data in traces
# MAGIC 6. Create traces without business context

# COMMAND ----------

# MAGIC %md
# MAGIC ## 12. Trace Filtering Examples for RCA

# COMMAND ----------

# Filter by status
failed_traces = mlflow.search_traces(filter_string="status = 'ERROR'", max_results=5)
logger.info(f"Failed traces: {len(failed_traces)}")

# Filter by endpoint
endpoint_traces = mlflow.search_traces(
    filter_string="tags.model_serving_endpoint_name = 'rca-agent-endpoint'", max_results=5
)
logger.info(f"Traces for RCA endpoint: {len(endpoint_traces)}")

# Filter by request type
rca_traces = mlflow.search_traces(
    filter_string="tags.request_type = 'anomaly_analysis'", max_results=5
)
logger.info(f"Anomaly analysis traces: {len(rca_traces)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC In this notebook, you learned:
# MAGIC - How to use MLflow tracing with the RCA agent
# MAGIC - Different span types for RCA workflows (Genie, attribution, hypothesis generation)
# MAGIC - Adding metadata and tags for RCA-specific filtering
# MAGIC - Searching and analyzing traces
# MAGIC - Best practices for production RCA tracing
# MAGIC
# MAGIC **Next**: [4.2_custom_agent.py](4.2_custom_agent.py) - Building custom RCA agents
