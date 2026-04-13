# Databricks notebook source
# MAGIC %md
# MAGIC # Lecture 3.6: Unity Catalog Functions as Agent Tools
# MAGIC
# MAGIC ## Topics Covered:
# MAGIC - Calling UC Functions from Python SDK
# MAGIC - Wrapping UC Functions as LangChain tools
# MAGIC - Integrating tools with an LLM agent
# MAGIC - Using session memory with tool calling

# COMMAND ----------

# MAGIC %pip install /Workspace/Users/aschelin@gmail.com/.bundle/llmops-databricks-course-nathalie-vishal-adriane/dev/files

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

import json
from typing import Any

from databricks.sdk import WorkspaceClient
from langchain_core.tools import StructuredTool
from loguru import logger
from pydantic import BaseModel, Field

from ordr_bhvr_rca_agent.config import ProjectConfig

cfg = ProjectConfig.from_yaml("../project_config.yml")
w = WorkspaceClient()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Discover Available UC Functions

# COMMAND ----------

# List all functions in our schema
functions = list(w.functions.list(catalog_name=cfg.catalog, schema_name=cfg.schema))

print(f"Found {len(functions)} functions in {cfg.catalog}.{cfg.schema}:\n")
for func in functions:
    print(f"- {func.name}")
    if func.comment:
        print(f"  {func.comment}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Call UC Functions Directly

# COMMAND ----------

# Get function metadata
function_name = f"{cfg.catalog}.{cfg.schema}.order_metrics_by_dimension"

try:
    function_info = w.functions.get(function_name)
    print(f"✓ Function: {function_name}")
    print(f"  Comment: {function_info.comment}")
    print("  Parameters:")
    for param in function_info.input_params.parameters:
        print(
            f"    - {param.name}: {param.type_name} ({param.comment or 'no description'})"
        )
except Exception as e:
    logger.warning(f"Function not found: {e}")
    logger.info("Run notebook 1.x to create order_metrics_by_dimension function")

# COMMAND ----------

# Call the function using SQL execution API
from databricks.sdk.service.sql import StatementParameterListItem

statement = f"""
SELECT {function_name}(
    start_date => :start_date,
    end_date => :end_date,
    time_bucket => :time_bucket,
    dimension => :dimension
)
"""

result = w.statement_execution.execute_statement(
    warehouse_id=cfg.warehouse_id,
    catalog=cfg.catalog,
    schema=cfg.schema,
    statement=statement,
    parameters=[
        StatementParameterListItem(name="start_date", value="2017-01-01"),
        StatementParameterListItem(name="end_date", value="2017-03-31"),
        StatementParameterListItem(name="time_bucket", value="month"),
        StatementParameterListItem(name="dimension", value="customer_state"),
    ],
    wait_timeout="30s",
)

# Parse results
if result.status and result.status.state == "SUCCEEDED":
    data = result.manifest.truncated if result.manifest.truncated else []
    logger.info(f"✓ Query succeeded, returned {len(data)} rows")

    # Display first few results
    for row in data[:3]:
        print(json.dumps(row, indent=2))
else:
    logger.error(f"Query failed: {result.status}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Wrap UC Functions as LangChain Tools

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Wrap UC Functions as LangChain Tools

# COMMAND ----------


# Define input schema for order_metrics tool
class OrderMetricsInput(BaseModel):
    """Input schema for order_metrics_by_dimension function."""

    start_date: str = Field(..., description="Start date in YYYY-MM-DD format")
    end_date: str = Field(..., description="End date in YYYY-MM-DD format")
    time_bucket: str = Field(
        ..., description="Time aggregation: 'day', 'week', 'month', or 'year'"
    )
    dimension: str = Field(
        ...,
        description="Dimension to group by: 'customer_state', 'product_category', etc.",
    )


def call_order_metrics(
    start_date: str, end_date: str, time_bucket: str, dimension: str
) -> dict[str, Any]:
    """Call order_metrics_by_dimension UC Function."""
    function_name = f"{cfg.catalog}.{cfg.schema}.order_metrics_by_dimension"

    statement = f"""
    SELECT {function_name}(
        start_date => :start_date,
        end_date => :end_date,
        time_bucket => :time_bucket,
        dimension => :dimension
    )
    """

    result = w.statement_execution.execute_statement(
        warehouse_id=cfg.warehouse_id,
        catalog=cfg.catalog,
        schema=cfg.schema,
        statement=statement,
        parameters=[
            StatementParameterListItem(name="start_date", value=start_date),
            StatementParameterListItem(name="end_date", value=end_date),
            StatementParameterListItem(name="time_bucket", value=time_bucket),
            StatementParameterListItem(name="dimension", value=dimension),
        ],
        wait_timeout="30s",
    )

    if result.status and result.status.state == "SUCCEEDED":
        data = result.manifest.truncated if result.manifest.truncated else []
        return {"success": True, "row_count": len(data), "data": data[:10]}
    else:
        return {"success": False, "error": str(result.status)}


# Create LangChain tool
order_metrics_tool = StructuredTool.from_function(
    func=call_order_metrics,
    name="order_metrics_by_dimension",
    description="Aggregate order metrics (count, GMV, AOV, customers) by dimension and time bucket. "
    "Use this to analyze order patterns across different dimensions like geography or product category.",
    args_schema=OrderMetricsInput,
)

print(f"✓ Created tool: {order_metrics_tool.name}")
print(f"  Description: {order_metrics_tool.description}")

# COMMAND ----------

# Test the tool directly
test_result = order_metrics_tool.invoke(
    {
        "start_date": "2017-01-01",
        "end_date": "2017-01-31",
        "time_bucket": "week",
        "dimension": "customer_state",
    }
)

logger.info(
    f"Tool result: {test_result['success']}, rows: {test_result.get('row_count', 0)}"
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Create More RCA Tools

# COMMAND ----------


# Attribution analysis tool
class AttributionInput(BaseModel):
    """Input schema for compute_order_attribution function."""

    baseline_start: str = Field(..., description="Baseline period start (YYYY-MM-DD)")
    baseline_end: str = Field(..., description="Baseline period end (YYYY-MM-DD)")
    comparison_start: str = Field(..., description="Comparison period start (YYYY-MM-DD)")
    comparison_end: str = Field(..., description="Comparison period end (YYYY-MM-DD)")
    dimension: str = Field(
        ..., description="Dimension for waterfall (e.g., 'product_category')"
    )


def call_attribution(
    baseline_start: str,
    baseline_end: str,
    comparison_start: str,
    comparison_end: str,
    dimension: str,
) -> dict[str, Any]:
    """Call compute_order_attribution UC Function."""
    function_name = f"{cfg.catalog}.{cfg.schema}.compute_order_attribution"

    statement = f"""
    SELECT {function_name}(
        baseline_start => :baseline_start,
        baseline_end => :baseline_end,
        comparison_start => :comparison_start,
        comparison_end => :comparison_end,
        dimension => :dimension
    )
    """

    result = w.statement_execution.execute_statement(
        warehouse_id=cfg.warehouse_id,
        catalog=cfg.catalog,
        schema=cfg.schema,
        statement=statement,
        parameters=[
            StatementParameterListItem(name="baseline_start", value=baseline_start),
            StatementParameterListItem(name="baseline_end", value=baseline_end),
            StatementParameterListItem(name="comparison_start", value=comparison_start),
            StatementParameterListItem(name="comparison_end", value=comparison_end),
            StatementParameterListItem(name="dimension", value=dimension),
        ],
        wait_timeout="30s",
    )

    if result.status and result.status.state == "SUCCEEDED":
        data = result.manifest.truncated if result.manifest.truncated else []
        return {"success": True, "attributions": data}
    else:
        return {"success": False, "error": str(result.status)}


attribution_tool = StructuredTool.from_function(
    func=call_attribution,
    name="compute_order_attribution",
    description="Decompose order metric changes into dimension-level contributions. "
    "Returns a waterfall showing which segments drove growth or decline.",
    args_schema=AttributionInput,
)

print(f"✓ Created tool: {attribution_tool.name}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Integrate with LLM Agent

# COMMAND ----------

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_community.chat_models import ChatDatabricks
from langchain_core.prompts import ChatPromptTemplate

# Create LLM
llm = ChatDatabricks(endpoint=cfg.llm_endpoint, max_tokens=2000)

# Define agent prompt
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are an RCA (Root Cause Analysis) assistant for e-commerce order data.

Available tools:
- order_metrics_by_dimension: Get order metrics aggregated by dimension
- compute_order_attribution: Decompose metric changes into contributions

When analyzing anomalies:
1. First get baseline metrics
2. Get comparison period metrics
3. Run attribution to find drivers
4. Explain findings clearly

Be concise and data-driven.""",
        ),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
)

# Create agent
tools = [order_metrics_tool, attribution_tool]
agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

print("✓ Agent created with tools:", [t.name for t in tools])

# COMMAND ----------

# Test the agent
response = agent_executor.invoke(
    {"input": "What were the order metrics for January 2017 by customer state?"}
)

print("\n" + "=" * 60)
print("AGENT RESPONSE:")
print("=" * 60)
print(response["output"])

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Add Session Memory

# COMMAND ----------

from uuid import uuid4

from databricks.sdk.service.postgres import PostgresAPI
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.runnables.history import RunnableWithMessageHistory

from ordr_bhvr_rca_agent.memory import LakebaseMemory


# Adapter: LakebaseMemory → LangChain ChatMessageHistory
class LakebaseChatHistory(BaseChatMessageHistory):
    """LangChain-compatible chat history backed by Lakebase."""

    def __init__(self, session_id: str, memory: LakebaseMemory):
        self.session_id = session_id
        self.memory = memory

    @property
    def messages(self) -> list[BaseMessage]:
        """Retrieve messages from Lakebase."""
        raw_messages = self.memory.load_messages(self.session_id)
        return [
            HumanMessage(content=msg["content"])
            if msg["role"] == "user"
            else AIMessage(content=msg["content"])
            for msg in raw_messages
        ]

    def add_message(self, message: BaseMessage) -> None:
        """Add a message to Lakebase."""
        role = "user" if isinstance(message, HumanMessage) else "assistant"
        self.memory.save_messages(
            self.session_id, [{"role": role, "content": message.content}]
        )

    def clear(self) -> None:
        """Clear message history (not implemented for Lakebase)."""
        pass


# COMMAND ----------

# Setup memory
pg_api = PostgresAPI(w.api_client)
project_id = cfg.lakebase_project_id

project = pg_api.get_project(name=f"projects/{project_id}")
default_branch = next(iter(pg_api.list_branches(parent=project.name)))
endpoint = next(iter(pg_api.list_endpoints(parent=default_branch.name)))
host = endpoint.status.hosts.host

memory = LakebaseMemory(host=host, instance_name=endpoint.name, pg_api=pg_api)

session_id = f"rca-session-{uuid4()}"
print(f"✓ Created session: {session_id}")

# COMMAND ----------


# Wrap agent with memory
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """Factory function for session history."""
    return LakebaseChatHistory(session_id=session_id, memory=memory)


agent_with_memory = RunnableWithMessageHistory(
    agent_executor,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history",
)

print("✓ Agent now has session memory")

# COMMAND ----------

# Test multi-turn conversation
response1 = agent_with_memory.invoke(
    {"input": "Get order metrics for January 2017 by product category"},
    config={"configurable": {"session_id": session_id}},
)

print("\n" + "=" * 60)
print("TURN 1:")
print("=" * 60)
print(response1["output"])

# COMMAND ----------

# Follow-up question (should remember context)
response2 = agent_with_memory.invoke(
    {"input": "Now compare those to February 2017. What changed?"},
    config={"configurable": {"session_id": session_id}},
)

print("\n" + "=" * 60)
print("TURN 2:")
print("=" * 60)
print(response2["output"])

# COMMAND ----------

# Verify memory persistence
loaded_history = memory.load_messages(session_id)
logger.info(f"✓ Session has {len(loaded_history)} messages stored in Lakebase")
for i, msg in enumerate(loaded_history, 1):
    print(f"{i}. [{msg['role']}] {msg['content'][:60]}...")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC ✅ **What we built:**
# MAGIC - UC Functions called via Python SDK (no Spark dependency)
# MAGIC - Tools wrapped for LangChain agent framework
# MAGIC - LLM agent with tool calling
# MAGIC - Session memory backed by Lakebase
# MAGIC - Multi-turn conversational RCA assistant
# MAGIC
# MAGIC **Next steps:**
# MAGIC - Add more UC Function tools (customer_flow_analysis, etc.)
# MAGIC - Integrate Vector Search for RAG
# MAGIC - Add hypothesis validation logic
# MAGIC - Deploy as Model Serving endpoint
