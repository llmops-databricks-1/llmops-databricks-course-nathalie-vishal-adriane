# Databricks notebook source
# COMMAND ----------
# MAGIC %pip install /Workspace/Users/nathalie-frisch@gmx.de/.bundle/llmops-databricks-course-nathalie-vishal-adriane/dev/artifacts/.internal/llmops_databricks_course_nathalie_vishal_adriane-0.0.1-py3-none-any.whl --force-reinstall --quiet

# COMMAND ----------
dbutils.library.restartPython()  # noqa: F821

# COMMAND ----------

import asyncio
import os
import random
import tempfile
from datetime import datetime

import mlflow
from databricks.sdk import WorkspaceClient
from loguru import logger
from mlflow.models import infer_signature
from mlflow.models.resources import (
    DatabricksGenieSpace,
    DatabricksServingEndpoint,
    DatabricksSQLWarehouse,
    DatabricksTable,
    DatabricksVectorSearchIndex,
)

from ordr_bhvr_rca_agent.agent import SimpleAgent
from ordr_bhvr_rca_agent.config import ProjectConfig
from ordr_bhvr_rca_agent.evaluation import (
    attribution_correctness_guideline,
    grounding_check,
    hypothesis_quality_guideline,
    includes_hypothesis,
    report_length_check,
    tool_diversity_check,
    validity_gate,
)
from ordr_bhvr_rca_agent.mcp import ToolInfo, create_mcp_tools
from ordr_bhvr_rca_agent.news_scraper import SCRAPE_NEWS_TOOL_SPEC, scrape_news
from ordr_bhvr_rca_agent.rag import ARXIV_TOOL_SPEC, make_arxiv_search_fn
from ordr_bhvr_rca_agent.vector_search import VectorSearchManager

# COMMAND ----------
# Initialize the RCA agent
cfg = ProjectConfig.from_yaml("../project_config.yml")
mlflow.set_experiment(cfg.experiment_name)

# Create workspace client
w = WorkspaceClient()

# Create MCP tools from Genie
host = w.config.host
mcp_urls = [f"{host}/api/2.0/mcp/genie/{cfg.genie_space_id}"]

import nest_asyncio

nest_asyncio.apply()
mcp_tools = asyncio.run(create_mcp_tools(w, mcp_urls))

# COMMAND ----------
# Create news scraping tool
news_tool = ToolInfo(
    name="scrape_news",
    spec=SCRAPE_NEWS_TOOL_SPEC,
    exec_fn=scrape_news,
)

logger.info("News scraping tool created")

# COMMAND ----------
# Create RAG tool backed by the arXiv paper index under mlops_dev.vishalkr
# The data pipeline (notebooks/2.x series) built this index under a different schema,
# so we point VectorSearchManager at it explicitly via index_name.

RAG_INDEX_NAME = "mlops_dev.vishalkr.arxiv_index"

vs_manager = VectorSearchManager(
    config=cfg,
    index_name=RAG_INDEX_NAME,
)

search_arxiv_papers = make_arxiv_search_fn(vs_manager)

RAG_TOOL_SPEC = ARXIV_TOOL_SPEC

rag_tool = ToolInfo(
    name="search_arxiv_papers",
    spec=RAG_TOOL_SPEC,
    exec_fn=search_arxiv_papers,
)

logger.info(f"RAG tool created — index: {RAG_INDEX_NAME}")

# COMMAND ----------
# Create enhanced agent with MCP tools + news scraping + RAG

all_tools = mcp_tools + [news_tool, rag_tool]

rca_agent_with_news = SimpleAgent(
    llm_endpoint=cfg.llm_endpoint,
    system_prompt=cfg.system_prompt,
    tools=all_tools,
)

logger.info(
    f"Created RCA agent with {len(mcp_tools)} MCP tools + 1 news scraping tool + 1 RAG tool = {len(all_tools)} total tools"
)

# COMMAND ----------
# Load RCA evaluation inputs
with open("../rca_eval_inputs.txt") as f:
    eval_data = [{"inputs": {"question": line.strip()}} for line in f if line.strip()]

import mlflow.openai

# Re-enable autologging — calls inside @mlflow.trace become child spans
mlflow.openai.autolog()


@mlflow.trace
def predict_fn_with_news(question: str) -> str:
    """Predict function using RCA agent with news scraping capability."""
    result = rca_agent_with_news.chat(question, max_iterations=12)
    return result


# COMMAND ----------
# Run evaluation with enhanced agent (MCP tools + news scraping)
results = mlflow.genai.evaluate(
    predict_fn=predict_fn_with_news,
    data=eval_data,
    scorers=[
        validity_gate,
        report_length_check,
        grounding_check,
        includes_hypothesis,
        tool_diversity_check,
        attribution_correctness_guideline,
        hypothesis_quality_guideline,
    ],
)

# COMMAND ----------

# Define resources used by the RCA agent
resources = [
    DatabricksServingEndpoint(endpoint_name=cfg.llm_endpoint),
    DatabricksGenieSpace(genie_space_id=cfg.genie_space_id),
    DatabricksTable(table_name=f"{cfg.catalog}.{cfg.schema}.{cfg.table_name}"),
    DatabricksSQLWarehouse(warehouse_id=cfg.warehouse_id),
    DatabricksServingEndpoint(endpoint_name=cfg.embedding_endpoint),
    DatabricksVectorSearchIndex(index_name=RAG_INDEX_NAME),
]

# COMMAND ----------
# Prepare model metadata
timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
session_id = f"s-{timestamp}-{random.randint(100000, 999999)}"
request_id = f"req-{timestamp}-{random.randint(100000, 999999)}"

test_request = {
    "input": [
        {
            "role": "user",
            "content": "Compare sales behavior from March 2018 to May 2018",
        }
    ],
    "custom_inputs": {
        "session_id": session_id,
        "request_id": request_id,
    },
}

model_config = {
    "catalog": cfg.catalog,
    "schema": cfg.schema,
    "genie_space_id": cfg.genie_space_id,
    "system_prompt": cfg.system_prompt,
    "llm_endpoint": cfg.llm_endpoint,
    "lakebase_project_id": cfg.lakebase_project_id,
}

git_sha = "abc"
run_id = "unset"

# Create wrapper file for MLflow PyFunc
wrapper_content = """\
import mlflow
from ordr_bhvr_rca_agent.pyfunc_model import RCAAgentModel

mlflow.models.set_model(RCAAgentModel())
"""
wrapper_dir = tempfile.mkdtemp()
wrapper_path = os.path.join(wrapper_dir, "pyfunc_model_wrapper.py")
with open(wrapper_path, "w") as f:
    f.write(wrapper_content)

# Define signature explicitly to avoid inference running the agent
output_example = {"response": "sample response"}
signature = infer_signature(test_request, output_example)

# COMMAND ----------
# Log model to MLflow
ts = datetime.now().strftime("%Y-%m-%d")
with mlflow.start_run(
    run_name=f"rca-agent-{ts}", tags={"git_sha": git_sha, "run_id": run_id}
) as run:
    model_info = mlflow.pyfunc.log_model(
        artifact_path="agent",
        python_model=wrapper_path,
        resources=resources,
        input_example=test_request,
        signature=signature,
        model_config=model_config,
    )
    mlflow.log_metrics(results.metrics)

# COMMAND ----------
# Register RCA model to Unity Catalog
model_name = f"{cfg.catalog}.{cfg.schema}.rca_agent"

registered_model = mlflow.register_model(
    model_uri=model_info.model_uri,
    name=model_name,
    tags={"git_sha": git_sha, "run_id": run_id},
    env_pack="databricks_model_serving",
)

# COMMAND ----------
# Set alias for the registered model
from mlflow import MlflowClient

client = MlflowClient()
client.set_registered_model_alias(
    name=model_name,
    alias="latest-model",
    version=registered_model.version,
)
