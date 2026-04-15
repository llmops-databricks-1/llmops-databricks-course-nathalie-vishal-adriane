# Databricks notebook source
# MAGIC %pip install /Workspace/Users/aschelin@gmail.com/.bundle/llmops-databricks-course-nathalie-vishal-adriane/dev/files/ nest-asyncio
# MAGIC dbutils.library.restartPython()

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
)

from ordr_bhvr_rca_agent.agent import SimpleAgent
from ordr_bhvr_rca_agent.config import ProjectConfig
from ordr_bhvr_rca_agent.evaluation import (
    attribution_correctness_guideline,
    hypothesis_quality_guideline,
    report_length_check,
)
from ordr_bhvr_rca_agent.mcp import ToolInfo, create_mcp_tools
from ordr_bhvr_rca_agent.news_scraper import SCRAPE_NEWS_TOOL_SPEC, scrape_news

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
# Create enhanced agent with MCP tools + news scraping

all_tools = mcp_tools + [news_tool]

rca_agent_with_news = SimpleAgent(
    llm_endpoint=cfg.llm_endpoint,
    system_prompt=cfg.system_prompt,
    tools=all_tools,
)

logger.info(
    f"Created RCA agent with {len(mcp_tools)} MCP tools + 1 news scraping tool = {len(all_tools)} total tools"
)

# COMMAND ----------
# Load RCA evaluation inputs
with open("../rca_eval_inputs.txt") as f:
    eval_data = [{"inputs": {"question": line.strip()}} for line in f if line.strip()]


def predict_fn_with_news(question: str) -> str:
    """Predict function using RCA agent with news scraping capability."""
    result = rca_agent_with_news.chat(question)
    return result


# COMMAND ----------
# Run evaluation with enhanced agent (MCP tools + news scraping)
results = mlflow.genai.evaluate(
    predict_fn=predict_fn_with_news,
    data=eval_data,
    scorers=[
        report_length_check,
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
    DatabricksServingEndpoint(endpoint_name="databricks-bge-large-en"),
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
