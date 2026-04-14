# Databricks notebook source
# MAGIC %md
# MAGIC # Lecture 5.1: RCA Agent Deployment & Testing
# MAGIC
# MAGIC ## Topics Covered:
# MAGIC - Deploying RCA agents using `agents.deploy()`
# MAGIC - Configuring environment variables and secrets
# MAGIC - Testing deployed RCA endpoints
# MAGIC - Using OpenAI-compatible client for RCA queries
# MAGIC
# MAGIC ## Prerequisites:
# MAGIC - For local execution: `pip install mlflow[databricks]` to access Unity Catalog models

# COMMAND ----------

# MAGIC %pip install /Workspace/Users/aschelin@gmail.com/.bundle/llmops-databricks-course-nathalie-vishal-adriane/dev/files/dist/*.whl
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import os

import mlflow
from databricks import agents
from databricks.sdk import WorkspaceClient
from loguru import logger
from mlflow import MlflowClient

from ordr_bhvr_rca_agent.config import ProjectConfig

# Setup MLflow tracking
if "DATABRICKS_RUNTIME_VERSION" not in os.environ:
    from dotenv import load_dotenv

    load_dotenv()
    profile = os.environ.get("PROFILE", "DEFAULT")
    mlflow.set_tracking_uri(f"databricks://{profile}")
    mlflow.set_registry_uri(f"databricks-uc://{profile}")

cfg = ProjectConfig.from_yaml("../project_config.yml")

model_name = f"{cfg.catalog}.{cfg.schema}.rca_agent"
endpoint_name = "rca-agent-endpoint-dev-course"
secret_scope = "rca-agent-scope"

model_version = (
    MlflowClient().get_model_version_by_alias(model_name, "latest-model").version
)

workspace = WorkspaceClient()
experiment = MlflowClient().get_experiment_by_name(cfg.experiment_name)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Deploy RCA Agent
# MAGIC
# MAGIC The `agents.deploy()` API handles:
# MAGIC - Endpoint creation and configuration
# MAGIC - Inference tables for monitoring RCA queries
# MAGIC - Environment variables and secrets
# MAGIC - Model versioning

# COMMAND ----------

git_sha = "local"

agents.deploy(
    model_name=model_name,
    model_version=int(model_version),
    endpoint_name=endpoint_name,
    usage_policy_id=cfg.usage_policy_id,
    scale_to_zero=True,
    workload_size="Small",
    deploy_feedback_model=False,
    environment_vars={
        "GIT_SHA": git_sha,
        "MODEL_VERSION": model_version,
        "MODEL_SERVING_ENDPOINT_NAME": endpoint_name,
        "MLFLOW_EXPERIMENT_ID": experiment.experiment_id,
        "LAKEBASE_SP_CLIENT_ID": f"{{{{secrets/{secret_scope}/client-id}}}}",
        "LAKEBASE_SP_CLIENT_SECRET": f"{{{{secrets/{secret_scope}/client-secret}}}}",
        "LAKEBASE_SP_HOST": WorkspaceClient().config.host,
    },
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Test the Deployed RCA Endpoint
# MAGIC
# MAGIC Wait for deployment to complete (5-10 minutes), then test the endpoint with RCA queries.

# COMMAND ----------

import random
from datetime import datetime

from openai import OpenAI

host = workspace.config.host
token = workspace.tokens.create(lifetime_seconds=2000).token_value

client = OpenAI(
    api_key=token,
    base_url=f"{host}/serving-endpoints",
)

timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
session_id = f"s-{timestamp}-{random.randint(100000, 999999)}"
request_id = f"req-{timestamp}-{random.randint(100000, 999999)}"

response = client.responses.create(
    model=endpoint_name,
    input=[
        {"role": "user", "content": "Compare sales behavior from March 2018 to May 2018"}
    ],
    extra_body={
        "custom_inputs": {
            "session_id": session_id,
            "request_id": request_id,
        }
    },
)

logger.info(f"Response ID: {response.id}")
logger.info(f"Session ID: {response.custom_outputs.get('session_id')}")
logger.info(f"Request ID: {response.custom_outputs.get('request_id')}")
logger.info("\nRCA Analysis:")
logger.info("-" * 80)
logger.info(response.output[0].content[0].text)
logger.info("-" * 80)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Multi-Turn RCA Conversation

# COMMAND ----------

# Start an RCA conversation
rca_session = (
    f"s-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{random.randint(100000, 999999)}"
)

# Turn 1: Initial anomaly query
response1 = client.responses.create(
    model=endpoint_name,
    input=[{"role": "user", "content": "What happened to order volumes in May 2018?"}],
    extra_body={
        "custom_inputs": {
            "session_id": rca_session,
            "request_id": f"req-1-{random.randint(100000, 999999)}",
        }
    },
)

logger.info("Turn 1:")
logger.info("User: What happened to order volumes in May 2018?")
logger.info(f"Agent: {response1.output[0].content[0].text}")

# Turn 2: Follow-up question
response2 = client.responses.create(
    model=endpoint_name,
    input=[
        {"role": "user", "content": "What happened to order volumes in May 2018?"},
        {"role": "assistant", "content": response1.output[0].content[0].text},
        {"role": "user", "content": "Which product categories were most affected?"},
    ],
    extra_body={
        "custom_inputs": {
            "session_id": rca_session,
            "request_id": f"req-2-{random.randint(100000, 999999)}",
        }
    },
)

logger.info("\nTurn 2:")
logger.info("User: Which product categories were most affected?")
logger.info(f"Agent: {response2.output[0].content[0].text}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC In this notebook, you learned:
# MAGIC - How to deploy RCA agents using `agents.deploy()`
# MAGIC - Configuring environment variables for deployed agents
# MAGIC - Testing RCA endpoints with OpenAI-compatible client
# MAGIC - Running multi-turn RCA conversations
# MAGIC
# MAGIC **Next**: [5.2_spn_permissions.py](5.2_spn_permissions.py) - Setting up service principal permissions
