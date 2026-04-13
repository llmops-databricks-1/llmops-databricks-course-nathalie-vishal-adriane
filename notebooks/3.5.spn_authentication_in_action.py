# Databricks notebook source

# COMMAND ----------

# MAGIC %pip install /Workspace/Users/aschelin@gmail.com/.bundle/llmops-databricks-course-nathalie-vishal-adriane/dev/files

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

import os
from uuid import uuid4

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.postgres import PostgresAPI
from loguru import logger

from ordr_bhvr_rca_agent.config import ProjectConfig
from ordr_bhvr_rca_agent.memory import LakebaseMemory

cfg = ProjectConfig.from_yaml("../project_config.yml")

w = WorkspaceClient()
pg_api = PostgresAPI(w.api_client)

project_id = cfg.lakebase_project_id

scope_name = "rca-agent-scope"
os.environ["LAKEBASE_SP_CLIENT_ID"] = dbutils.secrets.get(scope_name, "client_id")
os.environ["LAKEBASE_SP_CLIENT_SECRET"] = dbutils.secrets.get(scope_name, "client_secret")


w = WorkspaceClient()
os.environ["LAKEBASE_SP_HOST"] = w.config.host

# COMMAND ----------
instance_name = "rca-agent-instance"
instance = w.database.get_database_instance(instance_name)
lakebase_host = instance.read_write_dns

project = pg_api.get_project(name=f"projects/{project_id}")

memory = LakebaseMemory(
    project_id=project_id,
)

# COMMAND ----------

# Create a test session
session_id = f"test-session-{uuid4()}"

# Save some messages
test_messages = [
    {"role": "user", "content": "What factors contributed to the sales decline?"},
    {
        "role": "assistant",
        "content": "Based on the analysis, several factors contributed to the decline...",
    },
    {"role": "user", "content": "Tell me more about the seasonal trends"},
]

memory.save_messages(session_id, test_messages)
logger.info(f"✓ Saved {len(test_messages)} messages to session: {session_id}")

# COMMAND ----------

# Load messages back
loaded_messages = memory.load_messages(session_id)
