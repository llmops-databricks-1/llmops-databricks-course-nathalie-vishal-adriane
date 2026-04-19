# Databricks notebook source
# MAGIC %md
# MAGIC # Lecture 5.2: Service Principal Permissions for RCA Agent
# MAGIC
# MAGIC ## Topics Covered:
# MAGIC - Setting up service principal permissions
# MAGIC - Granting access to serving endpoints
# MAGIC - Granting access to vector search
# MAGIC - Granting access to Genie spaces
# MAGIC - Granting access to SQL warehouses

# COMMAND ----------

# MAGIC %pip install /Workspace/Users/aschelin@gmail.com/.bundle/llmops-databricks-course-nathalie-vishal-adriane/dev/files/dist/*.whl
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import EndpointPermissionLevel
from databricks.sdk.service.sql import AccessControl, PermissionLevel

from ordr_bhvr_rca_agent.config import ProjectConfig

cfg = ProjectConfig.from_yaml("../project_config.yml")
w = WorkspaceClient()

# COMMAND ----------
# Get service principal from secrets
spn_app_id = dbutils.secrets.get("dev_SPN", "client_id")  # noqa: F821

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Grant Permissions to Serving Endpoints
# MAGIC
# MAGIC The RCA agent needs access to:
# MAGIC - LLM endpoint (for hypothesis generation)
# MAGIC - Embedding endpoint (for vector search)

# COMMAND ----------

serving_endpoints = [
    cfg.llm_endpoint,
    "databricks-bge-large-en",
]

for ep_name in serving_endpoints:
    ep = w.serving_endpoints.get(ep_name)
    w.serving_endpoints.set_permissions(
        serving_endpoint_id=ep.id,
        access_control_list=[
            {
                "service_principal_name": spn_app_id,
                "permission_level": EndpointPermissionLevel.CAN_QUERY,
            }
        ],
    )
    print(f"✓ Granted CAN_QUERY permission to {ep_name}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Grant Permissions to Vector Search Endpoint
# MAGIC
# MAGIC For retrieving historical RCA reports and business context.

# COMMAND ----------

w.vector_search_endpoints.update_permissions(
    vector_search_endpoint_name=cfg.vector_search_endpoint,
    access_control_list=[
        {
            "service_principal_name": spn_app_id,
            "permission_level": "CAN_USE",
        }
    ],
)

print(
    f"✓ Granted CAN_USE permission to vector search endpoint: {cfg.vector_search_endpoint}"
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Grant Permissions to Genie Space
# MAGIC
# MAGIC For querying order metrics and e-commerce data.

# COMMAND ----------

w.genie.set_permissions(
    genie_space_id=cfg.genie_space_id,
    access_control_list=[
        {
            "service_principal_name": spn_app_id,
            "permission_level": "CAN_RUN",
        }
    ],
)

print(f"✓ Granted CAN_RUN permission to Genie space: {cfg.genie_space_id}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Grant Permissions to SQL Warehouse
# MAGIC
# MAGIC For executing Genie queries and accessing tables.

# COMMAND ----------

w.warehouses.set_permissions(
    warehouse_id=cfg.warehouse_id,
    access_control_list=[
        AccessControl(
            service_principal_name=spn_app_id,
            permission_level=PermissionLevel.CAN_USE,
        )
    ],
)

print(f"✓ Granted CAN_USE permission to SQL warehouse: {cfg.warehouse_id}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC Service principal permissions configured for RCA agent:
# MAGIC - ✓ Serving endpoints (LLM + embedding)
# MAGIC - ✓ Vector search endpoint
# MAGIC - ✓ Genie space
# MAGIC - ✓ SQL warehouse
# MAGIC
# MAGIC The RCA agent can now:
# MAGIC - Query Genie for order metrics
# MAGIC - Generate hypotheses using LLM
# MAGIC - Retrieve historical RCA reports via vector search
# MAGIC - Access e-commerce data tables
