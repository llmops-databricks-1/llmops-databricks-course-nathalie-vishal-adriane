# Databricks notebook source
# MAGIC %md
# MAGIC # Lecture 3.2: Model Context Protocol (MCP) Integration
# MAGIC
# MAGIC ## Topics Covered:
# MAGIC - What is MCP?
# MAGIC - MCP vs custom functions
# MAGIC - Databricks MCP servers
# MAGIC - Vector Search MCP
# MAGIC - Genie Space MCP
# MAGIC - Creating MCP tools for agents

# COMMAND ----------

# MAGIC %pip install /Workspace/Users/aschelin@gmail.com/.bundle/llmops-databricks-course-nathalie-vishal-adriane/dev/files

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

import asyncio
import json

import nest_asyncio
from databricks.sdk import WorkspaceClient
from databricks_mcp import DatabricksMCPClient
from loguru import logger
from pyspark.sql import SparkSession

from ordr_bhvr_rca_agent.agent import SimpleAgent
from ordr_bhvr_rca_agent.config import get_env, load_config
from ordr_bhvr_rca_agent.mcp import create_mcp_tools

# Enable nested event loops (required for Databricks notebooks)
nest_asyncio.apply()

# COMMAND ----------
spark = SparkSession.builder.getOrCreate()

# Load configuration
env = get_env(spark)
cfg = load_config("../project_config.yml", env)

w = WorkspaceClient()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. What is Model Context Protocol (MCP)?
# MAGIC
# MAGIC **MCP** is a standardized protocol for connecting AI models to external data sources and tools.
# MAGIC
# MAGIC ### Key Concepts:
# MAGIC
# MAGIC - **MCP Server**: Exposes tools and resources
# MAGIC - **MCP Client**: Connects to servers and calls tools
# MAGIC - **Tools**: Functions that can be called
# MAGIC - **Resources**: Data that can be accessed
# MAGIC
# MAGIC ### Why MCP?
# MAGIC
# MAGIC **Standardized**: Common protocol across different systems
# MAGIC **Reusable**: One MCP server, many agents
# MAGIC **Managed**: Databricks manages the infrastructure
# MAGIC **Secure**: Built-in authentication and authorization
# MAGIC **Scalable**: Enterprise-grade performance

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. MCP vs Custom Functions
# MAGIC
# MAGIC | Aspect | Custom Functions | MCP |
# MAGIC |--------|-----------------|-----|
# MAGIC | **Setup** | Write Python code | Use existing MCP servers |
# MAGIC | **Maintenance** | You maintain | Databricks maintains |
# MAGIC | **Reusability** | Per-agent | Across agents |
# MAGIC | **Security** | Manual | Built-in |
# MAGIC | **Scalability** | Manual | Automatic |
# MAGIC | **Best For** | Custom logic | Standard operations |
# MAGIC
# MAGIC **Use MCP when**: You need standard operations (search, query, etc.)
# MAGIC **Use Custom Functions when**: You need custom business logic

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Databricks MCP Servers
# MAGIC
# MAGIC Databricks provides managed MCP servers for:
# MAGIC
# MAGIC 1. **Vector Search MCP**: Search vector indexes
# MAGIC 2. **Genie Space MCP**: Query data using natural language
# MAGIC 3. **Unity Catalog Functions MCP**: Execute UC functions
# MAGIC 4. **SQL Warehouse MCP**: Execute SQL queries

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Vector Search MCP

# COMMAND ----------

# MAGIC %md
# MAGIC ### Vector Search MCP URL Format:
# MAGIC ```
# MAGIC {workspace_host}/api/2.0/mcp/vector-search/{catalog}/{schema}
# MAGIC ```
# MAGIC
# MAGIC **How it works:**
# MAGIC - The MCP server scans all vector search indexes in the specified catalog/schema
# MAGIC - For each index, it creates a tool with name: `catalog__schema__index_name`
# MAGIC - Each tool takes a single parameter: `query` (the search text)
# MAGIC - The tool automatically handles embedding and similarity search

# COMMAND ----------

# Create Vector Search MCP URL (matches O'Reilly source)
host = w.config.host
vector_search_mcp_url = f"{host}/api/2.0/mcp/vector-search/{cfg.catalog}/{cfg.schema}"

logger.info("Vector Search MCP URL:")
logger.info(vector_search_mcp_url)

# COMMAND ----------

# MAGIC %md
# MAGIC ### List Available Tools from Vector Search MCP

# COMMAND ----------

# Connect to Vector Search MCP
vs_mcp_client = DatabricksMCPClient(server_url=vector_search_mcp_url, workspace_client=w)

# List available tools
vs_tools = vs_mcp_client.list_tools()

logger.info(f"Vector Search MCP Tools ({len(vs_tools)}):")
logger.info("=" * 80)
for tool in vs_tools:
    logger.info(f"Tool: {tool.name}")
    logger.info(f"Description: {tool.description}")
    if tool.inputSchema:
        logger.info(f"Parameters: {list(tool.inputSchema.get('properties', {}).keys())}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Call Vector Search Tool
# MAGIC
# MAGIC **Important**: The MCP tool name uses double underscores:
# MAGIC - Tool name: `workspace__course_data__arxiv_index`
# MAGIC - Parameter: `query` (just the search query text)
# MAGIC - The index is already specified in the tool name itself

# COMMAND ----------

# Search for papers about machine learning
# The tool name is the index name with '__' separators
tool_name = f"{cfg.catalog}__{cfg.schema}__arxiv_index"

search_result = vs_mcp_client.call_tool(
    tool_name, {"query": "machine learning and neural networks"}
)

logger.info("Search Results:")
logger.info("=" * 80)
for content in search_result.content:
    logger.info(content.text)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Genie Space MCP

# COMMAND ----------

# MAGIC %md
# MAGIC ### Genie Space MCP URL Format:
# MAGIC ```
# MAGIC {workspace_host}/api/2.0/mcp/genie/{genie_space_id}
# MAGIC ```
# MAGIC
# MAGIC **Genie** allows natural language queries over your data.

# COMMAND ----------

# Check if Genie space is configured
if hasattr(cfg, "genie_space_id") and cfg.genie_space_id:
    genie_mcp_url = f"{host}/api/2.0/mcp/genie/{cfg.genie_space_id}"
    logger.info("Genie MCP URL:")
    logger.info(genie_mcp_url)

    # Connect to Genie MCP
    genie_mcp_client = DatabricksMCPClient(server_url=genie_mcp_url, workspace_client=w)

    # List available tools
    genie_tools = genie_mcp_client.list_tools()

    logger.info(f"Genie MCP Tools ({len(genie_tools)}):")
    logger.info("=" * 80)
    for tool in genie_tools:
        logger.info(f"Tool: {tool.name}")
        logger.info(f"Description: {tool.description}")
else:
    logger.warning("⚠️ Genie space not configured in project_config.yml")
    logger.info("To use Genie MCP, add 'genie_space_id' to your configuration")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Creating MCP Tools for Agents
# MAGIC
# MAGIC **Using `arxiv_curator.mcp` module:**
# MAGIC
# MAGIC We've imported the following from our custom package:
# MAGIC - `ToolInfo`: Pydantic model for tool information (name, spec, exec_fn)
# MAGIC - `create_managed_exec_fn()`: Creates execution functions for MCP tools
# MAGIC - `create_mcp_tools()`: Converts MCP server tools to agent-compatible tools
# MAGIC
# MAGIC These utilities handle:
# MAGIC 1. Connecting to MCP servers
# MAGIC 2. Listing available tools
# MAGIC 3. Creating OpenAI-compatible tool specifications
# MAGIC 4. Creating execution functions that call the MCP tools

# COMMAND ----------

# MAGIC %md
# MAGIC ### Load All MCP Tools

# COMMAND ----------

# Define MCP server URLs
mcp_urls = [f"{host}/api/2.0/mcp/vector-search/{cfg.catalog}/{cfg.schema}"]

# Add Genie if configured
if hasattr(cfg, "genie_space_id") and cfg.genie_space_id:
    mcp_urls.append(f"{host}/api/2.0/mcp/genie/{cfg.genie_space_id}")

logger.info(f"Loading tools from {len(mcp_urls)} MCP servers...")

# Create tools
mcp_tools = asyncio.run(create_mcp_tools(w, mcp_urls))

logger.info(f"✓ Loaded {len(mcp_tools)} tools from MCP servers")
logger.info("Available Tools:")
for i, tool in enumerate(mcp_tools, 1):
    logger.info(f"{i}. {tool.name}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Using MCP Tools

# COMMAND ----------

# Create a tools dictionary for easy access
tools_dict = {tool.name: tool for tool in mcp_tools}

# Example: Use vector search tool directly
# The tool name is the index name with '__' separators
vector_search_tool_name = f"{cfg.catalog}__{cfg.schema}__arxiv_index"

if vector_search_tool_name in tools_dict:
    search_tool = tools_dict[vector_search_tool_name]

    # Execute the tool - only takes 'query' parameter
    result = search_tool.exec_fn(query="deep learning architectures")

    logger.info("Search Results:")
    logger.info(result)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. MCP Tool Specifications

# COMMAND ----------

# View tool specifications (what the LLM sees)
if mcp_tools:
    logger.info("Tool Specifications for LLM:")
    logger.info("=" * 80)

    for tool in mcp_tools[:2]:  # Show first 2 tools
        logger.info(f"Tool: {tool.name}")
        logger.info(json.dumps(tool.spec, indent=2))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Benefits of MCP

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1. **No Code Required**
# MAGIC ```python
# MAGIC # Without MCP: Write custom function
# MAGIC def search_papers(query: str):
# MAGIC     # 50+ lines of code
# MAGIC     pass
# MAGIC
# MAGIC # With MCP: Just use it
# MAGIC tools = asyncio.run(create_mcp_tools(w, [mcp_url]))
# MAGIC ```
# MAGIC
# MAGIC ### 2. **Automatic Updates**
# MAGIC - Databricks maintains the MCP servers
# MAGIC - New features added automatically
# MAGIC - Bug fixes without code changes
# MAGIC
# MAGIC ### 3. **Consistent Interface**
# MAGIC - Same pattern for all MCP tools
# MAGIC - Easy to add new MCP servers
# MAGIC - Standardized error handling
# MAGIC
# MAGIC ### 4. **Enterprise Features**
# MAGIC - Built-in authentication
# MAGIC - Audit logging
# MAGIC - High availability

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. Troubleshooting MCP

# COMMAND ----------


def test_mcp_connection(mcp_url: str) -> bool:
    """Test if MCP server is accessible.

    Args:
        mcp_url: MCP server URL

    Returns:
        True if connection successful
    """
    try:
        client = DatabricksMCPClient(server_url=mcp_url, workspace_client=w)
        tools = client.list_tools()
        logger.info("✓ Connected to MCP server")
        logger.info(f"  URL: {mcp_url}")
        logger.info(f"  Tools available: {len(tools)}")
        return True
    except Exception as e:
        logger.error("✗ Failed to connect to MCP server")
        logger.error(f"  URL: {mcp_url}")
        logger.error(f"  Error: {e}")
        return False


# Test Vector Search MCP
logger.info("Testing Vector Search MCP:")
test_mcp_connection(vector_search_mcp_url)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 12. Using MCP Tools with an Agent
# MAGIC
# MAGIC Now let's create a simple agent that can use MCP tools.
# MAGIC
# MAGIC **Note:** We're using the `SimpleAgent` class from the `ordr_bhvr_rca_agent` package.
# MAGIC This is the same agent used in notebook 3.1, demonstrating that:
# MAGIC - Custom tools and MCP tools work with the same agent
# MAGIC - The agent doesn't care where tools come from
# MAGIC - You can mix custom and MCP tools in the same agent

# COMMAND ----------

# The SimpleAgent class is imported from the package
# It works identically with both custom tools and MCP tools
# See notebook 3.1 for the original introduction

# Create agent with MCP tools
agent = SimpleAgent(
    llm_endpoint=cfg.llm_endpoint,
    system_prompt="You are a helpful research assistant. Use the available tools to search for papers and answer questions.",
    tools=mcp_tools,
)

logger.info("✓ Agent created with MCP tools:")
for tool_name in agent._tools_dict.keys():
    logger.info(f"  - {tool_name}")

# COMMAND ----------

# Test agent with MCP vector search tool
logger.info("Testing agent with MCP tools:")
logger.info("=" * 80)

response = agent.chat("Give me sales from month May 2018")
logger.info(f"Agent response: {response}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 13. Comparison: RCA Agent with MCP vs Custom Tools
# MAGIC
# MAGIC In notebook 3.1, we built a Sales Order RCA Agent using **custom tools with REST API**.
# MAGIC Let's compare the Genie integration approaches.
# MAGIC
# MAGIC ### Approach 1: Custom Tool with REST API (from 3.1)
# MAGIC
# MAGIC ```python
# MAGIC def genie_query(question: str, space_id: str = None, timeout: int = 120) -> str:
# MAGIC     """60+ lines of code including:"""
# MAGIC     # 1. Start conversation via REST API
# MAGIC     resp = requests.post(f"{base}/start-conversation", headers=HEADERS, json={"content": question})
# MAGIC     conv_id = resp.json()["conversation_id"]
# MAGIC     msg_id = resp.json()["message_id"]
# MAGIC
# MAGIC     # 2. Poll for completion
# MAGIC     while time.time() < deadline:
# MAGIC         r = requests.get(poll_url, headers=HEADERS)
# MAGIC         status = r.json().get("status", "UNKNOWN")
# MAGIC         if status in ("COMPLETED", "FAILED"):
# MAGIC             break
# MAGIC         time.sleep(3)
# MAGIC
# MAGIC     # 3. Extract and parse results
# MAGIC     # ... 30+ more lines ...
# MAGIC ```
# MAGIC
# MAGIC **Complexity:**
# MAGIC - Manual REST API calls
# MAGIC - Polling logic
# MAGIC - Response parsing
# MAGIC - Error handling
# MAGIC - Authentication management
# MAGIC - **Total: ~60 lines of code**
# MAGIC
# MAGIC ### Approach 2: MCP (this notebook)
# MAGIC
# MAGIC ```python
# MAGIC # Step 1: Load Genie MCP tools
# MAGIC genie_mcp_url = f"{host}/api/2.0/mcp/genie/{genie_space_id}"
# MAGIC genie_tools = asyncio.run(create_mcp_tools(w, [genie_mcp_url]))
# MAGIC
# MAGIC # Step 2: Use them!
# MAGIC # That's it. No REST API code needed.
# MAGIC ```
# MAGIC
# MAGIC **Benefits:**
# MAGIC - ✅ No manual API calls
# MAGIC - ✅ No polling logic
# MAGIC - ✅ Automatic response parsing
# MAGIC - ✅ Built-in error handling
# MAGIC - ✅ Managed authentication
# MAGIC - **Total: ~3 lines of code**
# MAGIC
# MAGIC **The MCP approach is 20× simpler!**

# COMMAND ----------

# MAGIC %md
# MAGIC ### Building RCA Agent with MCP
# MAGIC
# MAGIC Let's build the same RCA agent, but using Genie MCP instead of REST API.

# COMMAND ----------

# First, check if we have Genie tools loaded
if hasattr(cfg, "genie_space_id") and cfg.genie_space_id:
    logger.info("Setting up RCA Agent with MCP...")

    # Load Genie MCP tools
    genie_mcp_url = f"{host}/api/2.0/mcp/genie/{cfg.genie_space_id}"

    try:
        genie_tools = asyncio.run(create_mcp_tools(w, [genie_mcp_url]))
        logger.info(f"✓ Loaded {len(genie_tools)} Genie MCP tools")

        # The Genie MCP tools are now ready to use!
        # No REST API code, no polling, no manual parsing!

    except Exception as e:
        logger.warning(f"Could not load Genie MCP: {e}")
        logger.info("Make sure your Genie space is properly configured")
        genie_tools = []
else:
    logger.warning("No Genie space configured - see notebook 3.2b to set one up")
    genie_tools = []

# COMMAND ----------

# MAGIC %md
# MAGIC ### Add Other RCA Tools
# MAGIC
# MAGIC We'll reuse the other RCA tools from 3.1 (news scraping, hypothesis generation, reporting)
# MAGIC since these don't have MCP equivalents. Only Genie integration is simplified with MCP.

# COMMAND ----------

# Install dependencies (if not already installed)
# %pip install feedparser databricks-openai --quiet

# Import for RCA tools

# Reuse the same tool functions from 3.1
# (scrape_news, generate_rca_hypotheses, generate_rca_report)
# These are identical - only the Genie tool changes!

logger.info("RCA tools from 3.1 can be reused here")
logger.info("Only difference: Genie MCP instead of Genie REST API")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Create Combined RCA Agent
# MAGIC
# MAGIC Combine Genie MCP tools with custom RCA tools.

# COMMAND ----------

if genie_tools:
    # For demonstration: show how to combine MCP and custom tools
    # In practice, you'd add the news/hypothesis/report tools here

    rca_agent_mcp = SimpleAgent(
        llm_endpoint=cfg.llm_endpoint,
        system_prompt="""You are a sales analytics assistant using Genie MCP for data queries.

Use the Genie tools to query sales data.
For full RCA workflow, additional tools would be:
- scrape_news
- generate_rca_hypotheses
- generate_rca_report

(These are the same custom tools from notebook 3.1)""",
        tools=genie_tools,
        workspace_client=w,
    )

    logger.info("✓ RCA Agent created with Genie MCP")
    logger.info(f"  Number of tools: {len(genie_tools)}")

    # Test it
    logger.info("\nTesting Genie MCP in RCA context:")
    try:
        test_response = rca_agent_mcp.chat(
            "Give me a brief summary of the available data", max_iterations=5
        )
        logger.info(f"Response: {test_response[:200]}...")
    except Exception as e:
        logger.error(f"Error: {e}")
else:
    logger.warning("Skipping RCA agent creation - no Genie tools available")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 14. Key Takeaways: MCP vs Custom Tools
# MAGIC
# MAGIC ### When to Use Custom Tools (Notebook 3.1)
# MAGIC
# MAGIC ✅ **Custom business logic**
# MAGIC - Domain-specific calculations
# MAGIC - Proprietary algorithms
# MAGIC - Unique workflows
# MAGIC
# MAGIC ✅ **External integrations**
# MAGIC - Third-party APIs (Google News, etc.)
# MAGIC - Legacy systems
# MAGIC - Custom data sources
# MAGIC
# MAGIC ✅ **Learning and control**
# MAGIC - Understanding the full implementation
# MAGIC - Custom error handling
# MAGIC - Specific performance optimizations
# MAGIC
# MAGIC ### When to Use MCP (This Notebook)
# MAGIC
# MAGIC ✅ **Standard Databricks operations**
# MAGIC - Vector search
# MAGIC - Genie queries
# MAGIC - UC function calls
# MAGIC - SQL warehouse queries
# MAGIC
# MAGIC ✅ **Production deployments**
# MAGIC - Managed infrastructure
# MAGIC - Automatic updates
# MAGIC - Enterprise security
# MAGIC - Built-in monitoring
# MAGIC
# MAGIC ✅ **Rapid development**
# MAGIC - Minimal code
# MAGIC - No maintenance burden
# MAGIC - Focus on business logic
# MAGIC
# MAGIC ### Best Practice: Hybrid Approach
# MAGIC
# MAGIC **Combine both!**
# MAGIC ```python
# MAGIC # MCP for Databricks-native operations
# MAGIC databricks_tools = asyncio.run(create_mcp_tools(w, [
# MAGIC     vector_search_mcp_url,
# MAGIC     genie_mcp_url
# MAGIC ]))
# MAGIC
# MAGIC # Custom tools for business logic
# MAGIC custom_tools = [
# MAGIC     scrape_news_tool,
# MAGIC     generate_hypotheses_tool,
# MAGIC     generate_report_tool
# MAGIC ]
# MAGIC
# MAGIC # Agent uses both
# MAGIC agent = SimpleAgent(
# MAGIC     llm_endpoint=cfg.llm_endpoint,
# MAGIC     system_prompt="...",
# MAGIC     tools=databricks_tools + custom_tools
# MAGIC )
# MAGIC ```
# MAGIC
# MAGIC ### The RCA Example Demonstrates:
# MAGIC
# MAGIC 1. **Notebook 3.1** - Full custom implementation
# MAGIC    - Shows how everything works
# MAGIC    - Great for learning
# MAGIC    - More code to maintain
# MAGIC
# MAGIC 2. **Notebook 3.2** - MCP simplification
# MAGIC    - Shows production best practice
# MAGIC    - Less code, same functionality
# MAGIC    - Easier to maintain
# MAGIC
# MAGIC **Both approaches are valid! Choose based on your needs.**
