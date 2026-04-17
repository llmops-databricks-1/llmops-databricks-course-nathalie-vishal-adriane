"""MCP (Model Context Protocol) integration utilities."""

from collections.abc import Callable

from databricks.sdk import WorkspaceClient
from databricks_mcp import DatabricksMCPClient
from pydantic import BaseModel


class ToolInfo(BaseModel):
    """Tool information for agent integration.

    Attributes:
        name: Tool name
        spec: JSON description of the tool (OpenAI Responses format)
        exec_fn: Function that implements the tool logic
    """

    name: str
    spec: dict
    exec_fn: Callable

    class Config:
        arbitrary_types_allowed = True


def create_managed_exec_fn(
    server_url: str, tool_name: str, w: WorkspaceClient
) -> Callable:
    """Create an execution function for an MCP tool.

    Args:
        server_url: MCP server URL
        tool_name: Name of the tool
        w: Databricks workspace client

    Returns:
        Callable that executes the tool
    """

    def exec_fn(**kwargs: object) -> str:
        client = DatabricksMCPClient(server_url=server_url, workspace_client=w)
        response = client.call_tool(tool_name, kwargs)
        return "".join([c.text for c in response.content])

    return exec_fn


async def create_mcp_tools(w: WorkspaceClient, url_list: list[str]) -> list[ToolInfo]:
    """Create tools from MCP servers.

    Args:
        w: Databricks workspace client
        url_list: List of MCP server URLs

    Returns:
        List of ToolInfo objects
    """
    tools = []
    for server_url in url_list:
        mcp_client = DatabricksMCPClient(server_url=server_url, workspace_client=w)
        mcp_tools = mcp_client.list_tools()
        for mcp_tool in mcp_tools:
            # inputSchema from the MCP SDK may be a Pydantic model or custom object.
            # Force it to a plain dict via JSON round-trip so the Databricks API
            # receives only native Python types (List[Map[String, Any]]).
            raw_schema = mcp_tool.inputSchema
            if raw_schema is None:
                input_schema: dict = {}
            elif hasattr(raw_schema, "model_dump"):
                # model_dump(mode='json') recursively converts all nested Pydantic
                # objects to plain JSON-safe types (Pydantic v2).
                input_schema = raw_schema.model_dump(mode="json")
            elif isinstance(raw_schema, dict):
                input_schema = raw_schema
            else:
                import dataclasses
                import json as _json

                def _clean(o: object) -> object:
                    if hasattr(o, "model_dump"):
                        return o.model_dump(mode="json")
                    if dataclasses.is_dataclass(o) and not isinstance(o, type):
                        return dataclasses.asdict(o)
                    return str(o)

                input_schema = _json.loads(_json.dumps(raw_schema, default=_clean))
            tool_spec = {
                "type": "function",
                "function": {
                    "name": mcp_tool.name,
                    "parameters": input_schema,
                    "description": mcp_tool.description or f"Tool: {mcp_tool.name}",
                },
            }
            exec_fn = create_managed_exec_fn(server_url, mcp_tool.name, w)
            tools.append(ToolInfo(name=mcp_tool.name, spec=tool_spec, exec_fn=exec_fn))
    return tools
