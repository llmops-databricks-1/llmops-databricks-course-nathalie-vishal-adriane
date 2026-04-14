"""Simple agent implementation for tool-calling workflows."""

import json
from typing import Any

from databricks.sdk import WorkspaceClient
from loguru import logger
from openai import OpenAI

from .mcp import ToolInfo


class SimpleAgent:
    """A simple agent that can call tools in a loop.

    This agent orchestrates a conversation with an LLM that can call tools
    to perform actions or retrieve information. The agent maintains conversation
    context and handles the tool calling loop.

    Attributes:
        llm_endpoint: Name of the LLM serving endpoint
        system_prompt: System prompt that defines the agent's behavior
        workspace_client: Databricks workspace client (optional, auto-created if not provided)
    """

    def __init__(
        self,
        llm_endpoint: str,
        system_prompt: str,
        tools: list[ToolInfo],
        workspace_client: WorkspaceClient | None = None,
    ):
        """Initialize the agent.

        Args:
            llm_endpoint: Name of the LLM serving endpoint
            system_prompt: System prompt that defines the agent's behavior
            tools: List of ToolInfo objects representing available tools
            workspace_client: Optional Databricks workspace client
        """
        self.llm_endpoint = llm_endpoint
        self.system_prompt = system_prompt
        self._tools_dict = {tool.name: tool for tool in tools}

        # Use provided workspace client or create new one
        w = workspace_client or WorkspaceClient()

        # Create OpenAI client
        self._client = OpenAI(
            api_key=w.tokens.create(lifetime_seconds=1200).token_value,
            base_url=f"{w.config.host}/serving-endpoints",
        )

    def get_tool_specs(self) -> list[dict]:
        """Get tool specifications for the LLM.

        Returns:
            List of tool specifications in OpenAI format
        """
        return [tool.spec for tool in self._tools_dict.values()]

    def execute_tool(self, tool_name: str, args: dict) -> Any:
        """Execute a tool by name.

        Args:
            tool_name: Name of the tool to execute
            args: Arguments to pass to the tool

        Returns:
            Tool execution result

        Raises:
            ValueError: If tool not found
        """
        if tool_name not in self._tools_dict:
            raise ValueError(f"Unknown tool: {tool_name}")
        return self._tools_dict[tool_name].exec_fn(**args)

    def chat(self, user_message: str, max_iterations: int = 10) -> str:
        """Chat with the agent, allowing tool calls.

        Args:
            user_message: User's message/question
            max_iterations: Maximum number of tool-calling iterations

        Returns:
            Agent's final response
        """
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_message},
        ]

        for iteration in range(max_iterations):
            # Call LLM
            response = self._client.chat.completions.create(
                model=self.llm_endpoint,
                messages=messages,
                tools=self.get_tool_specs() if self._tools_dict else None,
            )

            assistant_message = response.choices[0].message

            # Check if LLM wants to call tools
            if assistant_message.tool_calls:
                # Add assistant message with tool calls (exclude unsupported fields)
                messages.append(
                    {
                        "role": "assistant",
                        "content": assistant_message.content,
                        "tool_calls": [
                            {
                                "id": tc.id,
                                "type": "function",
                                "function": {
                                    "name": tc.function.name,
                                    "arguments": tc.function.arguments,
                                },
                            }
                            for tc in assistant_message.tool_calls
                        ],
                    }
                )

                # Execute each tool call
                for tool_call in assistant_message.tool_calls:
                    tool_name = tool_call.function.name
                    tool_args = json.loads(tool_call.function.arguments)

                    logger.info(f"Calling tool: {tool_name}({tool_args})")

                    try:
                        result = self.execute_tool(tool_name, tool_args)
                    except Exception as e:
                        result = f"Error: {str(e)}"

                    # Add tool result to messages
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": str(result),
                        }
                    )
            else:
                # No tool calls, return the response
                return assistant_message.content

        return "Max iterations reached."
