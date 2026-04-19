"""Simple agent implementation for tool-calling workflows."""

import json

from databricks.sdk import WorkspaceClient
from loguru import logger
from openai import OpenAI

from .mcp import ToolInfo

# Trim tool results to this many characters to prevent context explosion.
# Genie responses can be very large; only the key figures matter for the LLM.
_MAX_TOOL_RESULT_CHARS = 3000

# After this many cumulative tool calls the agent is nudged to stop querying and
# write the final report, preventing runaway data-gathering loops.
_SYNTHESIZE_AFTER_N_TOOL_CALLS = 8

_SYNTHESIS_NUDGE = (
    "You have now made enough data queries. "
    "Do NOT call any more tools. "
    "Write your final structured RCA report now using the data already retrieved."
)


class SimpleAgent:
    """A simple agent that can call tools in a loop.

    This agent orchestrates a conversation with an LLM that can call tools
    to perform actions or retrieve information. The agent maintains conversation
    context and handles the tool calling loop.

    Attributes:
        llm_endpoint: Name of the LLM serving endpoint
        system_prompt: System prompt that defines the agent's behavior
        workspace_client: Databricks workspace client (optional,
            auto-created if not provided)
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

    @staticmethod
    def _to_plain(obj: object) -> object:
        """Recursively convert any object to plain JSON-safe Python types.

        Pydantic v2 models expose model_dump(mode='json') which fully flattens
        nested objects.  For anything else we fall back to vars() then str().
        """
        if isinstance(obj, dict):
            return {k: SimpleAgent._to_plain(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [SimpleAgent._to_plain(i) for i in obj]
        if hasattr(obj, "model_dump"):
            return SimpleAgent._to_plain(obj.model_dump(mode="json"))
        if hasattr(obj, "dict"):  # Pydantic v1
            return SimpleAgent._to_plain(obj.dict())
        if isinstance(obj, (str, int, float, bool)) or obj is None:
            return obj
        # Last resort: convert to string so the JSON round-trip never blows up
        return str(obj)

    def get_tool_specs(self) -> list[dict]:
        """Get tool specifications for the LLM.

        Returns a list of plain Python dicts compatible with
        Databricks List[Map[String, Any]] requirement.
        """
        return [SimpleAgent._to_plain(tool.spec) for tool in self._tools_dict.values()]

    @staticmethod
    def _truncate_tool_result(result: str) -> str:
        """Trim a tool result to prevent context explosion.

        Large Genie responses are cut to the most important portion so the
        conversation history stays within a reasonable token budget.
        """
        if len(result) <= _MAX_TOOL_RESULT_CHARS:
            return result
        return result[:_MAX_TOOL_RESULT_CHARS] + (
            f"\n[...truncated — {len(result)} total chars. "
            "Only leading content shown; extract key figures from above.]"
        )

    def execute_tool(self, tool_name: str, args: dict) -> object:
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

    def chat(self, user_message: str, max_iterations: int = 15) -> str:
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

        tool_call_count = 0  # tracks cumulative tool calls across all iterations

        for iteration in range(max_iterations):
            # If the agent has already made enough tool calls, force synthesis by
            # removing tools from the next LLM call so it cannot call more.
            tools_available = self.get_tool_specs() if self._tools_dict else None
            if tool_call_count >= _SYNTHESIZE_AFTER_N_TOOL_CALLS:
                tools_available = None
                if (
                    messages[-1].get("role") != "user"
                    or messages[-1].get("content") != _SYNTHESIS_NUDGE
                ):
                    messages.append({"role": "user", "content": _SYNTHESIS_NUDGE})
                    logger.info(
                        f"Iteration {iteration}: tool_call_count={tool_call_count} "
                        "— forcing synthesis, tools disabled for this turn."
                    )

            # Call LLM
            # Pass tools via extra_body instead of the tools= parameter to prevent
            # the OpenAI SDK from injecting fields like "strict": null that the
            # Databricks serving endpoint rejects.
            response = self._client.chat.completions.create(
                model=self.llm_endpoint,
                messages=messages,
                **({"extra_body": {"tools": tools_available}} if tools_available else {}),
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

                    tool_call_count += 1

                    # Add tool result to messages, truncated to avoid context explosion
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": self._truncate_tool_result(str(result)),
                        }
                    )
            else:
                # No tool calls, return the response
                return assistant_message.content

        return "Max iterations reached."
