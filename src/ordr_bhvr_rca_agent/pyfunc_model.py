"""PyFunc wrapper for RCA Agent deployment."""

import asyncio

from databricks.sdk import WorkspaceClient
from mlflow.pyfunc import PythonModel

from ordr_bhvr_rca_agent.agent import SimpleAgent
from ordr_bhvr_rca_agent.mcp import create_mcp_tools


class RCAAgentModel(PythonModel):
    """MLflow PyFunc model wrapper for the RCA agent."""

    def load_context(self, context: object) -> None:
        """Load the model and initialize the agent.

        Args:
            context: MLflow context with model artifacts and config
        """
        import nest_asyncio

        nest_asyncio.apply()

        # Get model configuration
        cfg = context.model_config

        # Initialize workspace client
        w = WorkspaceClient()

        # Create MCP tools
        host = w.config.host
        mcp_urls = [f"{host}/api/2.0/mcp/genie/{cfg['genie_space_id']}"]
        mcp_tools = asyncio.run(create_mcp_tools(w, mcp_urls))

        # Initialize the agent
        self.agent = SimpleAgent(
            llm_endpoint=cfg["llm_endpoint"],
            system_prompt=cfg["system_prompt"],
            tools=mcp_tools,
        )

    def predict(
        self, context: object, model_input: object, params: object = None
    ) -> dict[str, str]:
        """Generate predictions using the RCA agent.

        Args:
            context: MLflow context
            model_input: Input data with 'question' or 'input' field
            params: Optional parameters (session_id, request_id, etc.)

        Returns:
            List of dictionaries with 'response' field
        """
        # Handle different input formats
        if isinstance(model_input, dict):
            if "question" in model_input:
                question = model_input["question"]
            elif "input" in model_input:
                # Handle chat-style input
                messages = model_input["input"]
                if isinstance(messages, list) and len(messages) > 0:
                    question = messages[-1].get("content", "")
                else:
                    question = str(messages)
            else:
                raise ValueError("Input must contain 'question' or 'input' field")
        else:
            question = str(model_input)

        # Get response from agent
        response = self.agent.chat(question)

        return {"response": response}
