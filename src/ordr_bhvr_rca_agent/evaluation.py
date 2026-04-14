"""Evaluation utilities for the RCA agent."""

import mlflow
from mlflow.genai.scorers import Guidelines

attribution_correctness_guideline = Guidelines(
    name="attribution_correctness",
    guidelines=[
        "The analysis must correctly identify the primary drivers of the anomaly based on the data",
        "Attribution percentages should align with the waterfall decomposition results",
        "The response must not make up attribution factors that aren't supported by the data",
    ],
    model="databricks:/databricks-gpt-oss-120b",
)

hypothesis_quality_guideline = Guidelines(
    name="hypothesis_quality",
    guidelines=[
        "Hypotheses must be specific and testable based on available data",
        "Each hypothesis should be supported by either internal data or external context",
        "Multiple hypotheses should be ranked by likelihood and supporting evidence",
    ],
    model="databricks:/databricks-gpt-oss-120b",
)

narrative_faithfulness_guideline = Guidelines(
    name="narrative_faithfulness",
    guidelines=[
        "The narrative must accurately reflect the quantitative findings from the analysis",
        "Claims in the narrative must be supported by the data or cited external sources",
        "The response must not exaggerate or minimize the significance of findings",
    ],
    model="databricks:/databricks-gpt-oss-120b",
)

scope_guideline = Guidelines(
    name="stays_in_scope",
    guidelines=[
        "The response must focus on order behavior analysis and root cause identification",
        "The response should not stray into unrelated business topics",
        "If asked about non-RCA topics, politely redirect to the anomaly investigation",
    ],
    model="databricks:/databricks-gpt-oss-120b",
)


@mlflow.genai.scorer
def mentions_metrics(outputs: list) -> bool:
    """Check if the response mentions specific metrics or data points.

    Args:
        outputs: List of output dictionaries

    Returns:
        True if metrics are mentioned, False otherwise
    """
    # Handle different output formats
    if isinstance(outputs, list) and len(outputs) > 0:
        if isinstance(outputs[0], dict) and "text" in outputs[0]:
            text = outputs[0]["text"]
        elif isinstance(outputs[0], str):
            text = outputs[0]
        else:
            text = str(outputs[0])
    else:
        text = str(outputs)

    text_lower = text.lower()
    keywords = [
        "orders",
        "volume",
        "mix",
        "revenue",
        "attribution",
        "contribution",
        "percent",
        "%",
        "decline",
        "increase",
    ]
    return any(keyword in text_lower for keyword in keywords)


@mlflow.genai.scorer
def includes_hypothesis(outputs: list) -> bool:
    """Check if the response includes a clear hypothesis or explanation.

    Args:
        outputs: List of output dictionaries

    Returns:
        True if hypothesis is present, False otherwise
    """
    # Handle different output formats
    if isinstance(outputs, list) and len(outputs) > 0:
        if isinstance(outputs[0], dict) and "text" in outputs[0]:
            text = outputs[0]["text"]
        elif isinstance(outputs[0], str):
            text = outputs[0]
        else:
            text = str(outputs[0])
    else:
        text = str(outputs)

    text_lower = text.lower()
    keywords = [
        "hypothesis",
        "because",
        "likely",
        "suggests",
        "indicates",
        "root cause",
        "driven by",
        "due to",
        "explanation",
    ]
    return any(keyword in text_lower for keyword in keywords)


@mlflow.genai.scorer
def report_length_check(outputs: list) -> bool:
    """Check that the output is between 100 and 500 words for RCA reports.

    Args:
        outputs: List of output dictionaries

    Returns:
        True if word count is appropriate, False otherwise
    """
    # Handle different output formats
    if isinstance(outputs, list) and len(outputs) > 0:
        if isinstance(outputs[0], dict) and "text" in outputs[0]:
            text = outputs[0]["text"]
        elif isinstance(outputs[0], str):
            text = outputs[0]
        else:
            text = str(outputs[0])
    else:
        text = str(outputs)

    word_count = len(text.split())
    return 100 <= word_count <= 500


def create_eval_data_from_file(eval_inputs_path: str) -> list[dict]:
    """Load evaluation data from a file.

    Args:
        eval_inputs_path: Path to file with one question/scenario per line

    Returns:
        List of evaluation data dictionaries
    """
    with open(eval_inputs_path) as f:
        eval_data = [{"inputs": {"question": line.strip()}} for line in f if line.strip()]
    return eval_data


def evaluate_rca_agent(
    agent, eval_inputs_path: str, scorers: list | None = None
) -> mlflow.models.EvaluationResult:
    """Run evaluation on the RCA agent.

    Args:
        agent: RCA agent instance with predict method
        eval_inputs_path: Path to evaluation inputs file
        scorers: Optional list of custom scorers (defaults to RCA scorers)

    Returns:
        MLflow EvaluationResult with metrics
    """
    eval_data = create_eval_data_from_file(eval_inputs_path)

    def predict_fn(question: str) -> str:
        request = {"input": [{"role": "user", "content": question}]}
        result = agent.predict(request)
        return result.output[-1].content

    if scorers is None:
        scorers = [
            report_length_check,
            mentions_metrics,
            includes_hypothesis,
            attribution_correctness_guideline,
            hypothesis_quality_guideline,
            narrative_faithfulness_guideline,
        ]

    return mlflow.genai.evaluate(
        predict_fn=predict_fn,
        data=eval_data,
        scorers=scorers,
    )
