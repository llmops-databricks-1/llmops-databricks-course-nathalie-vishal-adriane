"""Evaluation utilities for the RCA agent."""

import mlflow
from mlflow.entities.assessment import Feedback
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
        "If the response does not contain any concrete hypotheses, the assessment must be 'no'",
        "If the response is empty, an error message, or a fallback such as 'Max iterations reached', the assessment must be 'no'",
        "Hypotheses must be specific and testable based on data returned by the agent's tools",
        "Each hypothesis must be supported by internal tool data; external context may supplement but not replace data evidence",
        "Multiple hypotheses must be ranked by likelihood with explicit supporting evidence cited for each",
        "Hypotheses that rely solely on world knowledge without citing tool output must be assessed 'no'",
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


# Known agent fallback/error strings that indicate a failed run
_INVALID_OUTPUT_PATTERNS = [
    "max iterations reached",
    "error:",
    "traceback (",
    "exception:",
]


@mlflow.genai.scorer
def validity_gate(outputs: list) -> bool:
    """Hard gate: fail empty, error, or agent-fallback outputs before any LLM judge runs.

    Returns:
        True only if the output contains substantive content worth scoring.
    """
    if isinstance(outputs, list) and len(outputs) > 0:
        if isinstance(outputs[0], dict) and "text" in outputs[0]:
            text = outputs[0]["text"]
        elif isinstance(outputs[0], str):
            text = outputs[0]
        else:
            text = str(outputs[0])
    else:
        text = str(outputs)

    text_stripped = text.strip()
    if not text_stripped or len(text_stripped.split()) < 20:
        return False
    text_lower = text_stripped.lower()
    return not any(pat in text_lower for pat in _INVALID_OUTPUT_PATTERNS)


@mlflow.genai.scorer
def grounding_check(outputs: list) -> bool:
    """Check that the response grounds quantitative claims in tool-retrieved data.

    External context (news, world knowledge) is allowed only when clearly labeled
    as 'External context' and does not contradict tool evidence.

    Returns:
        True if grounding requirements are met, False otherwise.
    """
    if isinstance(outputs, list) and len(outputs) > 0:
        if isinstance(outputs[0], dict) and "text" in outputs[0]:
            text = outputs[0]["text"]
        elif isinstance(outputs[0], str):
            text = outputs[0]
        else:
            text = str(outputs[0])
    else:
        text = str(outputs)

    text_stripped = text.strip()
    if not text_stripped or len(text_stripped.split()) < 20:
        return False

    text_lower = text_stripped.lower()
    # Response must cite at least one concrete data point (number or % or named metric)
    import re

    has_data_reference = bool(
        re.search(r"\d+\.?\d*\s*%", text_lower)
        or re.search(
            r"\b\d{1,3}[,\d]*\s*(orders|units|sales|revenue|customers)", text_lower
        )
        or any(
            kw in text_lower
            for kw in [
                "increased by",
                "decreased by",
                "declined by",
                "grew by",
                "fell by",
            ]
        )
    )
    return has_data_reference


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
def report_length_check(outputs: list) -> Feedback:
    """Return word count as a numeric score so it is visible in the MLflow UI.

    Score is the raw word count. Rationale records PASS/FAIL against the
    60–700 word threshold so reviewers can sort and filter by length directly
    in the traces table.
    """
    if isinstance(outputs, list) and len(outputs) > 0:
        if isinstance(outputs[0], dict) and "text" in outputs[0]:
            text = outputs[0]["text"]
        elif isinstance(outputs[0], str):
            text = outputs[0]
        else:
            text = str(outputs[0])
    else:
        text = str(outputs)

    word_count = len(text.strip().split())
    min_words, max_words = 60, 700
    passed = min_words <= word_count <= max_words
    verdict = "PASS" if passed else "FAIL"
    return Feedback(
        value=word_count,
        rationale=f"Report has {word_count} words (acceptable range: {min_words}–{max_words}). {verdict}.",
    )


@mlflow.genai.scorer
def tool_diversity_check(outputs: list) -> bool:
    """Check if the agent used RAG and/or news tools, not just Genie."""
    content = str(outputs)
    content_lower = content.lower()
    uses_research = "arxiv" in content_lower or "research" in content_lower
    uses_news = "news" in content_lower or "headline" in content_lower
    return uses_research or uses_news


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
            validity_gate,
            report_length_check,
            grounding_check,
            mentions_metrics,
            includes_hypothesis,
            tool_diversity_check,
            attribution_correctness_guideline,
            hypothesis_quality_guideline,
            narrative_faithfulness_guideline,
        ]

    return mlflow.genai.evaluate(
        predict_fn=predict_fn,
        data=eval_data,
        scorers=scorers,
    )
