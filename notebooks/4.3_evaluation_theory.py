# Databricks notebook source
# MAGIC %md
# MAGIC # Lecture 4.3: GenAI Evaluation Theory for RCA Agents
# MAGIC
# MAGIC ## Topics Covered:
# MAGIC - Why evaluation matters for RCA agents
# MAGIC - Types of evaluation metrics for RCA
# MAGIC - MLflow evaluation framework
# MAGIC - Guidelines vs Judges for RCA
# MAGIC - Custom scorers for attribution and hypotheses
# MAGIC - Judge alignment with human feedback

# COMMAND ----------

# MAGIC %pip install /Workspace/Users/aschelin@gmail.com/.bundle/llmops-databricks-course-nathalie-vishal-adriane/dev/files/
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import os
from typing import Literal

import mlflow
from dotenv import load_dotenv
from loguru import logger
from mlflow.genai.judges import make_judge
from pyspark.sql import SparkSession

from ordr_bhvr_rca_agent.config import get_env, load_config
from ordr_bhvr_rca_agent.evaluation import (
    attribution_correctness_guideline,
    hypothesis_quality_guideline,
    includes_hypothesis,
    mentions_metrics,
    narrative_faithfulness_guideline,
    report_length_check,
)

# COMMAND ----------

# Setup
if "DATABRICKS_RUNTIME_VERSION" not in os.environ:
    load_dotenv()
    profile = os.environ["PROFILE"]
    mlflow.set_tracking_uri(f"databricks://{profile}")
    mlflow.set_registry_uri(f"databricks-uc://{profile}")

spark = SparkSession.builder.getOrCreate()

# Load configuration
env = get_env(spark)
cfg = load_config("../project_config.yml", env)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Why Evaluation Matters for RCA Agents
# MAGIC
# MAGIC Traditional ML evaluation doesn't work for RCA because:
# MAGIC
# MAGIC ### Challenges:
# MAGIC - **Subjective hypotheses**: Multiple valid explanations for anomalies
# MAGIC - **Multi-dimensional quality**: Attribution accuracy, hypothesis relevance, narrative clarity
# MAGIC - **Context-dependent**: Same metric drop may have different root causes
# MAGIC - **Data grounding**: Must balance quantitative evidence with qualitative insights
# MAGIC
# MAGIC ### Why Evaluate RCA Agents?
# MAGIC 1. **Quality assurance**: Ensure attributions are mathematically correct
# MAGIC 2. **Regression detection**: Catch degradation in hypothesis quality
# MAGIC 3. **Model comparison**: Choose the best LLM for hypothesis generation
# MAGIC 4. **Continuous improvement**: Identify weak reasoning patterns
# MAGIC 5. **Trust & reliability**: Validate agent findings before business decisions

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Types of Evaluation Metrics for RCA
# MAGIC
# MAGIC ### A. Automated Metrics (Quantitative)
# MAGIC
# MAGIC | Metric | What it Measures | Use Case |
# MAGIC |--------|-----------------|----------|
# MAGIC | **Attribution Accuracy** | Waterfall math correctness | Decomposition validation |
# MAGIC | **Metric Delta** | Actual vs. computed variance | Data integrity check |
# MAGIC | **Report Length** | Word count in range | Conciseness check |
# MAGIC | **Metric Mention** | Presence of quantitative data | Data-grounding check |
# MAGIC | **Hypothesis Count** | Number of hypotheses generated | Completeness check |
# MAGIC
# MAGIC ### B. LLM-as-Judge Metrics (Qualitative)
# MAGIC
# MAGIC | Metric | What it Measures | Use Case |
# MAGIC |--------|-----------------|----------|
# MAGIC | **Attribution Correctness** | Alignment with decomposition data | RCA accuracy |
# MAGIC | **Hypothesis Quality** | Testability and evidence support | Reasoning quality |
# MAGIC | **Narrative Faithfulness** | Grounded in quantitative findings | Trustworthiness |
# MAGIC | **Ranking Quality** | Hypotheses ranked by likelihood | Decision support |
# MAGIC | **External Context Integration** | News/events properly incorporated | Comprehensiveness |
# MAGIC
# MAGIC ### C. Human Evaluation (Ground Truth)
# MAGIC
# MAGIC - Domain experts validate root causes
# MAGIC - Used to validate automated metrics
# MAGIC - Essential for high-stakes business decisions

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. MLflow Evaluation Framework for RCA
# MAGIC
# MAGIC MLflow provides a comprehensive framework for RCA evaluation:
# MAGIC
# MAGIC ```python
# MAGIC results = mlflow.genai.evaluate(
# MAGIC     data=rca_eval_data,          # Test anomaly scenarios
# MAGIC     predict_fn=rca_agent,        # RCA agent to evaluate
# MAGIC     scorers=[scorer1, scorer2],  # RCA-specific metrics
# MAGIC )
# MAGIC ```
# MAGIC
# MAGIC ### Key Components:
# MAGIC 1. **Data**: Anomaly scenarios with known root causes
# MAGIC 2. **Predict Function**: Your RCA agent
# MAGIC 3. **Scorers**: Metrics to evaluate attribution and hypotheses
# MAGIC 4. **Results**: Detailed evaluation results with trace data

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Guidelines for RCA - Binary Pass/Fail

# COMMAND ----------

# Set experiment
mlflow.set_experiment(cfg.experiment_name)

# Using pre-defined Guidelines scorers from ordr_bhvr_rca_agent.evaluation
logger.info("RCA Evaluation Guidelines from ordr_bhvr_rca_agent.evaluation:")
logger.info("  1. attribution_correctness_guideline: Validates attribution math")
logger.info("  2. hypothesis_quality_guideline: Checks testability and evidence")
logger.info("  3. narrative_faithfulness_guideline: Ensures data-grounded narrative")
logger.info("  4. scope_guideline: Ensures RCA stays focused")
logger.info("Custom Scorers:")
logger.info("  5. report_length_check: 100-500 words")
logger.info("  6. mentions_metrics: Checks for quantitative data")
logger.info("  7. includes_hypothesis: Checks for clear hypotheses")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Test RCA Guidelines

# COMMAND ----------

# Test data with good and bad RCA responses
test_data = [
    {
        "inputs": {"question": "Why did orders decline in May 2018?"},
        "outputs": "Orders went down because of stuff. Maybe customers didn't like it.",
    },
    {
        "inputs": {"question": "Why did orders decline in May 2018?"},
        "outputs": "Orders declined 17.3% from March to May 2018. Attribution analysis shows: 86.7% volume contribution (-1300 orders) and 13.3% mix contribution. Top declining categories: Electronics (-25%), Home & Garden (-20%). Hypothesis: Seasonal demand decline combined with supply chain disruptions in Brazil (news: shipping delays reported in April 2018).",
    },
]

# Evaluate
results = mlflow.genai.evaluate(
    data=test_data,
    scorers=[
        attribution_correctness_guideline,
        hypothesis_quality_guideline,
        narrative_faithfulness_guideline,
    ],
)

logger.info("RCA Evaluation Results:")
logger.info("=" * 80)
display(results)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Judges for RCA - Scored Evaluation

# COMMAND ----------

# Create a Judge for hypothesis ranking quality
hypothesis_ranking_judge = make_judge(
    name="hypothesis_ranking_quality",
    instructions=(
        "Evaluate the quality of hypothesis ranking in {{ outputs }} for the anomaly in {{ inputs }}. "
        "Score from 1 to 5:\n"
        "1 - No ranking or random order\n"
        "2 - Some ranking but not based on evidence\n"
        "3 - Adequate ranking with some justification\n"
        "4 - Good ranking based on data and external context\n"
        "5 - Excellent ranking with clear likelihood reasoning and supporting evidence"
    ),
    model=f"databricks:/{cfg.llm_endpoint}",
    feedback_value_type=int,
)

logger.info("RCA Judge Scorer Created:")
logger.info(f"  Name: {hypothesis_ranking_judge.name}")
logger.info("  Type: Scored (1-5)")
logger.info(f"  Judge Model: {cfg.llm_endpoint}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Test RCA Judge

# COMMAND ----------

# Test data with varying hypothesis quality
judge_test_data = [
    {
        "inputs": {"question": "Why did orders decline in May 2018?"},
        "outputs": "Possible reasons: 1. Weather 2. Economy 3. Competition 4. Random chance",
    },
    {
        "inputs": {"question": "Why did orders decline in May 2018?"},
        "outputs": "Ranked hypotheses (by likelihood):\n1. Seasonal demand decline (87% probability): Historical data shows May has 15-20% lower volumes. Volume contribution accounts for 86.7% of decline.\n2. Supply chain disruptions (68% probability): News reports of Brazil shipping delays in April. Home & Garden category (-20%) aligns with import dependency.\n3. Competitor campaigns (42% probability): No evidence in news scraping, but Electronics category (-25%) suggests possible price competition.",
    },
]

# Evaluate
judge_results = mlflow.genai.evaluate(
    data=judge_test_data, scorers=[hypothesis_ranking_judge]
)

logger.info("Judge Evaluation Results:")
logger.info("=" * 80)
display(judge_results.tables["eval_results"])

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Custom Code-Based Scorers for RCA

# COMMAND ----------


@mlflow.genai.scorer
def attribution_math_correctness(outputs: list, inputs: list = None) -> bool:
    """Check that attribution percentages sum to ~100%."""
    text = outputs[0].get("text", "") if isinstance(outputs[0], dict) else str(outputs[0])

    # Extract percentages (simple pattern matching)
    import re

    percentages = re.findall(r"(\d+(?:\.\d+)?)\s*%", text)

    if not percentages:
        return False  # No percentages found

    total = sum(float(p) for p in percentages[:3])  # First 3 percentages
    return 95 <= total <= 105  # Allow 5% tolerance


@mlflow.genai.scorer
def includes_external_context(outputs: list) -> bool:
    """Check if output includes external context (news, events)."""
    text = outputs[0].get("text", "") if isinstance(outputs[0], dict) else str(outputs[0])
    keywords = [
        "news",
        "headline",
        "event",
        "economic",
        "supply chain",
        "labor",
        "regulation",
    ]
    return any(kw in text.lower() for kw in keywords)


@mlflow.genai.scorer
def hypothesis_completeness_score(outputs: list) -> float:
    """Score based on hypothesis count (3-5 is ideal)."""
    text = outputs[0].get("text", "") if isinstance(outputs[0], dict) else str(outputs[0])

    # Count numbered hypotheses
    import re

    hypotheses = re.findall(r"^\s*\d+\.\s+", text, re.MULTILINE)
    count = len(hypotheses)

    if 3 <= count <= 5:
        return 1.0
    elif count < 3:
        return count / 3.0
    else:
        return max(0.0, 1.0 - (count - 5) / 5.0)


logger.info("Custom RCA Scorers Created:")
logger.info("  1. attribution_math_correctness (boolean)")
logger.info("  2. includes_external_context (boolean)")
logger.info("  3. hypothesis_completeness_score (float 0-1)")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Test Custom RCA Scorers

# COMMAND ----------

custom_test_data = [
    {
        "inputs": {"question": "Analyze May 2018 order decline"},
        "outputs": "Volume contribution: 86.7%, Mix contribution: 13.3%. Total decline: -17.3%. Hypotheses: 1. Seasonal pattern 2. Supply issues (news: shipping delays) 3. Competition",
    },
    {
        "inputs": {"question": "Analyze May 2018 order decline"},
        "outputs": "Orders declined. Could be many reasons. Hard to say without more data.",
    },
]

# Evaluate with custom scorers
custom_results = mlflow.genai.evaluate(
    data=custom_test_data,
    scorers=[
        attribution_math_correctness,
        includes_external_context,
        hypothesis_completeness_score,
    ],
)

logger.info("Custom Scorer Results:")
logger.info("=" * 80)
display(custom_results.tables["eval_results"])

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Categorical Judges for RCA

# COMMAND ----------

# Judge with categorical output for RCA quality tiers
rca_quality_tier_judge = make_judge(
    name="rca_quality_tier",
    instructions=(
        "Classify the RCA quality in {{ outputs }} into one of these tiers:\n"
        "- 'actionable': Clear, data-driven, with testable hypotheses and supporting evidence\n"
        "- 'informative': Provides insights but lacks depth or supporting evidence\n"
        "- 'insufficient': Vague, unsupported, or missing key elements"
    ),
    feedback_value_type=Literal["actionable", "informative", "insufficient"],
    model=f"databricks:/{cfg.llm_endpoint}",
)

logger.info("Categorical RCA Judge Created:")
logger.info(f"  Name: {rca_quality_tier_judge.name}")
logger.info("  Tiers: actionable, informative, insufficient")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Combining Multiple RCA Scorers

# COMMAND ----------

# Combine different types of scorers
all_rca_scorers = [
    attribution_correctness_guideline,  # Binary guideline
    hypothesis_quality_guideline,  # Binary guideline
    narrative_faithfulness_guideline,  # Binary guideline
    hypothesis_ranking_judge,  # Numeric judge (1-5)
    report_length_check,  # Boolean custom
    mentions_metrics,  # Boolean custom
    includes_hypothesis,  # Boolean custom
    attribution_math_correctness,  # Boolean custom
    hypothesis_completeness_score,  # Float custom (0-1)
    rca_quality_tier_judge,  # Categorical judge
]

comprehensive_test_data = [
    {
        "inputs": {"question": "Explain the order volume decline from March to May 2018"},
        "outputs": "Orders declined 17.3% (from 7500 to 62 00). Attribution: 86.7% volume effect (-1300 orders), 13.3% mix effect (-200 orders due to average price decline from $120 to $115). Ranked hypotheses: 1. Seasonal demand (high likelihood - historical May declines of 15-20%), 2. Supply chain disruptions (medium likelihood - news reports of Brazil shipping delays in April 2018), 3. Competitor activity (lower likelihood - no direct evidence but Electronics category down 25%). Recommendation: Prepare inventory buffers for seasonal patterns and monitor supply chain news.",
    },
]

# Evaluate with all scorers
comprehensive_results = mlflow.genai.evaluate(
    data=comprehensive_test_data, scorers=all_rca_scorers
)

logger.info("Comprehensive RCA Evaluation Results:")
logger.info("=" * 80)
comprehensive_results

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Best Practices for RCA Evaluation

# COMMAND ----------

# MAGIC %md
# MAGIC ### ✅ RCA Evaluation Best Practices:
# MAGIC
# MAGIC 1. **Use multiple scorers** for comprehensive RCA quality assessment
# MAGIC 2. **Combine automated and LLM judges** for balance (math + reasoning)
# MAGIC 3. **Create RCA-specific guidelines** (attribution accuracy, hypothesis quality)
# MAGIC 4. **Validate judges with domain expert feedback** periodically
# MAGIC 5. **Track evaluation metrics over time** for regression detection
# MAGIC 6. **Use appropriate judge models** (strong reasoning for hypothesis evaluation)
# MAGIC 7. **Test edge cases** (small deltas, ambiguous anomalies, conflicting data)
# MAGIC 8. **Document evaluation criteria** with business stakeholders
# MAGIC 9. **Version your evaluation sets** with known root-cause scenarios
# MAGIC 10. **Align judges with business analyst feedback** for production use
# MAGIC 11. **Separate evaluation** for attribution (math) vs. hypotheses (reasoning)
# MAGIC 12. **Include external context quality** in evaluation (news relevance, recency)
# MAGIC
# MAGIC ### ❌ Don't:
# MAGIC 1. Rely on a single metric (e.g., only attribution accuracy)
# MAGIC 2. Use only automated metrics without hypothesis quality checks
# MAGIC 3. Ignore domain expert feedback on hypothesis relevance
# MAGIC 4. Evaluate on too few anomaly scenarios
# MAGIC 5. Forget to version evaluation data with business context
# MAGIC 6. Use the same model for both RCA generation and evaluation

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. RCA Evaluation Workflow

# COMMAND ----------

# MAGIC %md
# MAGIC ### Recommended RCA Evaluation Workflow:
# MAGIC
# MAGIC ```
# MAGIC 1. Define RCA Evaluation Criteria
# MAGIC    ├─ Attribution correctness (math validation)
# MAGIC    ├─ Hypothesis quality (testability, evidence, ranking)
# MAGIC    ├─ Narrative faithfulness (data-grounded claims)
# MAGIC    ├─ External context integration (news relevance)
# MAGIC    └─ Actionability (clear recommendations)
# MAGIC
# MAGIC 2. Create RCA Evaluation Dataset
# MAGIC    ├─ Historical anomalies with known root causes
# MAGIC    ├─ Edge cases (small deltas, ambiguous patterns)
# MAGIC    ├─ Synthetic scenarios (controlled attribution)
# MAGIC    └─ Representative of production anomalies
# MAGIC
# MAGIC 3. Choose RCA Scorers
# MAGIC    ├─ Guidelines for attribution correctness
# MAGIC    ├─ Judges for hypothesis quality and ranking
# MAGIC    ├─ Custom scorers for RCA-specific metrics
# MAGIC    └─ Domain expert validation criteria
# MAGIC
# MAGIC 4. Run RCA Evaluation
# MAGIC    ├─ Evaluate baseline agent
# MAGIC    ├─ Evaluate with improved prompts/tools
# MAGIC    └─ Compare different LLM models
# MAGIC
# MAGIC 5. Analyze RCA Results
# MAGIC    ├─ Identify failure patterns (weak hypotheses, wrong attribution)
# MAGIC    ├─ Find improvement opportunities (better news scraping, richer context)
# MAGIC    └─ Validate with domain experts
# MAGIC
# MAGIC 6. Iterate & Deploy
# MAGIC    ├─ Improve agent (prompts, tools, attribution logic)
# MAGIC    ├─ Re-evaluate with full scorer suite
# MAGIC    └─ Deploy if meets quality thresholds
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC In this notebook, you learned:
# MAGIC - Why evaluation is critical for RCA agents
# MAGIC - RCA-specific metrics (attribution correctness, hypothesis quality, narrative faithfulness)
# MAGIC - How to use MLflow evaluation framework for RCA
# MAGIC - Creating custom scorers for RCA use cases
# MAGIC - Combining multiple evaluation approaches
# MAGIC - Best practices for RCA evaluation
# MAGIC
# MAGIC **Next**: [4.4_mlflow_log_register.py](4.4_mlflow_log_register.py) - Logging and registering RCA agents
