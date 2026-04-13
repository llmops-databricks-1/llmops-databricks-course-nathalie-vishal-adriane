# Order Behavior RCA Agent

The Root Cause Analysis for e-comerce Our project aims to deliver an automated Root Cause Analysis (RCA) engine built on Databricks Genie to detect and explain anomalies in order mix and volume within the Olist e-commerce dataset. By integrating a reasoning agent framework, the system decomposes performance shifts into structured drivers such as lost customers, category mix shifts, or concentration spikes.

---

## What it does

1. **Watcher** — monitors daily order metrics and fires an alert when volume or mix deviates beyond a configurable threshold (simple rule-based detector).
2. **Attribution** — decomposes the anomaly into a waterfall of contributions by dimension (channel, category, region, cohort) using mix-shift and growth accounting.
3. **Hypothesis generator** — an LLM reviews the attribution results, proposes the most likely root-cause hypotheses, and calls data tools to validate each one (e.g. new-article launches, lost-customer segments, concentration spikes).
4. **Narrator** — produces an executive summary with supporting visuals and per-hypothesis confidence scores, plus a prioritised list of follow-up questions.

**LLMOps scope:** a scenario library of known order spikes with ground-truth attributions and an eval harness that scores attribution correctness and narrative faithfulness.

**Dataset:** Brazilian E-Commerce Public Dataset by Olist

It is a Brazilian ecommerce public dataset of orders made at Olist Store. The dataset has information of 100k orders from 2016 to 2018 made at multiple marketplaces in Brazil. Its features allows viewing an order from multiple dimensions: from order status, price, payment and freight performance to customer location, product attributes and finally reviews written by customers. This is real commercial data, it has been anonymised, and references to the companies and partners in the review text have been replaced with the names of Game of Thrones great houses.

---

## Unity Catalog structure

```
xxx
```

Environment-specific targets (`dev`, `acc`, `prd`) are defined in [`project_config.yml`](project_config.yml) and resolved at runtime via the `ProjectConfig` Pydantic model.

---

## Project structure

```
.
├── notebooks/
│   └── hello_world.py                             # Smoke-test notebook
├── resources/
│   └── hello_world_job.yml                        # Databricks Asset Bundle job definition
├── src/
│   └── your_custom_package/
│       ├── __init__.py
│       └── config.py                              # ProjectConfig Pydantic model
├── tests/
│   ├── __init__.py
│   └── test_basic.py
├── databricks.yml                                 # Asset Bundle configuration
├── project_config.yml                             # Per-environment catalog/schema/endpoint config
├── pyproject.toml                                 # Python project & dependency definitions
└── version.txt
```

### Development setup

Requires Python 3.12 and [`uv`](https://docs.astral.sh/uv/getting-started/installation/).

```bash
uv sync --extra dev
```

Run linting and tests:

```bash
uv run pre-commit run --all-files
uv run pytest
```

---

## Agent Components

### 1. Watcher

A rule‑based monitoring component that tracks daily order metrics and triggers the RCA agent when deviations exceed configurable thresholds (e.g., order volume drops or mix shifts).

---

### 2. Planner

An LLM‑based planning step that determines **which analyses and tools to run**, without accessing data directly.

**Typical planning output:**
- Identify affected metrics and dimensions  
- Run attribution analysis  
- Generate candidate hypotheses  
- Validate hypotheses using tools  
- Retrieve contextual evidence if needed  
- Produce a final narrative summary  

---

### 3. Deterministic Data Tools (UC Functions)

To ensure reproducibility and auditability, core analytical logic is implemented as **Unity Catalog (UC) Functions**.

#### Key UC Functions

**`order_metrics_by_dimension`**  
Aggregates order count, GMV, AOV, and unique customers across time windows and dimensions.

**`compute_order_attribution`**  
Performs mix‑shift / growth accounting to decompose metric changes into per‑dimension contributions.

**`customer_flow_analysis`**  
Identifies new, returning, and churned customers between two periods.

These tools provide **ground‑truth numerical evidence** for the agent’s reasoning.

✅ Some UC Functions are registered via **Managed Compute Plane (MCP)** to enable:
- Versioning  
- Reuse across agents  
- Governance and access control  

---

### 4. Retrieval‑Augmented Generation (RAG)

The project includes a **Vector Search database** containing processed documents such as:
- Historical RCA reports  
- Known anomaly explanations  
- Business context notes (campaigns, logistics incidents, holidays)  
- Prior patterns and scenarios  

#### Role of RAG

RAG is used **only after hypotheses are formed**, to:
- Provide historical or contextual evidence  
- Support or challenge hypotheses  
- Enrich the final explanation  

RAG is implemented as a **standard agent tool (non‑MCP)** because:
- Retrieval is agent‑specific and session‑scoped  
- It does not represent reusable business logic  
- Latency and flexibility are prioritized  

---

### 5. Hypothesis Validation and Confidence Scoring

For each hypothesis, the agent:
- Calls deterministic tools to validate impact  
- Evaluates consistency across dimensions  
- Assigns a confidence score based on evidence strength  

---

### 6. Narrator

The Narrator produces an **executive‑friendly RCA report**, including:
- Summary of the anomaly  
- Key drivers and quantitative contributions  
- Contextual evidence from retrieved documents  
- Confidence‑scored hypotheses  
- Suggested follow‑up questions  

---

## LLMOps Design Principles

This project explicitly follows modern **LLMOps best practices**:

- ✅ **Reproducibility** — deterministic computations via UC Functions  
- ✅ **Traceability** — tool calls and outputs logged with MLflow Tracing  
- ✅ **Separation of concerns** — math and data logic outside the LLM  
- ✅ **Evaluation readiness** — attribution results compared to scenario ground truth  
- ✅ **Governance** — selective use of MCP for stable business logic  

---

## MCP vs Non‑MCP Design Choices

| Component | MCP | Rationale |
|--------|-----|-----------|
| Metric aggregation UC Functions | Optional | Reusable business logic |
| Attribution UC Functions | ✅ Yes | Critical, versioned logic |
| Customer flow analysis | Optional | Domain primitive |
| Vector Search RAG tool | ❌ No | Session‑scoped retrieval |
| Hypothesis ranking logic | ❌ No | Agent‑specific reasoning |