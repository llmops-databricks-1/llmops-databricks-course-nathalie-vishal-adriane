# Databricks notebook source
# DBTITLE 1,Sales Order RCA Agent
# MAGIC %md
# MAGIC ## Sales Order Root Cause Analysis Agent
# MAGIC
# MAGIC This notebook implements an automated RCA agent that:
# MAGIC 1. **Queries your Genie space** to pull sales order and e-commerce data based on your question
# MAGIC 2. **Generates hypotheses** using an LLM about what might be driving observed anomalies
# MAGIC 3. **Scrapes news headlines** for external context (economic, supply chain, labor, regulatory, natural events)
# MAGIC 4. **Produces a structured report** combining data insights with external evidence
# MAGIC
# MAGIC **Usage:** Enter your question in the widget at the bottom (e.g. *"Compare sales behaviour from March 2018 to May 2018"*) and run the agent cell.

# COMMAND ----------

# DBTITLE 1,Setup and Configuration
# MAGIC %pip install feedparser databricks-openai --quiet
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Imports and Constants
import json
import re
import time
import urllib.parse

import feedparser
import requests
from databricks_openai import DatabricksOpenAI

# ── Configuration ────────────────────────────────────────────────────────────
GENIE_SPACE_ID = "01f122d81aa51743bbd0357323432b75"
LLM_ENDPOINT = "databricks-claude-sonnet-4-5"  # Change to any Foundation Model endpoint

# Databricks auth (auto-configured in notebooks)
DATABRICKS_HOST = (
    dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().get()
)
DATABRICKS_TOKEN = (
    dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
)

HEADERS = {
    "Authorization": f"Bearer {DATABRICKS_TOKEN}",
    "Content-Type": "application/json",
}

llm_client = DatabricksOpenAI()

print(f"Host:  {DATABRICKS_HOST}")
print(f"Genie: {GENIE_SPACE_ID}")
print(f"LLM:   {LLM_ENDPOINT}")

# COMMAND ----------


# DBTITLE 1,Genie Space Query Helper
def genie_ask(question: str, space_id: str = GENIE_SPACE_ID, timeout: int = 120) -> dict:
    """
    Send a question to the Genie space, poll until completion, and return
    the response text, generated SQL, and query results.
    """
    base = f"{DATABRICKS_HOST}/api/2.0/genie/spaces/{space_id}"

    # 1. Start a conversation
    resp = requests.post(
        f"{base}/start-conversation",
        headers=HEADERS,
        json={"content": question},
    )
    resp.raise_for_status()
    payload = resp.json()
    conv_id = payload["conversation_id"]
    msg_id = payload["message_id"]
    print(f"  Genie conversation {conv_id[:8]}\u2026 started")

    # 2. Poll until message is COMPLETED or times out
    poll_url = f"{base}/conversations/{conv_id}/messages/{msg_id}"
    deadline = time.time() + timeout
    status = "SUBMITTED"
    message = {}
    while time.time() < deadline:
        r = requests.get(poll_url, headers=HEADERS)
        r.raise_for_status()
        message = r.json()
        status = message.get("status", "UNKNOWN")
        if status in ("COMPLETED", "FAILED"):
            break
        time.sleep(3)

    if status == "FAILED":
        return {
            "error": message.get("error", "Genie query failed"),
            "sql": None,
            "data": None,
            "text": None,
        }

    # 3. Extract response text and SQL from attachments
    attachments = message.get("attachments", []) or []
    response_text = None
    sql_query = None
    query_data = None

    for att in attachments:
        if att.get("text") and att["text"].get("content"):
            response_text = att["text"]["content"]
        if att.get("query"):
            sql_query = att["query"].get("query", "")
            att_id = att.get("attachment_id")  # API uses "attachment_id", not "id"
            # 4. Fetch the query results
            if att_id:
                result_url = f"{base}/conversations/{conv_id}/messages/{msg_id}/attachments/{att_id}/query-result"
                rr = requests.get(result_url, headers=HEADERS)
                if rr.status_code == 200:
                    raw = rr.json()
                    # Data is nested under statement_response
                    stmt = raw.get("statement_response", {})
                    columns = (
                        stmt.get("manifest", {}).get("schema", {}).get("columns", [])
                    )
                    data_array = stmt.get("result", {}).get("data_array", [])
                    query_data = {
                        "columns": columns,
                        "data_array": data_array,
                    }

    row_count = len(query_data["data_array"]) if query_data else 0
    print(
        f"  Genie status: {status} | SQL: {'yes' if sql_query else 'no'} | Data rows: {row_count}"
    )
    return {
        "text": response_text,
        "sql": sql_query,
        "data": query_data,
        "error": None,
    }


# COMMAND ----------

# DBTITLE 1,News Scraper for External Context
RCA_KEYWORDS = {
    "economic": [
        "inflation",
        "recession",
        "interest rate hike",
        "currency devaluation",
        "economic crisis",
    ],
    "supply_chain": [
        "supply chain disruption",
        "shipping delay",
        "port congestion",
        "raw material shortage",
        "logistics crisis",
    ],
    "labor": [
        "strike",
        "labor strike",
        "worker protest",
        "union strike",
        "workforce shortage",
    ],
    "regulatory": [
        "trade tariff",
        "import ban",
        "sanctions",
        "new regulation",
        "tax increase",
    ],
    "natural": ["flood", "drought", "hurricane", "earthquake", "wildfire"],
}


def scrape_news_context(
    search_term: str, year: int, month: int, categories: list = None, top_n: int = 2
) -> dict:
    """
    Scrape Google News for headlines that may explain sales anomalies.
    Returns results grouped by RCA category.
    """
    start_date = f"{year}-{month:02d}-01"
    end_date = f"{year}-{month:02d}-30"
    selected = categories or list(RCA_KEYWORDS.keys())
    results = {}

    for cat in selected:
        cat_results = []
        for kw in RCA_KEYWORDS.get(cat, []):
            query = f"{search_term} {kw} after:{start_date} before:{end_date}"
            encoded = urllib.parse.quote(query)
            rss_url = (
                f"https://news.google.com/rss/search?q={encoded}&hl=en&gl=US&ceid=US:en"
            )
            feed = feedparser.parse(rss_url)
            cat_results.extend(
                {"keyword": kw, "title": e.title, "link": e.link}
                for e in feed.entries[:top_n]
            )
        if cat_results:
            results[cat] = cat_results
    return results


def format_news_summary(news: dict) -> str:
    """Format scraped news into a readable summary for the LLM."""
    if not news:
        return "No relevant news found for this period."
    lines = []
    for cat, articles in news.items():
        lines.append(f"\n### {cat.replace('_', ' ').title()} ({len(articles)} articles)")
        for a in articles:
            lines.append(f"  - [{a['keyword']}] {a['title']}")
    return "\n".join(lines)


# COMMAND ----------


# DBTITLE 1,LLM Helper Functions
def llm_chat(system_prompt: str, user_prompt: str, max_tokens: int = 4096) -> str:
    """Call the Foundation Model endpoint and return the text response."""
    response = llm_client.chat.completions.create(
        model=LLM_ENDPOINT,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=max_tokens,
        temperature=0.3,
    )
    return response.choices[0].message.content


def generate_genie_questions(user_question: str) -> list:
    """
    Given the user's high-level question, generate a list of specific
    data questions to ask the Genie space.
    """
    system = """You are a data analyst assistant. Given a user's root cause analysis question
about sales orders or e-commerce, generate 3-5 specific analytical questions to query
a sales database. Each question should help investigate different dimensions:
- Time trends (monthly/weekly aggregations)
- Product or category breakdowns
- Regional or geographic patterns
- Customer segment analysis
- Order volume vs revenue vs average order value

Return ONLY a JSON array of strings, no other text."""

    raw = llm_chat(system, f"User question: {user_question}")
    # Extract JSON array from response
    try:
        questions = json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\[.*\]", raw, re.DOTALL)
        questions = json.loads(match.group()) if match else [user_question]
    return questions


def generate_hypotheses(user_question: str, data_summary: str) -> dict:
    """
    Analyze the data retrieved from Genie and generate RCA hypotheses.
    Returns a dict with hypotheses and suggested news search terms.
    """
    system = """You are a senior business analyst specializing in root cause analysis
for sales and e-commerce anomalies. Given data from a sales database, generate:
1. A list of hypotheses that could explain the observed patterns
2. For each hypothesis, suggest a search term to look for confirming news/events
3. Identify the relevant time periods (year, month) to search

Return a JSON object with this structure:
{
  "data_insights": "Brief summary of what the data shows",
  "hypotheses": [
    {
      "hypothesis": "Description of the hypothesis",
      "search_term": "keyword to search for in news",
      "year": 2018,
      "month": 5,
      "categories": ["economic", "supply_chain"]
    }
  ]
}
Return ONLY valid JSON."""

    prompt = (
        f"User question: {user_question}\n\nData from sales database:\n{data_summary}"
    )
    raw = llm_chat(system, prompt)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        return (
            json.loads(match.group())
            if match
            else {"data_insights": raw, "hypotheses": []}
        )


def generate_report(
    user_question: str, data_summary: str, hypotheses: dict, news_summary: str
) -> str:
    """
    Produce the final RCA report combining all evidence.
    """
    system = """You are a senior business analyst writing a Root Cause Analysis report.
Combine the sales data analysis, hypotheses, and external news evidence into a clear,
actionable report. Structure it as:

1. **Executive Summary** - What was the question and key finding
2. **Data Analysis** - What the numbers show (with specific figures)
3. **Hypotheses & Evidence** - Each hypothesis with supporting/contradicting news
4. **Most Likely Root Causes** - Ranked by evidence strength
5. **Recommended Actions** - What the business should do next

Be specific, cite data points, and clearly distinguish between confirmed and speculative causes."""

    prompt = f"""## Original Question
{user_question}

## Sales Data Analysis
{data_summary}

## Hypotheses
{json.dumps(hypotheses, indent=2)}

## External News & Events
{news_summary}"""

    return llm_chat(system, prompt, max_tokens=6000)


# COMMAND ----------


# DBTITLE 1,RCA Agent Orchestrator
def format_genie_data(genie_response: dict) -> str:
    """Convert Genie query results into a readable text summary for the LLM."""
    if not genie_response or genie_response.get("error"):
        return f"Error: {genie_response.get('error', 'No data')}"

    parts = []
    if genie_response.get("text"):
        parts.append(f"Genie says: {genie_response['text']}")
    if genie_response.get("sql"):
        parts.append(f"SQL: {genie_response['sql']}")
    if genie_response.get("data"):
        data = genie_response["data"]
        columns = [
            c.get("name", f"col{i}") for i, c in enumerate(data.get("columns", []))
        ]
        rows = data.get("data_array", [])
        parts.append(f"Columns: {', '.join(columns)}")
        parts.append(f"Rows returned: {len(rows)}")
        # Include first 20 rows as a table
        if rows:
            header = " | ".join(columns)
            parts.append(header)
            parts.append("-" * len(header))
            for row in rows[:20]:
                parts.append(" | ".join(str(v) for v in row))
            if len(rows) > 20:
                parts.append(f"... ({len(rows) - 20} more rows)")
    return "\n".join(parts)


def run_rca_agent(user_question: str) -> str:
    """
    Main orchestrator: takes the user's question and produces an RCA report.
    """
    print("=" * 70)
    print(f"RCA AGENT: {user_question}")
    print("=" * 70)

    # ── Step 1: Break the question into specific data queries ─────────────
    print("\n\U0001f50d Step 1: Generating analytical questions...")
    genie_questions = generate_genie_questions(user_question)
    for i, q in enumerate(genie_questions, 1):
        print(f"  Q{i}: {q}")

    # ── Step 2: Query the Genie space for each question ─────────────────
    print("\n\U0001f4ca Step 2: Querying Genie space for data...")
    all_data_summaries = []
    for i, q in enumerate(genie_questions, 1):
        print(f"\n  Asking Q{i}: {q}")
        try:
            result = genie_ask(q)
            summary = format_genie_data(result)
            all_data_summaries.append(f"### Query {i}: {q}\n{summary}")
        except Exception as e:
            all_data_summaries.append(f"### Query {i}: {q}\nError: {str(e)}")
            print(f"  Error: {e}")

    combined_data = "\n\n".join(all_data_summaries)

    # ── Step 3: Generate hypotheses from the data ─────────────────────
    print("\n\U0001f9e0 Step 3: Generating hypotheses...")
    hypotheses = generate_hypotheses(user_question, combined_data)
    print(f"  Data insights: {hypotheses.get('data_insights', 'N/A')[:100]}...")
    for i, h in enumerate(hypotheses.get("hypotheses", []), 1):
        print(f"  H{i}: {h['hypothesis'][:80]}...")

    # ── Step 4: Scrape news for each hypothesis ───────────────────────
    print("\n\U0001f4f0 Step 4: Scraping news for external evidence...")
    all_news = {}
    for h in hypotheses.get("hypotheses", []):
        search_term = h.get("search_term", "")
        year = h.get("year", 2018)
        month = h.get("month", 1)
        cats = h.get("categories")
        # Only use categories that exist in RCA_KEYWORDS
        valid_cats = [c for c in (cats or []) if c in RCA_KEYWORDS] or None
        print(
            f"  Searching: '{search_term}' ({year}-{month:02d}), categories={valid_cats}"
        )
        try:
            news = scrape_news_context(
                search_term, year, month, categories=valid_cats, top_n=2
            )
            for cat, articles in news.items():
                all_news.setdefault(cat, []).extend(articles)
        except Exception as e:
            print(f"  News scrape error: {e}")

    news_text = format_news_summary(all_news)
    print(f"  Total news articles collected: {sum(len(v) for v in all_news.values())}")

    # ── Step 5: Generate the final RCA report ────────────────────────
    print("\n\U0001f4dd Step 5: Generating final report...")
    report = generate_report(user_question, combined_data, hypotheses, news_text)

    print("\n" + "=" * 70)
    print("REPORT COMPLETE")
    print("=" * 70)
    return report


# COMMAND ----------

# DBTITLE 1,Run the RCA Agent
# ── Enter your question here ─────────────────────────────────────────────────────
user_question = "Compare sales behaviour from March 2018 to May 2018"
# ─────────────────────────────────────────────────────────────────────

report = run_rca_agent(user_question)

# COMMAND ----------

# DBTITLE 1,Display Report
import re as _re


def _md_to_html(text):
    """Lightweight markdown to HTML for report rendering."""
    h = text
    h = _re.sub(r"^##### (.+)$", r"<h5>\1</h5>", h, flags=_re.MULTILINE)
    h = _re.sub(r"^#### (.+)$", r"<h4>\1</h4>", h, flags=_re.MULTILINE)
    h = _re.sub(r"^### (.+)$", r"<h3>\1</h3>", h, flags=_re.MULTILINE)
    h = _re.sub(r"^## (.+)$", r"<h2>\1</h2>", h, flags=_re.MULTILINE)
    h = _re.sub(r"^# (.+)$", r"<h1>\1</h1>", h, flags=_re.MULTILINE)
    h = _re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", h)
    h = _re.sub(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", r"<em>\1</em>", h)
    h = _re.sub(r"^[\-\*] (.+)$", r"<li>\1</li>", h, flags=_re.MULTILINE)
    h = _re.sub(r"((?:<li>.*?</li>\n?)+)", r"<ul>\1</ul>", h)
    h = _re.sub(r"^\d+\.\s+(.+)$", r"<li>\1</li>", h, flags=_re.MULTILINE)
    h = _re.sub(r"^---+$", "<hr>", h, flags=_re.MULTILINE)
    h = _re.sub(r"\n{2,}", "</p><p>", h)
    h = _re.sub(r"(?<!>)\n(?!<)", "<br>", h)
    return f"<p>{h}</p>"


report_html = _md_to_html(report)
displayHTML(f"""
<div style="max-width:900px; font-family:sans-serif; line-height:1.6; padding:20px;">
  <style>
    h1 {{ color:#1B3A5C; border-bottom:2px solid #1B3A5C; padding-bottom:8px; }}
    h2 {{ color:#2E6B9E; margin-top:24px; }}
    h3 {{ color:#3A7CA5; }}
    strong {{ color:#333; }}
    ul {{ margin:8px 0; padding-left:24px; }}
    li {{ margin:4px 0; }}
    hr {{ border:none; border-top:1px solid #ccc; margin:16px 0; }}
  </style>
  {report_html}
</div>
""")

# COMMAND ----------

print(f"report type: {type(report).__name__}")
print(f"report length: {len(report)}")
print(report)


# COMMAND ----------

genie_questions

# COMMAND ----------
