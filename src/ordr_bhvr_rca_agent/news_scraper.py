"""News scraping tool for RCA agent to find external context."""

import json
import socket
import urllib.parse

import feedparser
from loguru import logger

# Hard cap on how long a single Google News RSS fetch may block.
_NEWS_FETCH_TIMEOUT_SECONDS = 5

# News categories and keywords for RCA
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


def scrape_news(
    search_term: str,
    year: int,
    month: int,
    categories: list[str] | None = None,
    top_n: int = 2,
) -> str:
    """Scrape Google News for headlines related to potential sales anomaly causes.

    Args:
        search_term: Main search term (e.g., "retail sales", "consumer behavior")
        year: Year to search
        month: Month to search (1-12)
        categories: List of RCA categories (economic, supply_chain, labor,
            regulatory, natural)
        top_n: Number of articles to retrieve per keyword

    Returns:
        JSON string with news articles grouped by category
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

            try:
                old_timeout = socket.getdefaulttimeout()
                socket.setdefaulttimeout(_NEWS_FETCH_TIMEOUT_SECONDS)
                try:
                    feed = feedparser.parse(rss_url)
                finally:
                    socket.setdefaulttimeout(old_timeout)
                cat_results.extend(
                    [
                        {
                            "keyword": kw,
                            "title": e.title,
                            "link": e.link,
                            "published": e.get("published", ""),
                        }
                        for e in feed.entries[:top_n]
                    ]
                )
            except Exception as e:
                logger.warning(f"Error scraping news for {kw}: {e}")

        if cat_results:
            results[cat] = cat_results

    return json.dumps(results, indent=2)


# Tool specification for OpenAI function calling format
SCRAPE_NEWS_TOOL_SPEC = {
    "type": "function",
    "function": {
        "name": "scrape_news",
        "description": (
            "Scrape Google News for headlines that may explain sales anomalies "
            "across economic, supply chain, labor, regulatory, and natural "
            "event categories."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "search_term": {
                    "type": "string",
                    "description": (
                        "Main search term (e.g., 'retail sales', 'e-commerce')"
                    ),
                },
                "year": {
                    "type": "integer",
                    "description": "Year to search for news (e.g., 2018)",
                },
                "month": {"type": "integer", "description": "Month to search (1-12)"},
                "categories": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Optional categories: economic, supply_chain, labor, "
                        "regulatory, natural"
                    ),
                },
                "top_n": {
                    "type": "integer",
                    "description": "Articles per keyword (default: 2)",
                    "default": 2,
                },
            },
            "required": ["search_term", "year", "month"],
        },
    },
}
