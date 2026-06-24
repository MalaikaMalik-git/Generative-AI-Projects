"""
tools/search.py
Real web search using DuckDuckGo (ddgs library — no API key needed).
Replaces the stub in react_loop.py.
"""
from __future__ import annotations
import json
import time
from ddgs import DDGS
from agent.config import MAX_SEARCH_RESULTS


def web_search(query: str) -> str:
    """
    Search DuckDuckGo and return top results as a JSON string.

    Returns a JSON array of {title, url, body} dicts.
    Falls back to an error message if search fails.
    """
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=MAX_SEARCH_RESULTS))

        if not results:
            # DDG occasionally returns empty — retry once with a small delay
            time.sleep(2)
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=MAX_SEARCH_RESULTS))

        if not results:
            return json.dumps([{
                "title": "No results found",
                "url":   "",
                "body":  f"DuckDuckGo returned no results for: '{query}'. Try rephrasing.",
            }])

        # Normalise field names (ddgs uses 'href' not 'url')
        cleaned = [
            {
                "title": r.get("title", ""),
                "url":   r.get("href", r.get("url", "")),
                "body":  r.get("body", ""),
            }
            for r in results
        ]
        return json.dumps(cleaned, ensure_ascii=False)

    except Exception as e:
        return json.dumps([{
            "title": "Search error",
            "url":   "",
            "body":  f"Search failed for '{query}': {str(e)}. Try a different query.",
        }])