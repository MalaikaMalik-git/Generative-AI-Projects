"""
tools/fetch.py
Fetches a web page and returns clean plain text.
Uses regex instead of BeautifulSoup to avoid Python 3.9 compatibility issues.
"""
from __future__ import annotations
import re
import requests
from agent.config import REQUEST_TIMEOUT

_MAX_CHARS = 2000


def fetch_page(url: str) -> str:
    """Fetch a URL and return plain text (max 2000 chars). Never raises."""
    if not url or not url.startswith("http"):
        return f"Error: invalid URL '{url}'"
    try:
        headers = {"User-Agent": "Mozilla/5.0 (compatible; ResearchAgent/1.0)"}
        resp = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()

        # Strip HTML with regex — no BeautifulSoup needed
        text = resp.text
        text = re.sub(r'<script[^>]*>.*?</script>', ' ', text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'<style[^>]*>.*?</style>',  ' ', text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'<[^>]+>', ' ', text)
        text = re.sub(r'&\w+;', ' ', text)       # HTML entities
        text = re.sub(r'\s+', ' ', text).strip()

        if not text:
            return "Page fetched but no text content found."
        if len(text) > _MAX_CHARS:
            text = text[:_MAX_CHARS] + "... [truncated]"
        return text

    except requests.exceptions.Timeout:
        return f"Error: request to {url} timed out after {REQUEST_TIMEOUT}s."
    except requests.exceptions.HTTPError as e:
        return f"Error: HTTP {e.response.status_code} fetching {url}."
    except requests.exceptions.ConnectionError:
        return f"Error: could not connect to {url}."
    except Exception as e:
        return f"Error fetching {url}: {str(e)}"