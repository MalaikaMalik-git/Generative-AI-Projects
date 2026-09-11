"""
Session 2 - Scraper.

Fetches pages from agentixsystem.com, strips nav/script/style noise, and
saves clean plain text into data/raw/<page>.txt.

Already-scraped content is included in data/raw/ — you don't have to
re-run this to get started. Re-run it if you want to add more pages
(e.g. individual /agents/... detail pages) or refresh content.

Note: shammarianas.com was dropped from this project's RAG source — it's a
JS-rendered site with no static content to scrape, so agentixsystem.com is
the single source of truth for company knowledge.

Usage:
    python scripts/scrape.py
"""

import os
import re
import time
from urllib.parse import urlparse
from typing import Optional
import requests
from bs4 import BeautifulSoup

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")

# Add or remove pages here as needed. The 4 individual /agents/... detail
# pages linked from the products page are good candidates to add if you
# want richer per-agent content and have a few spare minutes.
PAGES_TO_SCRAPE = [
    "https://agentixsystem.com/",
    "https://agentixsystem.com/about",
    "https://agentixsystem.com/products",
]

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; AvatarAssistantBot/1.0; +for RAG ingestion)"
}


def fetch_page(url: str) -> Optional[str]:
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        resp.raise_for_status()
        return resp.text
    except requests.RequestException as exc:
        print(f"  [FAILED] {url} -> {exc}")
        return None


def clean_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")

    for tag in soup(["script", "style", "noscript", "svg", "form", "nav"]):
        tag.decompose()

    text = soup.get_text(separator="\n")

    lines = [line.strip() for line in text.splitlines()]
    lines = [line for line in lines if line]
    cleaned = "\n".join(lines)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned


def page_slug(url: str) -> str:
    parsed = urlparse(url)
    domain = parsed.netloc.replace("www.", "").split(".")[0]
    path = parsed.path.strip("/").replace("/", "_") or "home"
    return f"{domain}_{path}"


def looks_like_empty_shell(text: str) -> bool:
    """Heuristic: JS-only pages fetch as a handful of words (loading
    spinners, tracking pixels) rather than real page content."""
    return len(text.split()) < 30


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for url in PAGES_TO_SCRAPE:
        print(f"Scraping {url} ...")
        html = fetch_page(url)
        if html is None:
            continue

        text = clean_text(html)
        slug = page_slug(url)
        out_path = os.path.join(OUTPUT_DIR, f"{slug}.txt")

        if looks_like_empty_shell(text):
            print(
                f"  [WARNING] {url} returned almost no text "
                f"({len(text.split())} words) — likely JS-rendered or "
                f"blocked. Skipping so we don't overwrite good content."
            )
            continue

        with open(out_path, "w", encoding="utf-8") as f:
            f.write(f"SOURCE: {url}\n\n{text}")
        print(f"  Saved -> {out_path} ({len(text.split())} words)")

        time.sleep(1)  # be polite between requests

    print("\nDone. Review files in data/raw/ before running ingest.py.")


if __name__ == "__main__":
    main()
