from __future__ import annotations

import re
from typing import Any
import httpx


def search_duckduckgo(query: str, max_results: int = 3, timeout_seconds: int = 5) -> list[dict[str, str]]:
    """Zero-cost web search using DuckDuckGo Instant Answer API and HTML search without API keys."""
    results: list[dict[str, str]] = []
    clean_query = query.strip()
    if not clean_query:
        return results

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }

    # 1. Try DuckDuckGo Instant Answer API
    try:
        resp = httpx.get(
            "https://api.duckduckgo.com/",
            params={"q": clean_query, "format": "json", "no_html": "1", "skip_disambig": "1"},
            headers=headers,
            timeout=timeout_seconds,
        )
        if resp.status_code == 200:
            data = resp.json()
            abstract = data.get("AbstractText", "").strip()
            source_url = data.get("AbstractURL", "").strip()
            if abstract:
                results.append({
                    "title": data.get("Heading", "DuckDuckGo Instant Answer"),
                    "snippet": abstract,
                    "url": source_url or "https://duckduckgo.com",
                })
            # Also check RelatedTopics
            for topic in data.get("RelatedTopics", []):
                if len(results) >= max_results:
                    break
                if isinstance(topic, dict) and "Text" in topic and topic.get("Text"):
                    results.append({
                        "title": topic.get("FirstURL", "Topic").split("/")[-1].replace("_", " "),
                        "snippet": topic["Text"],
                        "url": topic.get("FirstURL", "https://duckduckgo.com"),
                    })
    except Exception:
        pass

    if results:
        return results[:max_results]

    # 2. Fallback: DuckDuckGo HTML Lite search
    try:
        resp = httpx.post(
            "https://html.duckduckgo.com/html/",
            data={"q": clean_query},
            headers=headers,
            timeout=timeout_seconds,
        )
        if resp.status_code == 200:
            html = resp.text
            # Extract snippets using regex
            snippets = re.findall(r'<a class="result__snippet[^>]*>(.*?)</a>', html, re.DOTALL)
            titles = re.findall(r'<a class="result__url[^>]*href="([^"]+)"[^>]*>(.*?)</a>', html, re.DOTALL)
            for i in range(min(max_results, len(snippets))):
                clean_snippet = re.sub(r"<[^>]+>", "", snippets[i]).strip()
                url = titles[i][0] if i < len(titles) else "https://duckduckgo.com"
                if clean_snippet:
                    results.append({
                        "title": f"Web Result #{i + 1}",
                        "snippet": clean_snippet,
                        "url": url,
                    })
    except Exception:
        pass

    return results[:max_results]
