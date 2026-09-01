from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import httpx


@dataclass
class WebSearchResult:
    title: str
    snippet: str
    url: str


def search_duckduckgo(query: str, max_results: int = 3) -> list[dict[str, Any]]:
    """Zero-cost public web search via DuckDuckGo HTML endpoint without API keys."""
    url = "https://html.duckduckgo.com/html/"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    }
    data = {"q": query}
    try:
        with httpx.Client(timeout=8.0, follow_redirects=True) as client:
            resp = client.post(url, data=data, headers=headers)
            if resp.status_code != 200:
                return []
            
            # Simple text parsing of result snippets
            from html.parser import HTMLParser
            
            class DDGParser(HTMLParser):
                def __init__(self):
                    super().__init__()
                    self.results = []
                    self.in_snippet = False
                    self.in_title = False
                    self.current_title = ""
                    self.current_snippet = ""
                    self.current_url = ""

                def handle_starttag(self, tag, attrs):
                    attrs_dict = dict(attrs)
                    cls = attrs_dict.get("class", "")
                    if tag == "a" and "result__snippet" in cls:
                        self.in_snippet = True
                        self.current_snippet = ""
                    elif tag == "a" and "result__url" in cls:
                        self.current_url = attrs_dict.get("href", "")
                    elif tag == "a" and "result__a" in cls:
                        self.in_title = True
                        self.current_title = ""

                def handle_endtag(self, tag):
                    if tag == "a":
                        if self.in_snippet and self.current_snippet:
                            self.results.append({
                                "title": self.current_title.strip(),
                                "snippet": self.current_snippet.strip(),
                                "url": self.current_url.strip(),
                            })
                            self.in_snippet = False
                        self.in_title = False

                def handle_data(self, data):
                    if self.in_snippet:
                        self.current_snippet += data
                    elif self.in_title:
                        self.current_title += data

            parser = DDGParser()
            parser.feed(resp.text)
            return parser.results[:max_results]
    except Exception:
        return []
