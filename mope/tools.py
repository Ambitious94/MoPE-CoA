from __future__ import annotations

import os
import json
from typing import List

try:
    import requests
except Exception:  # pragma: no cover
    requests = None  # type: ignore


def _require_requests():
    if requests is None:
        raise RuntimeError("requests not installed; install with 'pip install mope[tools]'")


def serpapi_search(prompt: str, api_key: str | None = None, top_k: int = 3) -> str:
    """Call SerpAPI to perform web search and return a compact textual summary.

    Expects SERPAPI_API_KEY in env or provided api_key.
    """
    _require_requests()
    key = api_key or os.getenv("SERPAPI_API_KEY")
    if not key:
        return f"tool.web_search: SERPAPI_API_KEY missing; prompt='{prompt}'"

    # Use SerpAPI Google Search JSON API endpoint
    url = "https://serpapi.com/search.json"
    params = {
        "q": prompt,
        "api_key": key,
        "num": max(1, top_k),
    }
    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        return f"tool.web_search error: {e} | prompt='{prompt}'"

    results = []
    # Prefer organic results
    for item in (data.get("organic_results") or [])[:top_k]:
        title = item.get("title") or ""
        snippet = item.get("snippet") or item.get("summary") or ""
        results.append(f"{title}: {snippet}")
    if not results:
        return f"tool.web_search: no results | prompt='{prompt}'"
    return " | ".join(results)


def jina_crawl(prompt: str, api_key: str | None = None, top_k: int = 1) -> str:
    """Call Jina AI crawl API to fetch page content summary.

    Expects JINA_API_KEY in env or provided api_key.
    """
    _require_requests()
    key = api_key or os.getenv("JINA_API_KEY")
    if not key:
        return f"tool.crawl_page: JINA_API_KEY missing; prompt='{prompt}'"

    # Jina reader endpoint (example: https://r.jina.ai/http://example.com)
    # Here we assume 'prompt' contains a URL to crawl; if not, return message.
    target = prompt.strip()
    if not (target.startswith("http://") or target.startswith("https://")):
        return f"tool.crawl_page: prompt is not a URL: '{prompt}'"

    url = f"https://r.jina.ai/{target}"
    headers = {"Authorization": f"Bearer {key}"}
    try:
        resp = requests.get(url, headers=headers, timeout=15)
        resp.raise_for_status()
        text = resp.text.strip()
    except Exception as e:
        return f"tool.crawl_page error: {e} | url='{target}'"

    # Return shortened content (first 500 chars)
    if not text:
        return f"tool.crawl_page: empty content | url='{target}'"
    return (text[:500] + ("..." if len(text) > 500 else ""))
