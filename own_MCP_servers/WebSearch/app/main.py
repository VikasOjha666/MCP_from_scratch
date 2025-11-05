# app/main.py
from mcp.server.fastmcp import FastMCP
from duckduckgo_search import DDGS
import requests
from bs4 import BeautifulSoup
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
from typing import List, Optional
import logging
import re
import os
import uvicorn


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mcp-websearch")

mcp = FastMCP(host="0.0.0.0",port=8001)

def extract_text_from_html(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for s in soup(["script", "style", "noscript", "header", "footer", "nav", "form", "svg"]):
        s.decompose()
    texts = []
    for tag in soup.find_all(["p", "h1", "h2", "h3", "h4", "li"]):
        t = tag.get_text(separator=" ", strip=True)
        if t:
            texts.append(t)
    if not texts:
        body = soup.get_text(separator=" ", strip=True)
        return body
    joined = "\n".join(texts)
    joined = re.sub(r'\s+', ' ', joined)
    return joined

def fetch_page_text(url: str, timeout: int = 8) -> Optional[str]:
    headers = {
        "User-Agent": "mcp-websearch-bot/1.0 (+https://example.com)"
    }
    try:
        r = requests.get(url, headers=headers, timeout=timeout)
        r.raise_for_status()
        return extract_text_from_html(r.text)
    except Exception as e:
        logger.warning("Failed to fetch %s : %s", url, e)
        return None

def summarize_text(text: str, sentences_count: int = 3) -> str:
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = LsaSummarizer()
    summary_sentences = summarizer(parser.document, sentences_count)
    summary = " ".join(str(s) for s in summary_sentences)
    return summary.strip()

@mcp.tool()
def web_search_summary(query: str, num_results: int = 3, summary_sentences: int = 3) -> dict:
    logger.info("Searching for: %s", query)
    if num_results < 1:
        num_results = 1
    if summary_sentences < 1:
        summary_sentences = 1

    # Use DDGS() to support modern duckduckgo_search versions
    try:
        with DDGS() as ddgs:
            ddg_results = list(ddgs.text(query, max_results=num_results))
    except Exception as e:
        logger.exception("DuckDuckGo search failed: %s", e)
        ddg_results = []

    results = []
    aggregated_texts = []
    for i, r in enumerate(ddg_results[:num_results]):
        # ddgs.text() yields dicts; handle common keys defensively
        title = r.get("title") or ""
        url = r.get("href") or r.get("url") or r.get("link") or ""
        snippet = r.get("body") or r.get("snippet") or ""

        page_text = None
        if url:
            page_text = fetch_page_text(url)

        if page_text and len(page_text.split()) > 30:
            try:
                summary = summarize_text(page_text, sentences_count=summary_sentences)
            except Exception as e:
                logger.warning("Summarization failed for %s: %s", url, e)
                summary = snippet or ""
            aggregated_texts.append(page_text)
        else:
            summary = snippet or ""

        results.append({
            "title": title,
            "url": url,
            "snippet": snippet,
            "summary": summary
        })

    aggregated_summary = ""
    if aggregated_texts:
        combined_text = "\n".join(aggregated_texts)
        try:
            aggregated_summary = summarize_text(combined_text, sentences_count=max(2, summary_sentences))
        except Exception as e:
            logger.warning("Aggregated summarization failed: %s", e)
            aggregated_summary = results[0]["summary"] if results else ""

    return {
        "query": query,
        "results": results,
        "aggregated_summary": aggregated_summary
    }


if __name__ == "__main__":
    # SSE transport - use HTTP instead for new projects
    mcp.run(transport="sse")
