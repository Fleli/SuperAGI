from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import requests

from superagi.ingestion.builders.common import (
    BuildResult,
    normalize_document_text,
    slugify,
    write_build_metadata,
    write_raw_document,
)


WIKIPEDIA_API_URL = "https://en.wikipedia.org/w/api.php"


@dataclass(frozen=True)
class WikipediaSearchResult:
    page_id: int
    title: str
    url: str


@dataclass(frozen=True)
class WikipediaArticle:
    page_id: int
    title: str
    url: str
    text: str


class WikipediaClient(Protocol):
    def search(self, query: str, limit: int) -> list[WikipediaSearchResult]:
        ...

    def fetch_article(self, title: str) -> WikipediaArticle:
        ...


class MediaWikiClient:
    def __init__(
        self,
        api_url: str = WIKIPEDIA_API_URL,
        user_agent: str = "SuperAGI-learning-corpus-builder/0.1",
        timeout: float = 20.0,
    ) -> None:
        self.api_url = api_url
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": user_agent})
        self.timeout = timeout

    def search(self, query: str, limit: int) -> list[WikipediaSearchResult]:
        response = self.session.get(
            self.api_url,
            params={
                "action": "query",
                "format": "json",
                "list": "search",
                "srnamespace": 0,
                "srsearch": query,
                "srlimit": limit,
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        payload = response.json()
        results = []
        for item in payload.get("query", {}).get("search", []):
            title = item["title"]
            results.append(
                WikipediaSearchResult(
                    page_id=int(item["pageid"]),
                    title=title,
                    url=f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}",
                )
            )
        return results

    def fetch_article(self, title: str) -> WikipediaArticle:
        response = self.session.get(
            self.api_url,
            params={
                "action": "query",
                "format": "json",
                "prop": "extracts",
                "explaintext": 1,
                "redirects": 1,
                "titles": title,
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        payload = response.json()
        pages = payload.get("query", {}).get("pages", {})
        page = next(iter(pages.values()))
        article_title = page["title"]
        return WikipediaArticle(
            page_id=int(page["pageid"]),
            title=article_title,
            url=f"https://en.wikipedia.org/wiki/{article_title.replace(' ', '_')}",
            text=page.get("extract", ""),
        )


def build_wikipedia_corpus(
    queries: list[str],
    raw_root: Path | str = Path("data/raw"),
    max_articles_per_query: int = 10,
    min_chars: int = 200,
    client: WikipediaClient | None = None,
) -> BuildResult:
    if not queries:
        raise ValueError("queries must contain at least one search term")
    if max_articles_per_query <= 0:
        raise ValueError("max_articles_per_query must be positive")
    if min_chars < 0:
        raise ValueError("min_chars must be non-negative")

    client = client or MediaWikiClient()
    output_dir = Path(raw_root) / "wikipedia"
    output_dir.mkdir(parents=True, exist_ok=True)

    document_paths = []
    documents_metadata = []
    seen_page_ids = set()
    document_number = 1
    for query in queries:
        for result in client.search(query, max_articles_per_query):
            if result.page_id in seen_page_ids:
                continue
            article = client.fetch_article(result.title)
            text = normalize_document_text(article.text)
            if len(text) < min_chars:
                continue
            seen_page_ids.add(article.page_id)
            filename_stem = f"{document_number:06d}-{slugify(article.title)}"
            document_path = write_raw_document(output_dir, filename_stem, text)
            document_paths.append(document_path)
            documents_metadata.append(
                {
                    "path": str(document_path),
                    "page_id": article.page_id,
                    "title": article.title,
                    "url": article.url,
                    "query": query,
                    "chars": len(text),
                }
            )
            document_number += 1

    metadata_path = write_build_metadata(
        output_dir=output_dir,
        metadata={
            "source": "wikipedia",
            "queries": queries,
            "max_articles_per_query": max_articles_per_query,
            "min_chars": min_chars,
            "documents": documents_metadata,
        },
    )
    return BuildResult(
        source="wikipedia",
        output_dir=output_dir,
        documents_written=len(document_paths),
        document_paths=tuple(document_paths),
        metadata_path=metadata_path,
    )
