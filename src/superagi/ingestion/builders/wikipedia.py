from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Callable, Protocol

import requests

from superagi.ingestion.builders.common import (
    BuildResult,
    normalize_document_text,
    slugify,
    write_build_metadata,
    write_raw_document,
)


WIKIPEDIA_API_URL = "https://en.wikipedia.org/w/api.php"
DEFAULT_WIKIPEDIA_USER_AGENT = (
    "SuperAGI-learning-corpus-builder/0.1 "
    "(local learning project; set WIKI_USER_AGENT for contact)"
)


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
        user_agent: str = DEFAULT_WIKIPEDIA_USER_AGENT,
        timeout: float = 20.0,
        request_delay_seconds: float = 0.25,
        max_retries: int = 3,
        retry_backoff_seconds: float = 5.0,
        session: requests.Session | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if request_delay_seconds < 0:
            raise ValueError("request_delay_seconds must be non-negative")
        if max_retries < 0:
            raise ValueError("max_retries must be non-negative")
        if retry_backoff_seconds < 0:
            raise ValueError("retry_backoff_seconds must be non-negative")

        self.api_url = api_url
        self.session = session or requests.Session()
        self.session.headers.update(
            {
                "User-Agent": user_agent,
                "Api-User-Agent": user_agent,
            }
        )
        self.timeout = timeout
        self.request_delay_seconds = request_delay_seconds
        self.max_retries = max_retries
        self.retry_backoff_seconds = retry_backoff_seconds
        self.sleep = sleep

    def search(self, query: str, limit: int) -> list[WikipediaSearchResult]:
        response = self._get(
            {
                "action": "query",
                "format": "json",
                "list": "search",
                "srnamespace": 0,
                "srsearch": query,
                "srlimit": limit,
            }
        )
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
        response = self._get(
            {
                "action": "query",
                "format": "json",
                "prop": "extracts",
                "explaintext": 1,
                "redirects": 1,
                "titles": title,
            }
        )
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

    def _get(self, params: dict[str, object]) -> requests.Response:
        for attempt in range(self.max_retries + 1):
            if self.request_delay_seconds:
                self.sleep(self.request_delay_seconds)
            response = self.session.get(
                self.api_url,
                params=params,
                timeout=self.timeout,
            )
            if response.status_code != 429:
                response.raise_for_status()
                return response
            if attempt == self.max_retries:
                response.raise_for_status()
            self.sleep(self._retry_delay(response, attempt))
        raise RuntimeError("unreachable retry state")

    def _retry_delay(self, response: requests.Response, attempt: int) -> float:
        retry_after = response.headers.get("Retry-After")
        if retry_after is not None:
            try:
                return float(retry_after)
            except ValueError:
                pass
        return self.retry_backoff_seconds * (attempt + 1)


def build_wikipedia_corpus(
    queries: list[str],
    raw_root: Path | str = Path("data/raw"),
    max_articles_per_query: int = 10,
    min_chars: int = 200,
    user_agent: str = DEFAULT_WIKIPEDIA_USER_AGENT,
    client: WikipediaClient | None = None,
) -> BuildResult:
    if not queries:
        raise ValueError("queries must contain at least one search term")
    if max_articles_per_query <= 0:
        raise ValueError("max_articles_per_query must be positive")
    if min_chars < 0:
        raise ValueError("min_chars must be non-negative")

    used_default_client = client is None
    client = client or MediaWikiClient(user_agent=user_agent)
    output_dir = Path(raw_root) / "wikipedia"
    output_dir.mkdir(parents=True, exist_ok=True)

    document_paths = []
    documents_metadata = []
    skipped_documents = []
    seen_page_ids = set()
    document_number = 1
    for query in queries:
        try:
            search_results = client.search(query, max_articles_per_query)
        except requests.RequestException as exc:
            skipped_documents.append(
                {
                    "query": query,
                    "title": None,
                    "error": str(exc),
                    "stage": "search",
                }
            )
            continue
        for result in search_results:
            if result.page_id in seen_page_ids:
                continue
            try:
                article = client.fetch_article(result.title)
            except requests.RequestException as exc:
                skipped_documents.append(
                    {
                        "query": query,
                        "title": result.title,
                        "page_id": result.page_id,
                        "error": str(exc),
                        "stage": "fetch_article",
                    }
                )
                continue
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
            "user_agent": user_agent if used_default_client else None,
            "documents": documents_metadata,
            "skipped_documents": skipped_documents,
        },
    )
    return BuildResult(
        source="wikipedia",
        output_dir=output_dir,
        documents_written=len(document_paths),
        document_paths=tuple(document_paths),
        metadata_path=metadata_path,
    )
