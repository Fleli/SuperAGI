import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

import requests

from superagi.ingestion.builders.c4 import build_c4_corpus
from superagi.ingestion.builders.common import (
    BuildResult,
    normalize_document_text,
    slugify,
    write_build_metadata,
    write_raw_document,
)
from superagi.ingestion.builders.wikipedia import (
    MediaWikiClient,
    WikipediaArticle,
    WikipediaSearchResult,
    build_wikipedia_corpus,
)


class FakeWikipediaClient:
    def search(self, query: str, limit: int) -> list[WikipediaSearchResult]:
        return [
            WikipediaSearchResult(
                page_id=1,
                title="Ada Lovelace",
                url="https://en.wikipedia.org/wiki/Ada_Lovelace",
            ),
            WikipediaSearchResult(
                page_id=2,
                title="Analytical Engine",
                url="https://en.wikipedia.org/wiki/Analytical_Engine",
            ),
        ][:limit]

    def fetch_article(self, title: str) -> WikipediaArticle:
        return WikipediaArticle(
            page_id=1 if title == "Ada Lovelace" else 2,
            title=title,
            url=f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}",
            text=f"{title}\n\nCore article text.",
        )


class FakeRateLimitResponse:
    def __init__(
        self,
        status_code: int,
        payload: dict | None = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        self.status_code = status_code
        self._payload = payload or {}
        self.headers = headers or {}

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(
                f"{self.status_code} test error",
                response=self,
            )

    def json(self) -> dict:
        return self._payload


class FakeSession:
    def __init__(self, responses: list[FakeRateLimitResponse]) -> None:
        self.headers = {}
        self.responses = list(responses)

    def get(self, *args, **kwargs) -> FakeRateLimitResponse:
        if not self.responses:
            raise AssertionError("no fake response configured")
        return self.responses.pop(0)


class FailingArticleClient:
    def search(self, query: str, limit: int) -> list[WikipediaSearchResult]:
        return [
            WikipediaSearchResult(
                page_id=1,
                title="Rate Limited",
                url="https://en.wikipedia.org/wiki/Rate_Limited",
            ),
            WikipediaSearchResult(
                page_id=2,
                title="Usable",
                url="https://en.wikipedia.org/wiki/Usable",
            ),
        ]

    def fetch_article(self, title: str) -> WikipediaArticle:
        if title == "Rate Limited":
            raise requests.HTTPError("429 test error")
        return WikipediaArticle(
            page_id=2,
            title=title,
            url=f"https://en.wikipedia.org/wiki/{title}",
            text="Usable article body.",
        )


class CorpusBuilderCommonTests(unittest.TestCase):
    def test_slugifies_titles_for_stable_filenames(self) -> None:
        self.assertEqual(slugify("Ada Lovelace: First Programmer?"), "ada-lovelace-first-programmer")

    def test_normalizes_document_text(self) -> None:
        text = "  First line  \r\n\r\n\n  Second line\t\n"

        self.assertEqual(normalize_document_text(text), "First line\n\nSecond line")

    def test_writes_raw_document_and_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir) / "data" / "raw" / "source"
            document_path = write_raw_document(
                output_dir=output_dir,
                filename_stem="example",
                text="hello world",
            )
            metadata_path = write_build_metadata(
                output_dir=output_dir,
                metadata={
                    "source": "source",
                    "documents": [{"path": str(document_path)}],
                },
            )
            result = BuildResult(
                source="source",
                output_dir=output_dir,
                documents_written=1,
                document_paths=(document_path,),
                metadata_path=metadata_path,
            )

            document_text = document_path.read_text(encoding="utf-8")
            payload = json.loads(metadata_path.read_text(encoding="utf-8"))

        self.assertEqual(document_path.name, "example.txt")
        self.assertEqual(document_text, "hello world\n")
        self.assertEqual(payload["source"], "source")
        self.assertEqual(result.documents_written, 1)


class WikipediaBuilderTests(unittest.TestCase):
    def test_builds_wikipedia_documents_and_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            raw_root = Path(tmp_dir) / "raw"

            result = build_wikipedia_corpus(
                queries=["computing"],
                raw_root=raw_root,
                max_articles_per_query=2,
                min_chars=1,
                client=FakeWikipediaClient(),
            )

            metadata = json.loads(result.metadata_path.read_text(encoding="utf-8"))
            document_texts = [
                path.read_text(encoding="utf-8") for path in result.document_paths
            ]

        self.assertEqual(result.source, "wikipedia")
        self.assertEqual(result.documents_written, 2)
        self.assertEqual([path.name for path in result.document_paths], [
            "000001-ada-lovelace.txt",
            "000002-analytical-engine.txt",
        ])
        self.assertEqual(metadata["queries"], ["computing"])
        self.assertEqual(metadata["documents"][0]["title"], "Ada Lovelace")
        self.assertEqual(metadata["documents"][0]["page_id"], 1)
        self.assertIn("Core article text.", document_texts[0])

    def test_mediawiki_client_sets_wikimedia_user_agent_headers(self) -> None:
        user_agent = "SuperAGI-learning-corpus-builder/0.1 (mailto:test@example.com)"
        session = FakeSession(responses=[])

        MediaWikiClient(session=session, user_agent=user_agent)

        self.assertEqual(session.headers["User-Agent"], user_agent)
        self.assertEqual(session.headers["Api-User-Agent"], user_agent)

    def test_mediawiki_client_retries_after_rate_limit(self) -> None:
        payload = {
            "query": {
                "search": [
                    {
                        "pageid": 1,
                        "title": "Machine learning",
                    }
                ]
            }
        }
        session = FakeSession(
            responses=[
                FakeRateLimitResponse(
                    status_code=429,
                    headers={"Retry-After": "2"},
                ),
                FakeRateLimitResponse(status_code=200, payload=payload),
            ]
        )
        sleep = Mock()
        client = MediaWikiClient(
            session=session,
            sleep=sleep,
            request_delay_seconds=0.0,
            max_retries=1,
        )

        results = client.search("machine learning", 1)

        self.assertEqual(results[0].title, "Machine learning")
        sleep.assert_called_once_with(2.0)

    def test_build_wikipedia_corpus_skips_failed_article_fetches(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            result = build_wikipedia_corpus(
                queries=["test"],
                raw_root=Path(tmp_dir),
                max_articles_per_query=2,
                min_chars=1,
                client=FailingArticleClient(),
            )

            metadata = json.loads(result.metadata_path.read_text(encoding="utf-8"))

        self.assertEqual(result.documents_written, 1)
        self.assertEqual(metadata["documents"][0]["title"], "Usable")
        self.assertEqual(metadata["skipped_documents"][0]["title"], "Rate Limited")


class C4BuilderTests(unittest.TestCase):
    def test_builds_c4_documents_and_metadata_from_iterable(self) -> None:
        examples = [
            {
                "text": "First C4 document.",
                "url": "https://example.com/first",
                "timestamp": "2020-01-01T00:00:00Z",
            },
            {
                "text": "Second C4 document.",
                "url": "https://example.com/second",
                "timestamp": "2020-01-02T00:00:00Z",
            },
        ]
        with tempfile.TemporaryDirectory() as tmp_dir:
            raw_root = Path(tmp_dir) / "raw"

            result = build_c4_corpus(
                raw_root=raw_root,
                max_documents=2,
                min_chars=1,
                examples=examples,
            )

            metadata = json.loads(result.metadata_path.read_text(encoding="utf-8"))
            document_texts = [
                path.read_text(encoding="utf-8") for path in result.document_paths
            ]

        self.assertEqual(result.source, "c4")
        self.assertEqual(result.documents_written, 2)
        self.assertEqual([path.name for path in result.document_paths], [
            "document-000001.txt",
            "document-000002.txt",
        ])
        self.assertEqual(metadata["dataset"], "allenai/c4")
        self.assertEqual(metadata["documents"][0]["url"], "https://example.com/first")
        self.assertEqual(metadata["documents"][1]["timestamp"], "2020-01-02T00:00:00Z")
        self.assertEqual(document_texts[0], "First C4 document.\n")


if __name__ == "__main__":
    unittest.main()
