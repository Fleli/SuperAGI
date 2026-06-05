import json
import tempfile
import unittest
from pathlib import Path

from superagi.ingestion.builders.c4 import build_c4_corpus
from superagi.ingestion.builders.common import (
    BuildResult,
    normalize_document_text,
    slugify,
    write_build_metadata,
    write_raw_document,
)
from superagi.ingestion.builders.wikipedia import (
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
