import json
import tempfile
import unittest
from pathlib import Path

from superagi.ingestion.sources import (
    DEFAULT_SOURCE_SPECS,
    SourceSpec,
    build_multi_source_token_shards,
    clean_source_example,
    iter_mixed_source_examples,
    parse_source_names,
)


class SourceIngestionTests(unittest.TestCase):
    def test_default_sources_include_requested_corpora(self) -> None:
        self.assertEqual(
            set(DEFAULT_SOURCE_SPECS),
            {
                "fineweb",
                "wikipedia",
                "dolma",
                "openwebmath",
                "arxiv",
                "pmc",
                "stackexchange",
                "gutenberg",
            },
        )
        self.assertEqual(DEFAULT_SOURCE_SPECS["fineweb"].dataset_name, "HuggingFaceFW/fineweb")
        self.assertEqual(DEFAULT_SOURCE_SPECS["openwebmath"].dataset_name, "open-web-math/open-web-math")
        self.assertEqual(DEFAULT_SOURCE_SPECS["dolma"].dataset_name, "emozilla/dolma-v1_7-3B")
        self.assertEqual(DEFAULT_SOURCE_SPECS["pmc"].dataset_name, "casperhansen/pmc-oa-markdown")
        self.assertEqual(DEFAULT_SOURCE_SPECS["gutenberg"].split, "en")

    def test_parse_source_names_preserves_requested_order(self) -> None:
        specs = parse_source_names("wikipedia,openwebmath,gutenberg")

        self.assertEqual([spec.name for spec in specs], ["wikipedia", "openwebmath", "gutenberg"])

    def test_parse_source_names_rejects_unknown_sources(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown source"):
            parse_source_names("fineweb,unknown")

    def test_wikipedia_cleaner_removes_markup_and_references(self) -> None:
        text = """== History ==
        Machine learning is useful.[12]

        === See also ===
        Ignore this

        Category:Computer science
        """

        cleaned = clean_source_example("wikipedia", {"text": text})

        self.assertEqual(cleaned, "History\nMachine learning is useful.")

    def test_arxiv_cleaner_keeps_abstract_and_conclusion(self) -> None:
        example = {
            "abstract": "We introduce a small model for experiments.",
            "article": (
                "1 Introduction\nEarlier work.\n\n"
                "5 Conclusion\nThe method improves training stability.\n"
                "References\n[1] Ignore this."
            ),
        }

        cleaned = clean_source_example("arxiv", example)

        self.assertEqual(
            cleaned,
            "Abstract\nWe introduce a small model for experiments.\n\n"
            "Conclusion\nThe method improves training stability.",
        )

    def test_pmc_cleaner_removes_references_and_figure_captions(self) -> None:
        example = {
            "text": (
                "Abstract\nA biomedical article.\n\n"
                "Figure 1. A microscopy image.\n\n"
                "Discussion\nThe treatment was effective.\n\n"
                "References\n1. Some citation."
            )
        }

        cleaned = clean_source_example("pmc", example)

        self.assertEqual(cleaned, "Abstract\nA biomedical article.\n\nDiscussion\nThe treatment was effective.")

    def test_stackexchange_cleaner_strips_html_and_code_blocks(self) -> None:
        example = {
            "title": "Why use attention?",
            "question": "<p>Why is <b>attention</b> useful?</p><pre>print('skip')</pre>",
            "accepted_answer": "<p>It lets each token use relevant context.</p>",
        }

        cleaned = clean_source_example("stackexchange", example)

        self.assertEqual(
            cleaned,
            "Question: Why use attention?\n\n"
            "Why is attention useful?\n\n"
            "Answer: It lets each token use relevant context.",
        )

    def test_gutenberg_cleaner_removes_project_gutenberg_boilerplate(self) -> None:
        example = {
            "text": (
                "Project Gutenberg header\n"
                "*** START OF THE PROJECT GUTENBERG EBOOK TEST ***\n"
                "Chapter 1\nA real paragraph from a book.\n"
                "*** END OF THE PROJECT GUTENBERG EBOOK TEST ***\n"
                "Project Gutenberg footer"
            )
        }

        cleaned = clean_source_example("gutenberg", example)

        self.assertEqual(cleaned, "Chapter 1\nA real paragraph from a book.")

    def test_mixed_source_iterator_round_robins_clean_documents(self) -> None:
        specs = (
            SourceSpec(name="alpha", dataset_name="unused-alpha", cleaner="generic"),
            SourceSpec(name="beta", dataset_name="unused-beta", cleaner="generic"),
        )
        source_examples = {
            "alpha": [{"text": "alpha one"}, {"text": "alpha two"}],
            "beta": [{"text": "beta one"}],
        }

        examples = list(
            iter_mixed_source_examples(
                specs=specs,
                max_documents_per_source=2,
                source_examples=source_examples,
            )
        )

        self.assertEqual([example["source"] for example in examples], ["alpha", "beta", "alpha"])
        self.assertEqual([example["text"] for example in examples], ["alpha one", "beta one", "alpha two"])

    def test_build_multi_source_token_shards_records_source_metadata(self) -> None:
        specs = (
            SourceSpec(name="fineweb", dataset_name="unused", cleaner="generic"),
            SourceSpec(name="openwebmath", dataset_name="unused", cleaner="openwebmath"),
        )
        source_examples = {
            "fineweb": [
                {"text": "general web text about machine learning systems"},
                {"text": "another useful general web document"},
            ],
            "openwebmath": [
                {"text": "Let x be a vector and compute the matrix product."},
                {"text": "A proof follows from the theorem and the lemma."},
            ],
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            result = build_multi_source_token_shards(
                processed_dir=Path(tmp_dir) / "processed",
                specs=specs,
                max_documents_per_source=2,
                tokenizer_sample_documents=2,
                shard_token_count=8,
                validation_token_count=4,
                min_chars=1,
                bpe_vocab_size=300,
                bpe_min_frequency=1,
                source_examples=source_examples,
            )
            vocab = json.loads(result.vocab_path.read_text(encoding="utf-8"))

        self.assertEqual(vocab["source"], "mixed")
        self.assertEqual(vocab["sources"], ["fineweb", "openwebmath"])
        self.assertGreaterEqual(result.documents_tokenized, 2)
        self.assertGreater(result.train_tokens, 0)


if __name__ == "__main__":
    unittest.main()
