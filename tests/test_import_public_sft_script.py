from __future__ import annotations

import importlib.util
import json
import re
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from superagi.ingestion.tokenizer import BpeTokenizer


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "import_public_sft.py"
SPEC = importlib.util.spec_from_file_location("import_public_sft", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
import_public_sft = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(import_public_sft)


class ImportPublicSftScriptTests(unittest.TestCase):
    def test_production_defaults_exclude_wildchat_but_recognize_it(self) -> None:
        self.assertEqual(
            import_public_sft.DEFAULT_SOURCES,
            ("no_robots", "dolly", "openassistant", "ultrachat"),
        )
        self.assertIn("wildchat", import_public_sft.SOURCE_DATASETS)

    def test_parse_args_accepts_deterministic_seed(self) -> None:
        parser = import_public_sft.build_parser()
        args = parser.parse_args(["--checkpoint", "checkpoint.pt", "--seed", "99"])

        self.assertEqual(args.seed, 99)

    def test_all_public_sources_pin_immutable_dataset_revisions(self) -> None:
        for source, spec in import_public_sft.SOURCE_DATASETS.items():
            with self.subTest(source=source):
                dataset_name, split, revision = spec
                self.assertTrue(dataset_name)
                self.assertTrue(split)
                self.assertRegex(revision, re.compile(r"^[0-9a-f]{40}$"))

    def test_source_loader_uses_pinned_revision(self) -> None:
        dataset_name, split, revision = import_public_sft.SOURCE_DATASETS["no_robots"]
        rows = [
            {
                "messages": [
                    {"role": "user", "content": "Explain trees."},
                    {
                        "role": "assistant",
                        "content": "Trees are perennial plants with woody stems.",
                    },
                ]
            }
        ]

        with patch.object(
            import_public_sft,
            "load_dataset",
            return_value=rows,
        ) as load_dataset:
            candidates = list(
                import_public_sft._iter_source_candidates(
                    source="no_robots",
                    max_rows=1,
                    max_messages=8,
                )
            )

        self.assertEqual(len(candidates), 1)
        load_dataset.assert_called_once_with(
            dataset_name,
            split=split,
            streaming=True,
            revision=revision,
        )

    def test_run_import_filters_globally_and_writes_selection_metadata(self) -> None:
        tokenizer = BpeTokenizer.from_text(
            "<bos><user> Explain something\n<agi> A detailed response with useful context for the reader.<eos>\n",
            vocab_size=300,
            min_frequency=1,
        )
        candidates = {
            "no_robots": [
                (
                    "no_robots:1",
                    [
                        {"role": "user", "content": "Explain something"},
                        {
                            "role": "agi",
                            "content": "A detailed response with useful context for the reader.",
                        },
                    ],
                ),
                (
                    "no_robots:2",
                    [
                        {"role": "user", "content": "Explain another topic"},
                        {
                            "role": "agi",
                            "content": "A separate response with different facts for the reader.",
                        },
                    ],
                ),
            ],
            "dolly": [
                (
                    "dolly:1",
                    [
                        {"role": "user", "content": "Explain something else"},
                        {
                            "role": "agi",
                            "content": "A detailed response with useful context for the reader!",
                        },
                    ],
                )
            ],
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            out_path = Path(tmp_dir) / "public.jsonl"
            metadata_path = Path(tmp_dir) / "public.metadata.json"
            args = SimpleNamespace(
                checkpoint="unused.pt",
                out=str(out_path),
                metadata=str(metadata_path),
                sources="no_robots,dolly",
                max_rows_per_source=10,
                max_examples_per_source=1,
                max_context_tokens=900,
                max_messages=8,
                max_agi_chars=1200,
                min_agi_chars=20,
                seed=1337,
            )
            with (
                patch.object(
                    import_public_sft,
                    "load_checkpoint",
                    return_value=SimpleNamespace(tokenizer=tokenizer),
                ),
                patch.object(
                    import_public_sft,
                    "_iter_source_candidates",
                    side_effect=lambda *, source, **_: iter(candidates[source]),
                ) as iter_candidates,
            ):
                self.assertEqual(import_public_sft.run_import(args), 0)

            records = [json.loads(line) for line in out_path.read_text().splitlines()]
            metadata = json.loads(metadata_path.read_text())

        self.assertEqual(len(records), 1)
        self.assertTrue(records[0]["source"].startswith("no_robots:"))
        self.assertEqual(iter_candidates.call_count, 2)
        self.assertEqual(metadata["seed"], 1337)
        self.assertEqual(metadata["selected_source_counts"], {"dolly": 0, "no_robots": 1})
        self.assertEqual(
            metadata["accepted_before_limit_source_counts"],
            {"dolly": 0, "no_robots": 2},
        )
        self.assertEqual(metadata["stats"]["accepted"], len(records))
        self.assertEqual(metadata["written_count"], len(records))
        self.assertEqual(metadata["accepted_before_limit"], 2)
        self.assertEqual(metadata["exact_duplicate_count"], 1)
        self.assertEqual(metadata["near_duplicate_count"], 0)
        self.assertEqual(metadata["filter_config"]["near_duplicate_threshold"], 0.88)
        self.assertGreater(metadata["token_counts"]["total"], 0)
        self.assertEqual(
            metadata["dataset_revisions"],
            {
                source: {
                    "dataset": import_public_sft.SOURCE_DATASETS[source][0],
                    "split": import_public_sft.SOURCE_DATASETS[source][1],
                    "revision": import_public_sft.SOURCE_DATASETS[source][2],
                }
                for source in ("dolly", "no_robots")
            },
        )


if __name__ == "__main__":
    unittest.main()
