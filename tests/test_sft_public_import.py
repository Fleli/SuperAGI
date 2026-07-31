import json
import tempfile
import unittest
from pathlib import Path

from scripts import import_public_sft
from superagi.chat.formatting import ChatMessage
from superagi.chat.sft_public_import import (
    ImportFilterConfig,
    ImportedSftExample,
    PublicSftImporter,
    convert_dolly_row,
    convert_no_robots_row,
    convert_ultrachat_row,
    convert_wildchat_row,
    iter_openassistant_conversations,
    seeded_source_sample,
)
from superagi.ingestion.tokenizer import BpeTokenizer


class PublicSftImportTests(unittest.TestCase):
    def test_converts_no_robots_messages_to_sft_roles(self) -> None:
        row = {
            "messages": [
                {"role": "user", "content": "What are you?"},
                {"role": "assistant", "content": "I am a small model."},
            ]
        }

        messages = convert_no_robots_row(row)

        self.assertEqual(
            [(message.role, message.content) for message in messages],
            [("user", "What are you?"), ("agi", "I am a small model.")],
        )

    def test_converts_dolly_context_into_single_user_turn(self) -> None:
        row = {
            "instruction": "When did it start?",
            "context": "The service started in 2020.",
            "response": "It started in 2020.",
        }

        messages = convert_dolly_row(row)

        self.assertEqual(messages[0].role, "user")
        self.assertIn("Context:\nThe service started in 2020.", messages[0].content)
        self.assertIn("Instruction:\nWhen did it start?", messages[0].content)
        self.assertEqual(messages[1].role, "agi")
        self.assertEqual(messages[1].content, "It started in 2020.")

    def test_converts_wildchat_and_limits_turns(self) -> None:
        row = {
            "conversation": [
                {"role": "user", "content": "One", "language": "English", "toxic": False, "redacted": False},
                {"role": "assistant", "content": "Two", "language": "English", "toxic": False, "redacted": False},
                {"role": "user", "content": "Three", "language": "English", "toxic": False, "redacted": False},
                {"role": "assistant", "content": "Four", "language": "English", "toxic": False, "redacted": False},
            ],
            "language": "English",
            "toxic": False,
            "redacted": False,
        }

        messages = convert_wildchat_row(row, max_turns=2)

        self.assertEqual(
            [(message.role, message.content) for message in messages],
            [("user", "One"), ("agi", "Two")],
        )

    def test_converts_ultrachat_messages(self) -> None:
        row = {
            "messages": [
                {"role": "user", "content": "Explain AI."},
                {"role": "assistant", "content": "AI is software that learns patterns."},
            ]
        }

        messages = convert_ultrachat_row(row)

        self.assertEqual(messages[0].role, "user")
        self.assertEqual(messages[1].role, "agi")

    def test_reconstructs_openassistant_parent_child_path(self) -> None:
        rows = [
            {
                "message_id": "root",
                "parent_id": None,
                "role": "prompter",
                "text": "What is ML?",
                "lang": "en",
                "deleted": False,
                "review_result": True,
            },
            {
                "message_id": "child",
                "parent_id": "root",
                "role": "assistant",
                "text": "ML learns patterns from data.",
                "lang": "en",
                "deleted": False,
                "review_result": True,
            },
        ]

        conversations = list(iter_openassistant_conversations(rows))

        self.assertEqual(len(conversations), 1)
        self.assertEqual(
            [(message.role, message.content) for message in conversations[0]],
            [("user", "What is ML?"), ("agi", "ML learns patterns from data.")],
        )

    def test_importer_filters_long_repeated_and_duplicate_answers(self) -> None:
        tokenizer = BpeTokenizer.from_texts(
            [
                "<bos><user> short\n<agi> Direct answer about housing prices.<eos>\n",
                "<bos><user> long\n<agi> "
                + " ".join(f"word{index}" for index in range(200))
                + "<eos>\n",
                "<bos><user> repeated\n<agi> echo echo echo echo echo echo echo echo.<eos>\n",
            ],
            vocab_size=300,
            min_frequency=1,
        )
        importer = PublicSftImporter(
            tokenizer=tokenizer,
            filter_config=ImportFilterConfig(
                max_context_tokens=40,
                min_agi_chars=5,
                max_agi_chars=3000,
                max_agi_tokens=10000,
                max_repeated_five_grams=1,
            ),
        )
        rows = [
            ("first", [{"role": "user", "content": "short"}, {"role": "agi", "content": "Direct answer about housing prices."}]),
            ("duplicate", [{"role": "user", "content": "short again"}, {"role": "agi", "content": "Direct answer about housing prices."}]),
            (
                "long",
                [
                    {"role": "user", "content": "long"},
                    {
                        "role": "agi",
                        "content": " ".join(f"word{index}" for index in range(200)),
                    },
                ],
            ),
            ("repeated", [{"role": "user", "content": "repeated"}, {"role": "agi", "content": "echo echo echo echo echo echo echo echo."}]),
        ]

        imported = importer.import_conversations(rows)

        self.assertEqual(len(imported.examples), 1)
        self.assertEqual(imported.examples[0].source, "first")
        self.assertEqual(imported.stats.accepted, 1)
        self.assertEqual(imported.stats.rejected_by_reason["duplicate_answer"], 1)
        self.assertEqual(imported.stats.rejected_by_reason["too_long"], 1)
        self.assertEqual(imported.stats.rejected_by_reason["repeated_phrase"], 1)

    def test_importer_rejects_response_above_token_budget(self) -> None:
        long_answer = " ".join(f"token{index}" for index in range(40))
        tokenizer = BpeTokenizer.from_texts(
            [
                "<bos><user> Short question\n<agi> A concise useful answer.<eos>\n",
                f"<bos><user> Long question\n<agi> {long_answer}<eos>\n",
            ],
            vocab_size=300,
            min_frequency=1,
        )
        importer = PublicSftImporter(
            tokenizer=tokenizer,
            filter_config=ImportFilterConfig(
                min_agi_chars=5,
                max_agi_chars=3000,
                max_agi_tokens=len(
                    tokenizer.encode_with_offsets(
                        "A concise useful answer."
                    ).ids
                ),
            ),
        )

        imported = importer.import_conversations(
            [
                (
                    "short:1",
                    [
                        {"role": "user", "content": "Short question"},
                        {"role": "agi", "content": "A concise useful answer."},
                    ],
                ),
                (
                    "long:1",
                    [
                        {"role": "user", "content": "Long question"},
                        {"role": "agi", "content": long_answer},
                    ],
                ),
            ]
        )

        self.assertEqual(
            [example.source for example in imported.examples],
            ["short:1"],
        )
        self.assertEqual(
            imported.stats.rejected_by_reason["answer_token_limit"],
            1,
        )

    def test_importer_enforces_ngram_budget_against_reference_corpus(self) -> None:
        shared = "shared boilerplate phrase appears right here"
        tokenizer = BpeTokenizer.from_texts(
            [
                f"<bos><user> Reference\n<agi> {shared} with reference details.<eos>\n",
                f"<bos><user> First\n<agi> {shared} followed by alpha beta gamma.<eos>\n",
                f"<bos><user> Second\n<agi> {shared} followed by delta epsilon zeta.<eos>\n",
                "<bos><user> Third\n<agi> A separate response with distinct useful facts.<eos>\n",
            ],
            vocab_size=300,
            min_frequency=1,
        )
        importer = PublicSftImporter(
            tokenizer=tokenizer,
            filter_config=ImportFilterConfig(
                min_agi_chars=5,
                near_duplicate_threshold=0.99,
                cross_example_ngram_size=5,
                max_cross_example_ngram_count=2,
            ),
        )
        reference = (
            ChatMessage(role="user", content="Reference"),
            ChatMessage(
                role="agi",
                content=f"{shared} with reference details.",
            ),
        )

        imported = importer.import_conversations(
            [
                (
                    "first:1",
                    [
                        {"role": "user", "content": "First"},
                        {
                            "role": "agi",
                            "content": f"{shared} followed by alpha beta gamma.",
                        },
                    ],
                ),
                (
                    "second:1",
                    [
                        {"role": "user", "content": "Second"},
                        {
                            "role": "agi",
                            "content": f"{shared} followed by delta epsilon zeta.",
                        },
                    ],
                ),
                (
                    "third:1",
                    [
                        {"role": "user", "content": "Third"},
                        {
                            "role": "agi",
                            "content": "A separate response with distinct useful facts.",
                        },
                    ],
                ),
            ],
            ngram_reference_conversations=[reference],
        )

        self.assertEqual(
            [example.source for example in imported.examples],
            ["first:1", "third:1"],
        )
        self.assertEqual(
            imported.stats.rejected_by_reason["cross_example_repetition"],
            1,
        )

    def test_importer_rejects_global_exact_and_near_duplicate_answers(self) -> None:
        tokenizer = BpeTokenizer.from_text(
            "<bos><user> Ask\n<agi> Alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron pi rho sigma.<eos>\n",
            vocab_size=300,
            min_frequency=1,
        )
        importer = PublicSftImporter(
            tokenizer=tokenizer,
            filter_config=ImportFilterConfig(min_agi_chars=5),
        )

        imported = importer.import_conversations(
            [
                (
                    "first:1",
                    [
                        {"role": "user", "content": "Ask one"},
                        {
                            "role": "agi",
                            "content": "Alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron pi rho sigma.",
                        },
                    ],
                ),
                (
                    "second:1",
                    [
                        {"role": "user", "content": "Ask two"},
                        {
                            "role": "agi",
                            "content": "ALPHA, beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron pi rho sigma!",
                        },
                    ],
                ),
                (
                    "third:1",
                    [
                        {"role": "user", "content": "Ask three"},
                        {
                            "role": "agi",
                            "content": "Alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron pi rho tau.",
                        },
                    ],
                ),
            ]
        )

        self.assertEqual([example.source for example in imported.examples], ["first:1"])
        self.assertEqual(imported.stats.rejected_by_reason["duplicate_answer"], 1)
        self.assertEqual(imported.stats.rejected_by_reason["near_duplicate"], 1)

    def test_importer_rejects_near_duplicates_with_different_openings(self) -> None:
        shared = " ".join(f"shared{index:02d}" for index in range(17))
        tokenizer = BpeTokenizer.from_text(
            f"<bos><user> Ask\n<agi> fresh opening one {shared}<eos>\n",
            vocab_size=300,
            min_frequency=1,
        )
        importer = PublicSftImporter(
            tokenizer=tokenizer,
            filter_config=ImportFilterConfig(min_agi_chars=5),
        )

        imported = importer.import_conversations(
            [
                (
                    "first:1",
                    [
                        {"role": "user", "content": "First question"},
                        {"role": "agi", "content": f"fresh opening one {shared}"},
                    ],
                ),
                (
                    "second:1",
                    [
                        {"role": "user", "content": "Second question"},
                        {"role": "agi", "content": f"different opening one {shared}"},
                    ],
                ),
            ]
        )

        self.assertEqual([example.source for example in imported.examples], ["first:1"])
        self.assertEqual(imported.stats.rejected_by_reason["near_duplicate"], 1)

    def test_importer_rejects_answer_duplicates_within_one_conversation(self) -> None:
        repeated_answer = "A thermostat compares the measured temperature with its target setting."
        tokenizer = BpeTokenizer.from_text(
            f"<bos><user> First question\n<agi> {repeated_answer}\n"
            f"<user> Follow-up question\n<agi> {repeated_answer}<eos>\n",
            vocab_size=300,
            min_frequency=1,
        )
        importer = PublicSftImporter(
            tokenizer=tokenizer,
            filter_config=ImportFilterConfig(min_agi_chars=5),
        )

        imported = importer.import_conversations(
            [
                (
                    "ultrachat:1",
                    [
                        {"role": "user", "content": "First question"},
                        {"role": "agi", "content": repeated_answer},
                        {"role": "user", "content": "Follow-up question"},
                        {"role": "agi", "content": repeated_answer},
                    ],
                )
            ]
        )

        self.assertEqual(imported.examples, ())
        self.assertEqual(imported.stats.rejected_by_reason["near_duplicate"], 1)

    def test_importer_seeds_answer_deduplication_from_reference_corpus(self) -> None:
        shared = "A heat pump moves existing heat instead of creating heat directly."
        tokenizer = BpeTokenizer.from_texts(
            [
                f"<bos><user> Reference\n<agi> {shared}<eos>\n",
                f"<bos><user> Public\n<agi> {shared} efficiently.<eos>\n",
            ],
            vocab_size=300,
            min_frequency=1,
        )
        importer = PublicSftImporter(
            tokenizer=tokenizer,
            filter_config=ImportFilterConfig(min_agi_chars=5),
        )
        reference = (
            ChatMessage(role="user", content="Reference"),
            ChatMessage(role="agi", content=shared),
        )

        imported = importer.import_conversations(
            [
                (
                    "no_robots:1",
                    [
                        {"role": "user", "content": "Public"},
                        {"role": "agi", "content": f"{shared} efficiently."},
                    ],
                )
            ],
            ngram_reference_conversations=[reference],
        )

        self.assertEqual(imported.examples, ())
        self.assertEqual(imported.stats.rejected_by_reason["near_duplicate"], 1)

    def test_importer_rejects_near_duplicate_prompt_answer_pairs(self) -> None:
        long_prompt = " ".join(f"sharedprompt{index}" for index in range(50))
        tokenizer = BpeTokenizer.from_texts(
            [
                f"<bos><user> {long_prompt}\n<agi> Choose red today.<eos>\n",
                f"<bos><user> {long_prompt} please\n<agi> Select blue now.<eos>\n",
            ],
            vocab_size=500,
            min_frequency=1,
        )
        importer = PublicSftImporter(
            tokenizer=tokenizer,
            filter_config=ImportFilterConfig(
                min_agi_chars=5,
                max_user_chars=10000,
            ),
        )

        imported = importer.import_conversations(
            [
                (
                    "dolly:1",
                    [
                        {"role": "user", "content": long_prompt},
                        {"role": "agi", "content": "Choose red today."},
                    ],
                ),
                (
                    "dolly:2",
                    [
                        {"role": "user", "content": f"{long_prompt} please"},
                        {"role": "agi", "content": "Select blue now."},
                    ],
                ),
            ]
        )

        self.assertEqual(
            [example.source for example in imported.examples],
            ["dolly:1"],
        )
        self.assertEqual(
            imported.stats.rejected_by_reason["near_duplicate_prompt_answer_pair"],
            1,
        )

    def test_importer_seeds_pair_deduplication_from_reference_corpus(self) -> None:
        long_prompt = " ".join(f"referenceprompt{index}" for index in range(50))
        tokenizer = BpeTokenizer.from_texts(
            [
                f"<bos><user> {long_prompt}\n<agi> Choose red today.<eos>\n",
                f"<bos><user> {long_prompt} please\n<agi> Select blue now.<eos>\n",
            ],
            vocab_size=500,
            min_frequency=1,
        )
        importer = PublicSftImporter(
            tokenizer=tokenizer,
            filter_config=ImportFilterConfig(
                min_agi_chars=5,
                max_user_chars=10000,
            ),
        )
        reference = (
            ChatMessage(role="user", content=long_prompt),
            ChatMessage(role="agi", content="Choose red today."),
        )

        imported = importer.import_conversations(
            [
                (
                    "dolly:1",
                    [
                        {"role": "user", "content": f"{long_prompt} please"},
                        {"role": "agi", "content": "Select blue now."},
                    ],
                )
            ],
            ngram_reference_conversations=[reference],
        )

        self.assertEqual(imported.examples, ())
        self.assertEqual(
            imported.stats.rejected_by_reason["near_duplicate_prompt_answer_pair"],
            1,
        )

    def test_rarity_order_avoids_large_posting_retrieval_for_shared_openings(self) -> None:
        answers = [
            "a generic opening phrase "
            + " ".join(f"z{index:03d}{suffix}" for suffix in range(12))
            for index in range(300)
        ]
        tokenizer = BpeTokenizer.from_text(
            "<bos><user> Ask\n<agi> " + "\n".join(answers) + "<eos>\n",
            vocab_size=1000,
            min_frequency=1,
        )
        importer = PublicSftImporter(
            tokenizer=tokenizer,
            filter_config=ImportFilterConfig(min_agi_chars=5),
        )
        rows = [
            (
                f"source:{index}",
                [
                    {"role": "user", "content": f"Question {index}"},
                    {"role": "agi", "content": answer},
                ],
            )
            for index, answer in enumerate(answers)
        ]
        rows.append(
            (
                "source:final",
                [
                    {"role": "user", "content": "Final question"},
                    {
                        "role": "agi",
                        "content": "a generic opening phrase "
                        + " ".join(f"zfinal{suffix}" for suffix in range(12)),
                    },
                ],
            )
        )

        imported = importer.import_conversations(rows)

        self.assertEqual(len(imported.examples), 301)
        self.assertLessEqual(importer._near_duplicate_retrieval_work, len(answers) * 4)

    def test_importer_rejects_invalid_roles_identity_refusals_and_artifacts(self) -> None:
        tokenizer = BpeTokenizer.from_text(
            "<bos><user> Ask\n<agi> A sufficiently detailed ordinary answer.<eos>\n",
            vocab_size=300,
            min_frequency=1,
        )
        importer = PublicSftImporter(
            tokenizer=tokenizer,
            filter_config=ImportFilterConfig(min_agi_chars=5),
        )
        rows = [
            (
                "roles:1",
                [
                    {"role": "agi", "content": "This starts with the wrong role."},
                    {"role": "user", "content": "Why?"},
                ],
            ),
            (
                "identity:1",
                [
                    {"role": "user", "content": "Can you advise me?"},
                    {"role": "agi", "content": "I am a licensed financial adviser."},
                ],
            ),
            (
                "identity:2",
                [
                    {"role": "user", "content": "Where are you?"},
                    {"role": "agi", "content": "I live in London."},
                ],
            ),
            (
                "identity:3",
                [
                    {"role": "user", "content": "How did you find that?"},
                    {"role": "agi", "content": "I browsed the web to find it."},
                ],
            ),
            (
                "identity:4",
                [
                    {"role": "user", "content": "How long?"},
                    {"role": "agi", "content": "I have worked here for 20 years."},
                ],
            ),
            (
                "refusal:1",
                [
                    {"role": "user", "content": "Tell me a short joke."},
                    {"role": "agi", "content": "As an AI, I cannot help with that."},
                ],
            ),
            (
                "artifact:1",
                [
                    {"role": "user", "content": "Hello"},
                    {"role": "agi", "content": "Here is a replacement character: \ufffd"},
                ],
            ),
            (
                "artifact:2",
                [
                    {"role": "user", "content": "Hello"},
                    {"role": "agi", "content": "Leaked <agi> marker in this answer."},
                ],
            ),
            (
                "artifact:3",
                [
                    {"role": "user", "content": "Rewrite this code"},
                    {"role": "agi", "content": "The snippet contains a leaked <unk> marker."},
                ],
            ),
        ]

        imported = importer.import_conversations(rows)

        self.assertEqual(imported.examples, ())
        self.assertEqual(imported.stats.rejected_by_reason["role_sequence"], 1)
        self.assertEqual(imported.stats.rejected_by_reason["false_capability_or_identity"], 4)
        self.assertEqual(imported.stats.rejected_by_reason["generic_refusal"], 1)
        self.assertEqual(imported.stats.rejected_by_reason["artifact"], 3)

    def test_importer_rejects_forum_code_and_persona_artifacts(self) -> None:
        tokenizer = BpeTokenizer.from_text(
            "<bos><user> Ask\n<agi> A sufficiently detailed ordinary answer.<eos>\n",
            vocab_size=300,
            min_frequency=1,
        )
        importer = PublicSftImporter(
            tokenizer=tokenizer,
            filter_config=ImportFilterConfig(min_agi_chars=5),
        )
        rows = [
            (
                "forum:1",
                [
                    {"role": "user", "content": "How are you?"},
                    {
                        "role": "agi",
                        "content": (
                            "I am confused about the question. "
                            "- Jonathan Software Jun 10 '17 at 21:29"
                        ),
                    },
                ],
            ),
            (
                "code-fence:1",
                [
                    {"role": "user", "content": "Tell me about yourself."},
                    {
                        "role": "agi",
                        "content": "```I am a short prose response in a code fence.",
                    },
                ],
            ),
            (
                "boilerplate:1",
                [
                    {"role": "user", "content": "What should I cook?"},
                    {
                        "role": "agi",
                        "content": (
                            "As an AI language model, I can provide a few general ideas."
                        ),
                    },
                ],
            ),
            (
                "persona:1",
                [
                    {"role": "user", "content": "Who are you?"},
                    {
                        "role": "agi",
                        "content": "I am a citizen and have been elected to Congress.",
                    },
                ],
            ),
            (
                "persona:2",
                [
                    {"role": "user", "content": "How much do you earn?"},
                    {
                        "role": "agi",
                        "content": "My salary is less than two hundred dollars per month.",
                    },
                ],
            ),
        ]

        imported = importer.import_conversations(rows)

        self.assertEqual(imported.examples, ())
        self.assertEqual(imported.stats.rejected_by_reason["forum_attribution"], 1)
        self.assertEqual(imported.stats.rejected_by_reason["code_fence"], 1)
        self.assertEqual(imported.stats.rejected_by_reason["ai_boilerplate"], 1)
        self.assertEqual(
            imported.stats.rejected_by_reason["false_capability_or_identity"],
            2,
        )

    def test_importer_rejects_common_boilerplate_signature_and_persona_variants(self) -> None:
        tokenizer = BpeTokenizer.from_text(
            "<bos><user> Ask\n<agi> A sufficiently detailed ordinary answer.<eos>\n",
            vocab_size=300,
            min_frequency=1,
        )
        cases = {
            "ai_identity": "I am an AI language model trained to answer questions.",
            "language_model": "As a language model, I can suggest a few options.",
            "large_language_model": "As a large language model, I can help.",
            "named_model_identity": "I am ChatGPT and can answer that question.",
            "biography": "I grew up in Boston and work as a nurse.",
            "posted_by": "A plausible answer to the question. Posted by john_doe",
            "edited_by": "A plausible answer to the question. Last edited by admin",
            "mobile_signature": "Here is the answer you requested. Sent from my iPhone",
            "letter_signature": "Here is the answer you requested. Best regards, John",
            "tilde_fence": "~~~python\nprint('hello')\n~~~",
        }

        for label, answer in cases.items():
            with self.subTest(label=label):
                importer = PublicSftImporter(
                    tokenizer=tokenizer,
                    filter_config=ImportFilterConfig(min_agi_chars=5),
                )
                imported = importer.import_conversations(
                    [
                        (
                            f"variant:{label}",
                            [
                                {"role": "user", "content": "Please answer directly."},
                                {"role": "agi", "content": answer},
                            ],
                        )
                    ]
                )

                self.assertEqual(imported.examples, ())

    def test_short_response_defaults_match_library_and_cli(self) -> None:
        config = ImportFilterConfig()
        args = import_public_sft.build_parser().parse_args(
            ["--checkpoint", "checkpoint.pt"]
        )

        self.assertEqual(config.max_messages, 6)
        self.assertEqual(config.max_agi_chars, 700)
        self.assertEqual(config.max_agi_tokens, 192)
        self.assertEqual(args.max_messages, config.max_messages)
        self.assertEqual(args.max_agi_chars, config.max_agi_chars)
        self.assertEqual(args.max_agi_tokens, config.max_agi_tokens)

    def test_seeded_source_sample_is_stable_and_not_first_n(self) -> None:
        examples = tuple(
            ImportedSftExample(
                source=f"dolly:{index}",
                messages=(
                    ChatMessage(role="user", content=f"Question {index}"),
                    ChatMessage(role="agi", content=f"Answer {index} with useful detail."),
                ),
                token_count=10,
                supervised_token_count=5,
            )
            for index in range(10)
        )

        first = seeded_source_sample(examples, limit=3, seed=1337, source="dolly")
        second = seeded_source_sample(examples, limit=3, seed=1337, source="dolly")

        self.assertEqual(first, second)
        self.assertNotEqual(first, examples[:3])

    def test_writes_jsonl_and_metadata(self) -> None:
        tokenizer = BpeTokenizer.from_text(
            "<bos><user> Hello\n<agi> Hello back with a direct answer.<eos>\n",
            vocab_size=300,
            min_frequency=1,
        )
        importer = PublicSftImporter(tokenizer=tokenizer)
        imported = importer.import_conversations(
            [
                (
                    "unit",
                    [
                        {"role": "user", "content": "Hello"},
                        {"role": "agi", "content": "Hello back with a direct answer."},
                    ],
                )
            ]
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            out_path = Path(tmp_dir) / "sft.jsonl"
            metadata_path = Path(tmp_dir) / "metadata.json"
            importer.write_import(imported, out_path=out_path, metadata_path=metadata_path)

            lines = out_path.read_text(encoding="utf-8").splitlines()
            payload = json.loads(lines[0])
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

        self.assertEqual(payload["source"], "unit")
        self.assertEqual(payload["messages"][0]["role"], "user")
        self.assertEqual(metadata["stats"]["accepted"], 1)


if __name__ == "__main__":
    unittest.main()
