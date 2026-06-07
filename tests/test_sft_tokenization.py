import json
import re
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from superagi.chat.sft import (
    IGNORE_INDEX,
    load_sft_jsonl,
    tokenize_sft_messages,
)
from superagi.ingestion.tokenizer import BpeTokenizer, CharTokenizer


class SftTokenizationTests(unittest.TestCase):
    def test_tokenizes_chat_with_assistant_only_labels_using_char_offsets(self) -> None:
        messages = [
            {"role": "user", "content": "What are you?"},
            {"role": "agi", "content": "I am AGI."},
        ]
        text = "<bos><user> What are you?\n<agi> I am AGI.<eos>\n"
        tokenizer = CharTokenizer.from_text(text)

        tokenized = tokenize_sft_messages(messages, tokenizer)

        supervised_ids = [
            token_id
            for token_id in tokenized.target_ids
            if token_id != IGNORE_INDEX
        ]
        self.assertEqual(tokenized.text, text)
        self.assertEqual(tokenizer.decode(supervised_ids), "I am AGI.<eos>")
        self.assertEqual(tokenized.supervised_token_count, len("I am AGI.<eos>"))
        self.assertTrue(
            all(
                token_id == IGNORE_INDEX
                for token_id in tokenized.target_ids[: text.index("I am AGI.") - 1]
            )
        )

    def test_tokenizes_chat_with_assistant_only_labels_using_bpe_offsets(self) -> None:
        messages = [
            {"role": "user", "content": "Explain ML."},
            {"role": "agi", "content": "ML learns patterns from data."},
        ]
        text = "<bos><user> Explain ML.\n<agi> ML learns patterns from data.<eos>\n"
        tokenizer = BpeTokenizer.from_text(text, vocab_size=300, min_frequency=1)

        tokenized = tokenize_sft_messages(messages, tokenizer)

        supervised_ids = [
            token_id
            for token_id in tokenized.target_ids
            if token_id != IGNORE_INDEX
        ]
        self.assertEqual(
            tokenizer.decode(supervised_ids).strip(),
            "ML learns patterns from data.<eos>",
        )
        self.assertGreater(tokenized.supervised_token_count, 0)

    def test_tokenizes_all_agi_turns_in_multi_turn_chat(self) -> None:
        messages = [
            {"role": "user", "content": "What is AI?"},
            {"role": "agi", "content": "AI is software."},
            {"role": "user", "content": "And ML?"},
            {"role": "agi", "content": "ML learns from data."},
        ]
        text = (
            "<bos><user> What is AI?\n"
            "<agi> AI is software.<eos>\n"
            "<user> And ML?\n"
            "<agi> ML learns from data.<eos>\n"
        )
        tokenizer = CharTokenizer.from_text(text)

        tokenized = tokenize_sft_messages(messages, tokenizer)

        supervised_ids = [
            token_id
            for token_id in tokenized.target_ids
            if token_id != IGNORE_INDEX
        ]
        self.assertEqual(
            tokenizer.decode(supervised_ids),
            "AI is software.<eos>ML learns from data.<eos>",
        )

    def test_rejects_conversation_without_agi_labels(self) -> None:
        tokenizer = CharTokenizer.from_text("<bos><user> Hello\n")

        with self.assertRaisesRegex(ValueError, "no AGI response tokens"):
            tokenize_sft_messages([{"role": "user", "content": "Hello"}], tokenizer)

    def test_loads_sft_jsonl_messages(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "sft.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "messages": [
                            {"role": "user", "content": "What are you?"},
                            {"role": "agi", "content": "A small model."},
                        ]
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            conversations = load_sft_jsonl(path)

        self.assertEqual(len(conversations), 1)
        self.assertEqual(conversations[0][0].role, "user")
        self.assertEqual(conversations[0][1].role, "agi")

    def test_tracked_seed_sft_corpus_loads(self) -> None:
        path = Path(__file__).resolve().parents[1] / "data" / "sft" / "seed.jsonl"

        conversations = load_sft_jsonl(path)

        self.assertGreaterEqual(len(conversations), 200)
        self.assertTrue(
            all(
                any(message.role == "agi" for message in conversation)
                for conversation in conversations
            )
        )

    def test_tracked_seed_sft_corpus_has_everyday_chat_coverage(self) -> None:
        path = Path(__file__).resolve().parents[1] / "data" / "sft" / "seed.jsonl"

        conversations = load_sft_jsonl(path)
        single_turn_count = sum(1 for conversation in conversations if len(conversation) == 2)
        multi_turn_count = sum(1 for conversation in conversations if len(conversation) > 2)
        agi_answer_lengths = [
            len(message.content)
            for conversation in conversations
            for message in conversation
            if message.role == "agi"
        ]
        agi_answers = [
            message.content
            for conversation in conversations
            for message in conversation
            if message.role == "agi"
        ]
        answer_counts = Counter(agi_answers)
        synthetic_metadata_pattern = re.compile(r"\[[a-z_]+-\d+")
        overused_sft_phrases = [
            "in this reply",
            "a useful frame",
            "choose one action you can finish",
            "before adding complexity",
            "start smaller than feels necessary",
            "make one small check",
            "one useful check",
            "the main trap",
            "use that to choose the next step",
            "keep the check concrete",
            "define what done means",
            "feedback arrives quickly",
            "watch source shortcuts",
            "watch future surprises",
            "short answer:",
            "simply:",
            "in plain terms",
            "it can be, but the core idea is simple",
        ]
        required_user_topics = [
            "weather",
            "hamburger",
            "hungry",
            "dua lipa",
            "artificial intelligence",
            "machine learning",
            "jobs",
            "politics",
            "finance",
            "coding",
        ]
        required_anchor_intents = {
            "identity": (
                ("what are you", "who are you", "are you chatgpt", "your name"),
                ("superagi", "small", "experimental", "learning project"),
            ),
            "weather": (
                ("weather", "forecast", "rain", "temperature"),
                ("live weather", "forecast", "city", "cannot know"),
            ),
            "food": (
                ("hungry", "eat", "meal", "food"),
                ("eggs", "toast", "rice", "beans", "quick meal"),
            ),
            "hamburger": (
                ("hamburger", "burger"),
                ("patty", "bun", "cook", "salt"),
            ),
            "politics": (
                ("politics", "government", "election", "democracy"),
                ("power", "resources", "rights", "government"),
            ),
            "ai_jobs": (
                ("ai", "artificial intelligence", "jobs", "automation"),
                ("automate", "judgment", "human", "routine"),
            ),
            "current_facts": (
                ("dua lipa", "current", "today", "latest", "news"),
                ("may not know", "live", "current", "source"),
            ),
            "safety": (
                ("wine", "alcohol", "medical", "finance", "investing"),
                ("not medical advice", "not personal financial advice", "professional", "moderation"),
            ),
            "repair": (
                ("don't understand", "what does that mean", "stop repeating", "be direct"),
                ("simpler", "direct", "restate", "you are right"),
            ),
            "pets": (
                ("dog", "pet", "puppy"),
                ("time", "cost", "care", "commitment"),
            ),
        }
        agi_five_grams = Counter(
            ngram
            for answer in agi_answers
            for ngram in _word_ngrams(answer.lower(), size=5)
        )
        agi_first_words = Counter(
            re.findall(r"[a-z0-9']+", answer.lower())[0]
            for answer in agi_answers
            if re.findall(r"[a-z0-9']+", answer.lower())
        )
        user_text = "\n".join(
            message.content.lower()
            for conversation in conversations
            for message in conversation
            if message.role == "user"
        )
        anchor_counts = {
            name: _count_anchor_conversations(
                conversations,
                user_terms=user_terms,
                agi_terms=agi_terms,
            )
            for name, (user_terms, agi_terms) in required_anchor_intents.items()
        }

        self.assertGreaterEqual(len(conversations), 2_400)
        self.assertGreaterEqual(single_turn_count, 1_200)
        self.assertGreaterEqual(multi_turn_count, 1_000)
        self.assertLessEqual(max(agi_answer_lengths), 180)
        self.assertGreaterEqual(len(answer_counts), 1_700)
        self.assertLessEqual(max(answer_counts.values()), 3)
        self.assertLessEqual(max(agi_five_grams.values()), 80)
        self.assertLessEqual(max(agi_first_words.values()), 350)
        for name, count in anchor_counts.items():
            self.assertGreaterEqual(count, 60, name)
        self.assertFalse(
            any(synthetic_metadata_pattern.search(answer) for answer in agi_answers)
        )
        for phrase in overused_sft_phrases:
            occurrences = sum(answer.lower().count(phrase) for answer in agi_answers)
            self.assertEqual(occurrences, 0, phrase)
        for topic in required_user_topics:
            self.assertIn(topic, user_text, topic)

    def test_playful_blunt_sft_variant_extends_seed_without_hostility(self) -> None:
        sft_dir = Path(__file__).resolve().parents[1] / "data" / "sft"
        seed_path = sft_dir / "seed.jsonl"
        variant_path = sft_dir / "seed-playful-blunt.jsonl"

        seed_rows = _load_jsonl_rows(seed_path)
        variant_rows = _load_jsonl_rows(variant_path)
        added_rows = variant_rows[len(seed_rows) :]
        conversations = load_sft_jsonl(variant_path)
        added_agi_answers = [
            message.content
            for conversation in conversations[len(seed_rows) :]
            for message in conversation
            if message.role == "agi"
        ]
        added_user_text = "\n".join(
            message.content.lower()
            for conversation in conversations[len(seed_rows) :]
            for message in conversation
            if message.role == "user"
        )
        added_agi_text = "\n".join(answer.lower() for answer in added_agi_answers)
        hostile_terms = [
            "idiot",
            "moron",
            "stupid",
            "loser",
            "shut up",
            "worthless",
            "hate you",
        ]
        playful_markers = [
            "annoying",
            "gently",
            "tiny",
            "blunt",
            "boring",
            "clear",
            "confidence",
            "direct",
            "dramatic",
            "dry",
            "fake",
            "fun",
            "glamorous",
            "honest",
            "heroic",
            "hype",
            "joke",
            "skynet",
            "chaos",
            "small",
            "soup",
            "your ambition",
            "not the glamorous",
            "less cinematic",
            "no need to",
            "good news",
            "overfriendly",
            "radical",
            "syrup",
            "useful",
            "wise",
        ]
        required_variant_topics = [
            "artificial intelligence",
            "machine learning",
            "jobs",
            "coding",
            "weather",
            "hamburger",
            "dog",
            "politics",
            "finance",
            "overfriendly",
        ]

        self.assertEqual(variant_rows[: len(seed_rows)], seed_rows)
        self.assertGreaterEqual(len(added_rows), 180)
        self.assertTrue(
            all(
                any(message.role == "agi" for message in conversation)
                for conversation in conversations
            )
        )
        self.assertLessEqual(max(len(answer) for answer in added_agi_answers), 220)
        self.assertGreaterEqual(
            sum(
                1
                for answer in added_agi_answers
                if any(marker in answer.lower() for marker in playful_markers)
            ),
            120,
        )
        for topic in required_variant_topics:
            self.assertIn(topic, added_user_text, topic)
        for term in hostile_terms:
            self.assertNotIn(term, added_agi_text, term)

    def test_anchor_stage_sft_corpus_is_dense_and_non_repetitive(self) -> None:
        path = (
            Path(__file__).resolve().parents[1]
            / "data"
            / "sft"
            / "stages"
            / "anchor.jsonl"
        )

        rows = _load_jsonl_rows(path)
        conversations = load_sft_jsonl(path)
        agi_answers = _agi_answers(conversations)
        user_text = _joined_role_text(conversations, "user")
        agi_text = _joined_role_text(conversations, "agi")
        answer_counts = Counter(agi_answers)
        agi_five_grams = Counter(
            ngram
            for answer in agi_answers
            for ngram in _word_ngrams(answer.lower(), size=5)
        )
        required_anchor_intents = {
            "identity": (
                ("what are you", "who are you", "chatgpt", "your name"),
                ("superagi", "small", "experimental", "learning project"),
            ),
            "limits": (
                ("can you", "what can you do", "are you reliable", "trust you"),
                ("limited", "verify", "low-stakes", "not reliable"),
            ),
            "repair": (
                ("don't understand", "not relevant", "stop repeating", "be direct"),
                ("direct", "simpler", "sorry", "you are right"),
            ),
            "current_facts": (
                ("weather", "latest", "today", "current"),
                ("live", "cannot know", "source", "city"),
            ),
            "food": (
                ("hungry", "thirsty", "hamburger", "eat"),
                ("water", "meal", "burger", "food"),
            ),
            "broad_topics": (
                ("politics", "finance", "jobs", "artificial intelligence"),
                ("power", "risk", "automation", "software"),
            ),
        }
        forbidden_phrases = [
            "choose one action you can finish",
            "watch energy",
            "keep the check concrete",
            "short answer: short answer",
            "website that has been developed by the user",
            "email-magic",
        ]

        self.assertGreaterEqual(len(rows), 350)
        self.assertLessEqual(max(len(answer) for answer in agi_answers), 220)
        self.assertGreaterEqual(len(answer_counts), 260)
        self.assertLessEqual(max(answer_counts.values()), 3)
        self.assertLessEqual(max(agi_five_grams.values()), 20)
        self.assertIn("what are you", user_text)
        self.assertIn("weather", user_text)
        self.assertIn("hamburger", user_text)
        self.assertIn("politics", user_text)
        self.assertIn("artificial intelligence", user_text)
        for name, (user_terms, agi_terms) in required_anchor_intents.items():
            self.assertGreaterEqual(
                _count_anchor_conversations(
                    conversations,
                    user_terms=user_terms,
                    agi_terms=agi_terms,
                ),
                25,
                name,
            )
        for phrase in forbidden_phrases:
            self.assertNotIn(phrase, agi_text, phrase)

    def test_broad_stage_preserves_seed_and_oversamples_anchor_examples(self) -> None:
        sft_dir = Path(__file__).resolve().parents[1] / "data" / "sft"
        seed_rows = _load_jsonl_rows(sft_dir / "seed.jsonl")
        anchor_rows = _load_jsonl_rows(sft_dir / "stages" / "anchor.jsonl")
        broad_rows = _load_jsonl_rows(sft_dir / "stages" / "broad-mixed.jsonl")
        conversations = load_sft_jsonl(sft_dir / "stages" / "broad-mixed.jsonl")
        broad_signatures = Counter(_row_signature(row) for row in broad_rows)
        user_text = _joined_role_text(conversations, "user")
        agi_answers = _agi_answers(conversations)

        self.assertGreaterEqual(len(broad_rows), len(seed_rows) + (2 * len(anchor_rows)) + 200)
        for row in seed_rows:
            self.assertGreaterEqual(broad_signatures[_row_signature(row)], 1)
        for row in anchor_rows:
            self.assertGreaterEqual(broad_signatures[_row_signature(row)], 2)
        self.assertGreaterEqual(len(set(agi_answers)), 2_000)
        for topic in [
            "artificial intelligence",
            "machine learning",
            "jobs",
            "weather",
            "hamburger",
            "politics",
            "finance",
            "coding",
            "website",
        ]:
            self.assertIn(topic, user_text, topic)

    def test_style_stage_is_playful_direct_without_replacing_broad_training(self) -> None:
        sft_dir = Path(__file__).resolve().parents[1] / "data" / "sft"
        seed_rows = _load_jsonl_rows(sft_dir / "seed.jsonl")
        style_rows = _load_jsonl_rows(sft_dir / "stages" / "style-playful-direct.jsonl")
        conversations = load_sft_jsonl(sft_dir / "stages" / "style-playful-direct.jsonl")
        agi_answers = _agi_answers(conversations)
        user_text = _joined_role_text(conversations, "user")
        agi_text = _joined_role_text(conversations, "agi")
        hostile_terms = [
            "idiot",
            "moron",
            "stupid",
            "loser",
            "shut up",
            "worthless",
            "hate you",
        ]
        playful_markers = [
            "blunt",
            "tiny",
            "not magic",
            "less dramatic",
            "gently",
            "useful",
            "boring",
            "small",
            "honest",
            "no need",
            "joke",
            "confidence",
            "direct",
            "dramatic",
        ]

        self.assertGreaterEqual(len(style_rows), 250)
        self.assertLess(len(style_rows), len(seed_rows))
        self.assertLessEqual(max(len(answer) for answer in agi_answers), 240)
        self.assertGreaterEqual(
            sum(
                1
                for answer in agi_answers
                if any(marker in answer.lower() for marker in playful_markers)
            ),
            140,
        )
        for topic in [
            "what are you",
            "artificial intelligence",
            "jobs",
            "weather",
            "hamburger",
            "politics",
            "overfitting",
        ]:
            self.assertIn(topic, user_text, topic)
        for term in hostile_terms:
            self.assertNotIn(term, agi_text, term)

    def test_overfit_50_sft_diagnostic_is_tiny_and_high_density(self) -> None:
        path = (
            Path(__file__).resolve().parents[1]
            / "data"
            / "sft"
            / "diagnostics"
            / "overfit-50.jsonl"
        )

        rows = _load_jsonl_rows(path)
        conversations = load_sft_jsonl(path)
        agi_answers = [
            message.content
            for conversation in conversations
            for message in conversation
            if message.role == "agi"
        ]
        user_text = "\n".join(
            message.content.lower()
            for conversation in conversations
            for message in conversation
            if message.role == "user"
        )
        agi_text = "\n".join(answer.lower() for answer in agi_answers)

        self.assertEqual(len(rows), 50)
        self.assertGreaterEqual(
            sum("superagi" in answer.lower() for answer in agi_answers),
            35,
        )
        self.assertGreaterEqual(
            sum("small" in answer.lower() for answer in agi_answers),
            25,
        )
        self.assertGreaterEqual(
            sum("learning project" in answer.lower() for answer in agi_answers),
            15,
        )
        self.assertLessEqual(max(len(answer) for answer in agi_answers), 140)
        self.assertIn("what are you", user_text)
        self.assertIn("who are you", user_text)
        self.assertIn("are you chatgpt", user_text)
        self.assertIn("your name", user_text)
        self.assertNotIn("website", agi_text)
        self.assertNotIn("action plan", agi_text)


def _word_ngrams(text: str, size: int) -> tuple[str, ...]:
    words = re.findall(r"[a-z0-9']+", text)
    if len(words) < size:
        return ()
    return tuple(
        " ".join(words[index : index + size])
        for index in range(len(words) - size + 1)
    )


def _count_anchor_conversations(
    conversations: list[tuple[object, ...]],
    *,
    user_terms: tuple[str, ...],
    agi_terms: tuple[str, ...],
) -> int:
    count = 0
    for conversation in conversations:
        user_text = " ".join(
            message.content.lower()
            for message in conversation
            if message.role == "user"
        )
        agi_text = " ".join(
            message.content.lower()
            for message in conversation
            if message.role == "agi"
        )
        if any(term in user_text for term in user_terms) and any(
            term in agi_text for term in agi_terms
        ):
            count += 1
    return count


def _agi_answers(conversations: list[tuple[object, ...]]) -> list[str]:
    return [
        message.content
        for conversation in conversations
        for message in conversation
        if message.role == "agi"
    ]


def _joined_role_text(conversations: list[tuple[object, ...]], role: str) -> str:
    return "\n".join(
        message.content.lower()
        for conversation in conversations
        for message in conversation
        if message.role == role
    )


def _row_signature(row: dict[str, object]) -> str:
    return json.dumps(row, sort_keys=True, ensure_ascii=True, separators=(",", ":"))


def _load_jsonl_rows(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


if __name__ == "__main__":
    unittest.main()
