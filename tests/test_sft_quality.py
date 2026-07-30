import unittest

from superagi.chat.formatting import ChatMessage
from superagi.chat.sft_quality import (
    canonical_text,
    conversation_fingerprint,
    conversation_group_key,
    validate_role_sequence,
)


class SftQualityTests(unittest.TestCase):
    def test_rejects_conversation_that_does_not_start_with_user_or_system(self) -> None:
        messages = (ChatMessage(role="agi", content="Hello."),)
        with self.assertRaisesRegex(ValueError, "expected 'user'"):
            validate_role_sequence(messages)

    def test_rejects_consecutive_user_messages(self) -> None:
        messages = (
            ChatMessage(role="user", content="First."),
            ChatMessage(role="user", content="Second."),
        )
        with self.assertRaisesRegex(ValueError, "expected 'agi'"):
            validate_role_sequence(messages)

    def test_rejects_conversation_that_does_not_end_with_agi(self) -> None:
        messages = (ChatMessage(role="user", content="Hello?"),)
        with self.assertRaisesRegex(ValueError, "end with an agi response"):
            validate_role_sequence(messages)

    def test_group_key_matches_case_and_punctuation_variants(self) -> None:
        first = (
            ChatMessage(role="user", content="What is AI?"),
            ChatMessage(role="agi", content="AI is software."),
        )
        second = (
            ChatMessage(role="user", content="WHAT IS AI"),
            ChatMessage(role="agi", content="AI is software!"),
        )
        self.assertEqual(conversation_group_key(first), conversation_group_key(second))

    def test_group_key_differs_for_unrelated_conversations(self) -> None:
        first = (
            ChatMessage(role="user", content="What is AI?"),
            ChatMessage(role="agi", content="AI is software."),
        )
        second = (
            ChatMessage(role="user", content="How do tides work?"),
            ChatMessage(role="agi", content="Gravity moves ocean water."),
        )
        self.assertNotEqual(conversation_group_key(first), conversation_group_key(second))

    def test_canonical_text_normalizes_unicode_whitespace_and_punctuation(self) -> None:
        self.assertEqual(canonical_text("\uff37\uff48\uff41\uff54\t is\nAI?!"), "what is ai")

    def test_fingerprint_includes_the_system_message(self) -> None:
        first = (
            ChatMessage(role="system", content="Be concise."),
            ChatMessage(role="user", content="Hello"),
            ChatMessage(role="agi", content="Hi."),
        )
        second = (
            ChatMessage(role="system", content="Be detailed."),
            ChatMessage(role="user", content="Hello"),
            ChatMessage(role="agi", content="Hi."),
        )
        self.assertNotEqual(conversation_fingerprint(first), conversation_fingerprint(second))


if __name__ == "__main__":
    unittest.main()
