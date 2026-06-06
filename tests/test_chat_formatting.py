import unittest

from superagi.chat.formatting import (
    ChatMessage,
    format_chat_messages,
    format_user_prompt,
)


class ChatFormattingTests(unittest.TestCase):
    def test_formats_user_prompt_for_generation(self) -> None:
        formatted = format_user_prompt("What are you?")

        self.assertEqual(formatted.text, "User: What are you?\nAGI:")
        self.assertEqual(formatted.agi_spans, ())

    def test_formats_chat_messages_with_agi_spans_for_sft(self) -> None:
        formatted = format_chat_messages(
            [
                {"role": "user", "content": "What are you?"},
                {
                    "role": "agi",
                    "content": "I am a small experimental model.",
                },
            ]
        )

        self.assertEqual(
            formatted.text,
            "User: What are you?\nAGI: I am a small experimental model.\n",
        )
        self.assertEqual(
            formatted.text[formatted.agi_spans[0].start : formatted.agi_spans[0].end],
            "I am a small experimental model.",
        )

    def test_accepts_chat_message_instances(self) -> None:
        formatted = format_chat_messages(
            [
                ChatMessage(role="user", content="Explain ML."),
                ChatMessage(role="agi", content="ML learns patterns from data."),
            ]
        )

        self.assertEqual(
            formatted.text,
            "User: Explain ML.\nAGI: ML learns patterns from data.\n",
        )

    def test_can_append_generation_prompt_after_history(self) -> None:
        formatted = format_chat_messages(
            [
                {"role": "user", "content": "What is AI?"},
                {"role": "agi", "content": "AI is software that performs intelligent tasks."},
                {"role": "user", "content": "One sentence?"},
            ],
            add_generation_prompt=True,
        )

        self.assertEqual(
            formatted.text,
            "User: What is AI?\n"
            "AGI: AI is software that performs intelligent tasks.\n"
            "User: One sentence?\n"
            "AGI:",
        )

    def test_rejects_unknown_roles(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown chat role"):
            format_chat_messages([{"role": "assistant", "content": "Nope."}])

    def test_rejects_empty_content(self) -> None:
        with self.assertRaisesRegex(ValueError, "content must not be empty"):
            format_chat_messages([{"role": "user", "content": "   "}])


if __name__ == "__main__":
    unittest.main()
