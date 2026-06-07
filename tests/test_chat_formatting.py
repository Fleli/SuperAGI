import unittest

from superagi.chat.formatting import (
    ChatMessage,
    format_chat_messages,
    format_user_prompt,
)


class ChatFormattingTests(unittest.TestCase):
    def test_formats_user_prompt_for_generation(self) -> None:
        formatted = format_user_prompt("What are you?")

        self.assertEqual(formatted.text, "<bos><user> What are you?\n<agi> ")
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
            "<bos><user> What are you?\n<agi> I am a small experimental model.<eos>\n",
        )
        self.assertEqual(
            formatted.text[formatted.agi_spans[0].start : formatted.agi_spans[0].end],
            "I am a small experimental model.<eos>",
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
            "<bos><user> Explain ML.\n<agi> ML learns patterns from data.<eos>\n",
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
            "<bos><user> What is AI?\n"
            "<agi> AI is software that performs intelligent tasks.<eos>\n"
            "<user> One sentence?\n"
            "<agi> ",
        )

    def test_rejects_unknown_roles(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown chat role"):
            format_chat_messages([{"role": "assistant", "content": "Nope."}])

    def test_rejects_empty_content(self) -> None:
        with self.assertRaisesRegex(ValueError, "content must not be empty"):
            format_chat_messages([{"role": "user", "content": "   "}])


if __name__ == "__main__":
    unittest.main()
