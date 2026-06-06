import unittest

from superagi.chat.formatting import ChatMessage
from superagi.chat.session import build_chat_prompt, extract_chat_reply


class ChatSessionTests(unittest.TestCase):
    def test_builds_generation_prompt_from_chat_history(self) -> None:
        prompt = build_chat_prompt(
            [
                ChatMessage(role="user", content="What are you?"),
                ChatMessage(role="agi", content="I am a small model."),
                ChatMessage(role="user", content="Can you chat?"),
            ]
        )

        self.assertEqual(
            prompt,
            "User: What are you?\n"
            "AGI: I am a small model.\n"
            "User: Can you chat?\n"
            "AGI:",
        )

    def test_extracts_reply_before_next_user_turn(self) -> None:
        prompt = "User: Hello\nAGI: "
        generated = "User: Hello\nAGI: Hi there.\nUser: Another turn"

        reply = extract_chat_reply(prompt, generated)

        self.assertEqual(reply, "Hi there.")

    def test_extracts_reply_before_extra_newline_text(self) -> None:
        prompt = "User: Hello\nAGI: "
        generated = "User: Hello\nAGI: Hi there.\nThis should not become chat history."

        reply = extract_chat_reply(prompt, generated)

        self.assertEqual(reply, "Hi there.")

    def test_extracts_reply_when_tokenizer_roundtrip_changes_prefix(self) -> None:
        prompt = "User: Hello\nAGI: "
        generated = "Something unexpected"

        reply = extract_chat_reply(prompt, generated)

        self.assertEqual(reply, "Something unexpected")


if __name__ == "__main__":
    unittest.main()
