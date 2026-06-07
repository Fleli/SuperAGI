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
            "<bos><user> What are you?\n"
            "<agi> I am a small model.<eos>\n"
            "<user> Can you chat?\n"
            "<agi>",
        )

    def test_extracts_reply_before_next_user_turn(self) -> None:
        prompt = "<bos><user> Hello\n<agi> "
        generated = "<bos><user> Hello\n<agi> Hi there.<user> Another turn"

        reply = extract_chat_reply(prompt, generated)

        self.assertEqual(reply, "Hi there.")

    def test_extracts_reply_before_eos(self) -> None:
        prompt = "<bos><user> Hello\n<agi> "
        generated = "<bos><user> Hello\n<agi> Hi there.<eos>\n<user> Another turn"

        reply = extract_chat_reply(prompt, generated)

        self.assertEqual(reply, "Hi there.")

    def test_extracts_reply_before_legacy_next_user_turn(self) -> None:
        prompt = "User: Hello\nAGI: "
        generated = "User: Hello\nAGI: Hi there.\nUser: Another turn"

        reply = extract_chat_reply(prompt, generated)

        self.assertEqual(reply, "Hi there.")

    def test_extracts_reply_when_tokenizer_roundtrip_changes_prefix(self) -> None:
        prompt = "User: Hello\nAGI: "
        generated = "Something unexpected"

        reply = extract_chat_reply(prompt, generated)

        self.assertEqual(reply, "Something unexpected")


if __name__ == "__main__":
    unittest.main()
