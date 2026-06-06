import contextlib
import importlib.util
import io
import tempfile
import unittest
from pathlib import Path
from unittest import mock


class ChatScriptTests(unittest.TestCase):
    def test_chat_script_reports_missing_checkpoint_without_traceback(self) -> None:
        chat_script = self._load_chat_script()

        error = io.StringIO()
        with contextlib.redirect_stderr(error):
            exit_code = chat_script.main(
                [
                    "--checkpoint",
                    "missing-chat-model.pt",
                    "--new-tokens",
                    "5",
                    "--device",
                    "cpu",
                ]
            )

        self.assertEqual(exit_code, 1)
        self.assertIn("Missing checkpoint: missing-chat-model.pt", error.getvalue())
        self.assertNotIn("Traceback", error.getvalue())

    def test_chat_script_preserves_history_between_turns(self) -> None:
        chat_script = self._load_chat_script()
        calls = []

        def fake_generate_chat_reply(*, messages, on_text, **kwargs):
            calls.append(tuple((message.role, message.content) for message in messages))
            reply = "First answer." if len(calls) == 1 else "Second answer."
            on_text(reply)
            return reply

        fake_checkpoint = mock.Mock()
        output = io.StringIO()
        with tempfile.TemporaryDirectory() as tmp_dir:
            checkpoint_path = Path(tmp_dir) / "model.pt"
            checkpoint_path.touch()
            with mock.patch.object(chat_script, "load_checkpoint", return_value=fake_checkpoint):
                with mock.patch.object(chat_script, "generate_chat_reply", side_effect=fake_generate_chat_reply):
                    with mock.patch.object(chat_script, "_resolve_device", return_value="cpu"):
                        with mock.patch("builtins.input", side_effect=["Hello", "Again", "/exit"]):
                            with contextlib.redirect_stdout(output):
                                exit_code = chat_script.main(
                                    [
                                        "--checkpoint",
                                        str(checkpoint_path),
                                        "--new-tokens",
                                        "5",
                                        "--device",
                                        "cpu",
                                    ]
                                )

        self.assertEqual(exit_code, 0)
        self.assertEqual(
            calls[0],
            (("user", "Hello"),),
        )
        self.assertEqual(
            calls[1],
            (
                ("user", "Hello"),
                ("agi", "First answer."),
                ("user", "Again"),
            ),
        )
        self.assertIn("AGI: First answer.\n", output.getvalue())
        self.assertIn("AGI: Second answer.\n", output.getvalue())

    def _load_chat_script(self):
        path = Path(__file__).resolve().parents[1] / "scripts" / "chat.py"
        spec = importlib.util.spec_from_file_location("superagi_chat_script", path)
        if spec is None or spec.loader is None:
            raise RuntimeError("could not load scripts/chat.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module


if __name__ == "__main__":
    unittest.main()
