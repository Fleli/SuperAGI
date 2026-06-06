import tempfile
import unittest
from pathlib import Path

from superagi.initialization.read_config import load_project_config


class ConfigLoadingTests(unittest.TestCase):
    def test_default_config_maps_to_transformer_config(self) -> None:
        config = load_project_config()
        transformer_config = config.to_transformer_config(vocab_size=100)

        self.assertGreater(config.parameters.n_layers, 0)
        self.assertGreater(config.parameters.dim_embedding, 0)
        self.assertGreater(config.parameters.dim_key, 0)
        self.assertGreater(config.parameters.ctx_window, 0)
        self.assertEqual(
            config.parameters.dim_embedding % config.parameters.dim_key,
            0,
        )
        self.assertEqual(transformer_config.vocab_size, 100)
        self.assertEqual(transformer_config.context_length, config.parameters.ctx_window)
        self.assertEqual(
            transformer_config.dim_embedding,
            config.parameters.dim_embedding,
        )
        self.assertEqual(transformer_config.n_layers, config.parameters.n_layers)
        self.assertEqual(transformer_config.n_heads, config.parameters.n_heads)

    def test_loads_model_config_from_yaml(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "config.yaml"
            config_path.write_text(
                """
init:
  range_min: -1
  range_max: 1

parameters:
  n_layers: 2
  dim_embedding: 12
  dim_key: 4
  ctx_window: 16
""",
                encoding="utf-8",
            )

            config = load_project_config(config_path)
            transformer_config = config.to_transformer_config(vocab_size=10)

        self.assertEqual(config.init.range_min, -1)
        self.assertEqual(config.init.range_max, 1)
        self.assertEqual(config.parameters.n_heads, 3)
        self.assertEqual(transformer_config.vocab_size, 10)
        self.assertEqual(transformer_config.context_length, 16)
        self.assertEqual(transformer_config.dim_embedding, 12)
        self.assertEqual(transformer_config.n_layers, 2)
        self.assertEqual(transformer_config.n_heads, 3)

    def test_rejects_embedding_dimension_not_divisible_by_key_dimension(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "config.yaml"
            config_path.write_text(
                """
init:
  range_min: -1
  range_max: 1

parameters:
  n_layers: 2
  dim_embedding: 10
  dim_key: 4
  ctx_window: 16
""",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "dim_embedding must be divisible"):
                load_project_config(config_path)


if __name__ == "__main__":
    unittest.main()
