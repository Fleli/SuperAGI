# Training Data

Use this directory for local training data and generated artifacts.

- `raw/`: source text files to ingest, such as `.txt` or `.md` files. Corpus builders write into source-specific subdirectories like `raw/wikipedia/` and `raw/c4/`.
- `processed/`: generated token tensors and tokenizer metadata.
- `checkpoints/`: generated model checkpoints.
- `sft/`: small supervised fine-tuning corpora and evaluation prompts for chat behavior.

Generated `processed/` and `checkpoints/` contents are ignored by Git. Keep large raw corpora out of Git unless they are intentionally tiny examples.
