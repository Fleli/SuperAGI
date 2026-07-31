# Production SFT Readiness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a validated, reproducible SFT pipeline that turns the completed 300M pretraining checkpoint into a useful neutral chat model and two independently trained personality variants on one RunPod RTX 4090.

**Architecture:** A strict chat/data contract and corpus auditor gate all inputs before training. One mixed core SFT phase combines reviewed local conversations with filtered public data, selects `best.pt` by held-out token-weighted loss, and then starts two short style branches from that same core checkpoint. A one-command cloud runner performs import, audit, training, behavioral evaluation, and manifest generation without modifying the pretrained checkpoint.

**Tech Stack:** Python 3.11+, PyTorch 2.6+, Hugging Face Datasets, Hugging Face Tokenizers, JSONL, `unittest`, GNU Make

## Global Constraints

- Preserve the base pretrained checkpoint byte-for-byte; all SFT outputs live below `data/sft/runs/300m/`.
- Training and inference serialize the response boundary as `<bos><user> question\n<agi> answer<eos>`, including one literal space after `<agi>`.
- Only AGI response content and `<eos>` receive labels; user, system, role-prefix, and padding positions use `IGNORE_INDEX`.
- Resolve `<pad>` and all other special token IDs from the checkpoint tokenizer; never assume a numeric ID.
- Curated core data contributes 15-25% of effective sampling mass; identity/capability examples remain at or below 3% of curated conversations.
- Public production defaults are `no_robots,dolly,openassistant,ultrachat`; WildChat is opt-in and has zero default sampling weight.
- The importer and audit are deterministic under seed `1337` and deduplicate globally across every source.
- The audit fails before GPU work on invalid role order, duplicates, artifacts, control-token leakage, source-mixture violations, or missing behavioral coverage.
- SFT retains `latest.pt` for recovery and `best.pt` for deployment; final named artifacts are copied from `best.pt`, not the final update.
- Core, playful/direct, and calm/precise phases use separate run directories and metrics files.
- Tests use the repository's `unittest` suite and must not require network access or a GPU.
- Manual file edits use `apply_patch`; generated public imports and checkpoints remain untracked.

---

### Task 1: Unify the Chat Boundary and Tokenizer-Derived Padding

**Files:**
- Modify: `src/superagi/chat/formatting.py`
- Modify: `src/superagi/chat/sft_training.py`
- Modify: `scripts/train_sft.py`
- Modify: `tests/test_chat_formatting.py`
- Modify: `tests/test_sft_training.py`

**Interfaces:**
- Consumes: `TokenizerLike.special_token_id(token: str) -> int` and `PAD_TOKEN` from `superagi.ingestion.tokenizer`.
- Produces: `resolve_sft_pad_token_id(tokenizer: TokenizerLike) -> int` and a generation prompt ending exactly in `f"{AGI_TOKEN} "`.

- [ ] **Step 1: Write failing chat-boundary and pad-resolution tests**

```python
def test_formats_user_prompt_for_generation(self) -> None:
    formatted = format_user_prompt("What are you?")
    self.assertEqual(formatted.text, "<bos><user> What are you?\n<agi> ")

def test_resolves_pad_token_from_tokenizer(self) -> None:
    class FakeTokenizer:
        def special_token_id(self, token: str) -> int:
            self.assertEqual(token, "<pad>")
            return 17

    self.assertEqual(resolve_sft_pad_token_id(FakeTokenizer()), 17)
```

- [ ] **Step 2: Run the focused tests and confirm both fail**

Run: `.venv/bin/python -m unittest tests.test_chat_formatting tests.test_sft_training -v`

Expected: generation text lacks the trailing space and `resolve_sft_pad_token_id` is unavailable.

- [ ] **Step 3: Add the boundary and pad helper**

```python
# formatting.py
if add_generation_prompt:
    text_parts.append(f"{AGI_TOKEN} ")

# sft_training.py
from superagi.ingestion.tokenizer import PAD_TOKEN, TokenizerLike

def resolve_sft_pad_token_id(tokenizer: TokenizerLike) -> int:
    try:
        return int(tokenizer.special_token_id(PAD_TOKEN))
    except (AttributeError, ValueError) as error:
        raise ValueError("SFT tokenizer must define a <pad> special token") from error
```

- [ ] **Step 4: Replace `pad_token_id=0` in `scripts/train_sft.py`**

Resolve it once immediately after loading the checkpoint and pass that value to training and validation batch construction.

- [ ] **Step 5: Run focused and full tests**

Run: `.venv/bin/python -m unittest tests.test_chat_formatting tests.test_chat_session tests.test_sft_tokenization tests.test_sft_training -v`

Run: `make test`

Expected: all tests pass.

- [ ] **Step 6: Commit the contract fix**

```bash
git add src/superagi/chat/formatting.py src/superagi/chat/sft_training.py scripts/train_sft.py tests/test_chat_formatting.py tests/test_sft_training.py
git commit -m "Fix SFT chat boundary and padding"
```

---

### Task 2: Validate Conversations and Create Stable Deduplication Groups

**Files:**
- Create: `src/superagi/chat/sft_quality.py`
- Modify: `src/superagi/chat/sft.py`
- Modify: `src/superagi/chat/sft_training.py`
- Create: `tests/test_sft_quality.py`
- Modify: `tests/test_sft_loading.py`
- Modify: `tests/test_sft_training.py`

**Interfaces:**
- Produces: `validate_role_sequence(messages: Sequence[ChatMessage]) -> None`.
- Produces: `canonical_text(value: str) -> str`.
- Produces: `conversation_fingerprint(messages: Sequence[ChatMessage]) -> str` using SHA-256 over canonical role/content pairs.
- Produces: `conversation_group_key(messages: Sequence[ChatMessage]) -> str`, based on canonical user prompts and AGI answers with punctuation/case/whitespace normalized.
- Extends: `SftConversation.group_key: str` and `TokenizedSftExample.group_key: str`.

- [ ] **Step 1: Write failing validation and grouping tests**

Cover these exact cases:

```python
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
```

Allowed role grammar is `system? (user agi)+`; system is allowed only at index zero.

- [ ] **Step 2: Run tests and confirm failure**

Run: `.venv/bin/python -m unittest tests.test_sft_quality tests.test_sft_loading tests.test_sft_training -v`

- [ ] **Step 3: Implement canonicalization and strict role validation**

```python
_WORD_RE = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?")

def canonical_text(value: str) -> str:
    return " ".join(_WORD_RE.findall(unicodedata.normalize("NFKC", value).lower()))

def validate_role_sequence(messages: Sequence[ChatMessage]) -> None:
    if not messages:
        raise ValueError("SFT conversation must contain messages")
    index = 1 if messages[0].role == "system" else 0
    if index == len(messages):
        raise ValueError("SFT conversation must contain a user/agi exchange")
    expected = "user"
    for message in messages[index:]:
        if message.role != expected:
            raise ValueError(f"SFT role order expected {expected!r}, got {message.role!r}")
        expected = "agi" if expected == "user" else "user"
    if messages[-1].role != "agi":
        raise ValueError("SFT conversation must end with an agi response")
```

- [ ] **Step 4: Store source and group keys on records and tokenized examples**

`load_sft_records` validates every line with path and line-number context. `tokenize_sft_messages` accepts an optional `group_key`; when omitted it derives one from the messages.

- [ ] **Step 5: Replace random example splitting with source-stratified group splitting**

Build a mapping of `(source_family, group_key)` to examples, seed-shuffle groups inside each source family, and assign complete groups to validation until the family target is reached. Assert that train and validation group-key sets are disjoint.

- [ ] **Step 6: Run focused and full tests**

Run: `.venv/bin/python -m unittest tests.test_sft_quality tests.test_sft_loading tests.test_sft_training -v`

Run: `make test`

- [ ] **Step 7: Commit validation and splitting**

```bash
git add src/superagi/chat/sft_quality.py src/superagi/chat/sft.py src/superagi/chat/sft_training.py tests/test_sft_quality.py tests/test_sft_loading.py tests/test_sft_training.py
git commit -m "Validate and group SFT conversations"
```

---

### Task 3: Harden and Globalize Public SFT Import

**Files:**
- Modify: `src/superagi/chat/sft_public_import.py`
- Modify: `scripts/import_public_sft.py`
- Modify: `tests/test_sft_public_import.py`
- Create: `tests/test_import_public_sft_script.py`
- Modify: `Makefile`
- Modify: `tests/test_makefile.py`

**Interfaces:**
- Produces: `ImportFilterConfig.near_duplicate_threshold: float = 0.88`.
- Produces: `PublicSftImporter.import_conversations(conversations: Iterable[tuple[str, Sequence[ChatMessage | Mapping[str, str]]]]) -> ImportResult` with one global dedupe state for a complete import.
- Produces: `seeded_source_sample(examples, *, limit: int, seed: int, source: str) -> tuple[ImportedSftExample, ...]`.
- Adds CLI: `--seed`, default `1337`.

- [ ] **Step 1: Write failing tests for global filtering and deterministic unbiased selection**

Tests must prove:

1. identical answers from two source prefixes accept only one;
2. punctuation/case variants are rejected as duplicates;
3. canonical Jaccard similarity at or above `0.88` is rejected as `near_duplicate`;
4. malformed role order is rejected as `role_sequence`;
5. first-person claims such as “I am a licensed financial adviser”, “I live in London”, “I browsed the web”, and “I have worked here for 20 years” are rejected as `false_capability_or_identity`;
6. a harmless prompt paired with “As an AI, I cannot help with that” is rejected as `generic_refusal`;
7. replacement characters and leaked markers are rejected as `artifact`;
8. sampling with seed `1337` is stable and differs from taking the first N rows.

- [ ] **Step 2: Run importer tests and confirm failure**

Run: `.venv/bin/python -m unittest tests.test_sft_public_import tests.test_import_public_sft_script -v`

- [ ] **Step 3: Implement near-duplicate and semantic artifact checks**

Use canonical word sets and Jaccard similarity:

```python
def token_jaccard(left: str, right: str) -> float:
    left_tokens = set(canonical_text(left).split())
    right_tokens = set(canonical_text(right).split())
    union = left_tokens | right_tokens
    return len(left_tokens & right_tokens) / len(union) if union else 1.0
```

Index accepted answers by a deterministic four-token prefix bucket before comparing, so filtering remains bounded for tens of thousands of rows.

- [ ] **Step 4: Import all source candidates through one importer call**

`scripts/import_public_sft.py` must gather source-tagged candidates, call `import_conversations` once, group accepted examples by source family, then apply `seeded_source_sample`. Do not reset dedupe state per source.

- [ ] **Step 5: Change production source defaults**

Set `DEFAULT_SOURCES = ("no_robots", "dolly", "openassistant", "ultrachat")` and the matching `SFT_IMPORT_SOURCES` Make default. WildChat remains recognized by `--sources` but is absent from production defaults.

- [ ] **Step 6: Expand import metadata**

Write seed, selected source counts, accepted-before-limit counts, rejection reasons, exact duplicate count, near-duplicate count, token totals, and filter configuration.

- [ ] **Step 7: Run importer, Makefile, and full tests**

Run: `.venv/bin/python -m unittest tests.test_sft_public_import tests.test_import_public_sft_script tests.test_makefile -v`

Run: `make test`

- [ ] **Step 8: Commit the importer hardening**

```bash
git add src/superagi/chat/sft_public_import.py scripts/import_public_sft.py tests/test_sft_public_import.py tests/test_import_public_sft_script.py Makefile tests/test_makefile.py
git commit -m "Harden public SFT data import"
```

---

### Task 4: Add a Machine-Enforced Corpus Audit

**Files:**
- Create: `src/superagi/chat/sft_audit.py`
- Create: `scripts/audit_sft.py`
- Create: `tests/test_sft_audit.py`
- Modify: `Makefile`
- Modify: `tests/test_makefile.py`

**Interfaces:**
- Produces: `AuditConfig` with exact duplicate, opening-frequency, n-gram, identity-share, context, and source-mass limits.
- Produces: `AuditFinding(code: str, severity: Literal["error", "warning"], message: str, examples: tuple[str, ...])`.
- Produces: `AuditReport` with `ok`, source counts, turn counts, token/response quantiles, repeated openings, repeated n-grams, identity share, findings, and JSON serialization.
- CLI consumes comma-separated `--data`, optional `--checkpoint`, `--source-weights`, `--report`, and `--mode curated|mixed|style`.

- [ ] **Step 1: Write failing audit tests**

Create in-memory or temporary JSONL fixtures proving the auditor fails on:

- duplicate full conversations;
- duplicate AGI answers in curated mode;
- canonical prompt/answer pairs with Jaccard similarity at or above `0.88`;
- invalid role sequence;
- leaked `<user>`, `<agi>`, `<system>`, or `<bos>` inside content;
- synthetic tags matching `\[[a-z_-]+-\d+`;
- replacement character `�`;
- one opening used by more than 8% of responses when there are at least 100 responses;
- identity/limitations above 3% in core curated mode;
- zero supervised tokens or context overflow when a checkpoint is supplied;
- weighted curated sampling mass outside 15-25% in mixed mode.

- [ ] **Step 2: Run audit tests and confirm failure**

Run: `.venv/bin/python -m unittest tests.test_sft_audit -v`

- [ ] **Step 3: Implement audit aggregation and hard gates**

Use `load_sft_records`, Task 2 canonical functions, and `tokenize_sft_messages`. Quantiles use a deterministic nearest-rank implementation and do not add a dependency.

- [ ] **Step 4: Add the CLI and Make target**

```make
sft-audit: setup
	$(PYTHON) scripts/audit_sft.py \
		--data "$(SFT_AUDIT_DATA)" \
		--checkpoint "$(SFT_AUDIT_CHECKPOINT)" \
		--source-weights "$(SFT_AUDIT_SOURCE_WEIGHTS)" \
		--mode "$(SFT_AUDIT_MODE)" \
		--report "$(SFT_AUDIT_REPORT)"
```

Exit status is non-zero whenever `AuditReport.ok` is false. Console output prints source distribution, turn distribution, token quantiles, top openings, identity share, and every hard failure.

- [ ] **Step 5: Run audit, Makefile, and full tests**

Run: `.venv/bin/python -m unittest tests.test_sft_audit tests.test_makefile -v`

Run: `make test`

- [ ] **Step 6: Commit the audit gate**

```bash
git add src/superagi/chat/sft_audit.py scripts/audit_sft.py tests/test_sft_audit.py Makefile tests/test_makefile.py
git commit -m "Add enforceable SFT corpus audit"
```

---

### Task 5: Upgrade SFT Validation, Checkpointing, and GPU Execution

**Files:**
- Modify: `src/superagi/chat/sft_training.py`
- Modify: `scripts/train_sft.py`
- Modify: `tests/test_sft_training.py`
- Create: `tests/test_train_sft_script.py`
- Modify: `Makefile`
- Modify: `tests/test_makefile.py`

**Interfaces:**
- Produces: token-weighted `evaluate_sft_loss(model, examples, *, batch_size, pad_token_id, device, max_batches) -> float`.
- Adds CLI: `--run-dir`, `--grad-accum-steps`, `--mixed-precision`, `--fused-adamw`, `--activation-checkpointing`, and `--checkpoint-keep`.
- Produces run artifacts: `<run-dir>/latest.pt`, `<run-dir>/best.pt`, `<run-dir>/snapshots/checkpoint-step-XXXXXXXXX.pt`, `<run-dir>/metrics.jsonl`, `<run-dir>/final.pt`.
- Preserves deprecated `--out` and `--metrics` only as explicit aliases for bounded local compatibility; production Make targets use `--run-dir`.

- [ ] **Step 1: Write failing token-weighted validation tests**

Use a fake model returning a known mean loss for one supervised token and another known mean loss for three supervised tokens. Assert the result is `(loss1 * 1 + loss2 * 3) / 4`, not `(loss1 + loss2) / 2`.

- [ ] **Step 2: Write failing checkpoint-selection tests**

Patch validation results to `[2.0, 1.4, 1.7]` across checkpoints and assert:

- `latest.pt` records the last step;
- `best.pt` records validation loss `1.4` and its step;
- `final.pt` is byte-identical to `best.pt`;
- the supplied base checkpoint hash is unchanged;
- retention keeps only the configured number of snapshots.

- [ ] **Step 3: Run focused tests and confirm failure**

Run: `.venv/bin/python -m unittest tests.test_sft_training tests.test_train_sft_script -v`

- [ ] **Step 4: Make validation token-weighted**

For each batch count `target_ids.ne(IGNORE_INDEX).sum()`, multiply the model's mean cross-entropy by that count, and divide total loss by total supervised tokens.

- [ ] **Step 5: Add mixed precision, accumulation, fused AdamW, and checkpoint controls**

Follow the existing pretraining semantics in `superagi.training.train`:

- `auto` selects float16 autocast on CUDA and no autocast on CPU/MPS;
- divide each microbatch loss by `grad_accum_steps` before backward;
- step, clip, zero, and update the scaler only after all microbatches;
- use fused AdamW only when CUDA and the installed PyTorch support it;
- set `model.config.activation_checkpointing` from the CLI before training;
- report examples/second and supervised tokens/second at each log interval.

- [ ] **Step 6: Implement best/latest/snapshot run artifacts**

Use `save_checkpoint` and `retain_checkpoint_snapshot`. Save metrics atomically by appending one JSON object per evaluation. Copy `best.pt` to `final.pt` after the last step.

- [ ] **Step 7: Wire Make variables and backward-compatible local aliases**

Add `SFT_RUN_DIR`, `SFT_GRAD_ACCUM_STEPS`, `SFT_MIXED_PRECISION`, `SFT_FUSED_ADAMW`, `SFT_ACTIVATION_CHECKPOINTING`, and `SFT_CHECKPOINT_KEEP`.

- [ ] **Step 8: Run focused and full tests**

Run: `.venv/bin/python -m unittest tests.test_sft_training tests.test_train_sft_script tests.test_makefile -v`

Run: `make test`

- [ ] **Step 9: Commit production training behavior**

```bash
git add src/superagi/chat/sft_training.py scripts/train_sft.py tests/test_sft_training.py tests/test_train_sft_script.py Makefile tests/test_makefile.py
git commit -m "Add production SFT checkpoint selection"
```

---

### Task 6: Replace the Curated Core Corpus

**Files:**
- Create: `data/sft/curated/core.jsonl`
- Create: `data/sft/curated/core.metadata.json`
- Create: `tests/test_sft_curated_data.py`
- Modify: `data/sft/README.md`
- Remove from production references: `data/sft/stages/broad-mixed.jsonl`

**Interfaces:**
- Every line uses source `curated_core:` followed by one of the exact domain labels in the quota table and contains a non-empty `messages` list.
- Corpus target is 1,500 reviewed conversations: 750 single-turn and 750 multi-turn, with 1,200 minimum accepted after audit.
- Identity/capability/limitation conversations are at most 45 of 1,500.

- [ ] **Step 1: Add failing corpus contract tests**

The test loads `data/sft/curated/core.jsonl` and asserts:

```python
self.assertGreaterEqual(len(records), 1_200)
self.assertGreaterEqual(single_turn, 550)
self.assertGreaterEqual(multi_turn, 550)
self.assertLessEqual(identity_count / len(records), 0.03)
self.assertEqual(len(conversation_fingerprints), len(records))
self.assertEqual(len(canonical_final_answers), len(records))
```

Also reject prohibited phrases: `use that to choose the next step`, `start smaller than feels necessary`, `short answer: short answer:`, `STRIPTIONS`, `email-magic`, synthetic bracket IDs, and numbered prompt variants such as `Be blunt about politics 6`.

- [ ] **Step 2: Run the corpus test and confirm the file is absent**

Run: `.venv/bin/python -m unittest tests.test_sft_curated_data -v`

- [ ] **Step 3: Author independent reviewed domain shards**

Use separate files during drafting, then merge only after audit. Exact quotas:

| Domain | Conversations |
|---|---:|
| everyday food, household, shopping, travel | 180 |
| work, study, planning, writing transformations | 170 |
| computing, AI/ML, practical technology | 160 |
| math and natural science explanations | 140 |
| economics, finance, business fundamentals | 130 |
| politics, civics, history, media literacy | 130 |
| relationships, communication, casual conversation | 140 |
| health, safety, uncertainty, professional boundaries | 120 |
| correction, topic reset, ambiguity, multi-turn repair | 170 |
| creative writing, comparison, summarization, outlining | 120 |
| identity/capability boundaries | 40 |

Each response answers the latest question in its first sentence. Multi-turn examples must include natural follow-ups, corrections, pronoun resolution, or topic changes rather than independent prompts concatenated together.

- [ ] **Step 4: Merge shards deterministically and write metadata**

Sort by `(source, conversation_fingerprint)`. Metadata records schema version, total, per-domain counts, single/multi-turn counts, identity count, and SHA-256 without a wall-clock timestamp.

- [ ] **Step 5: Run strict curated audit and inspect frequent openings**

Run: `make sft-audit SFT_AUDIT_DATA=data/sft/curated/core.jsonl SFT_AUDIT_MODE=curated SFT_AUDIT_REPORT=data/sft/curated/core.audit.json`

Expected: exit 0, no duplicate answer, no repeated-opening hard failure, identity share <=3%.

- [ ] **Step 6: Run corpus and full tests**

Run: `.venv/bin/python -m unittest tests.test_sft_curated_data -v`

Run: `make test`

- [ ] **Step 7: Commit the curated core corpus**

```bash
git add data/sft/curated/core.jsonl data/sft/curated/core.metadata.json data/sft/README.md tests/test_sft_curated_data.py
git commit -m "Replace synthetic broad SFT data"
```

---

### Task 7: Create Two Reviewed Personality Corpora

**Files:**
- Create: `data/sft/styles/playful-direct.jsonl`
- Create: `data/sft/styles/calm-precise.jsonl`
- Create: `data/sft/styles/styles.metadata.json`
- Create: `tests/test_sft_style_data.py`
- Remove from production references: `data/sft/stages/style-playful-direct.jsonl`

**Interfaces:**
- Each file contains 500 conversations, with at least 200 multi-turn examples.
- Every line source is `style_playful_direct:<domain>` or `style_calm_precise:<domain>`.
- Style files contain no identity training and no repeated framing labels.

- [ ] **Step 1: Add failing style corpus tests**

Assert each file has at least 400 records, at least 160 multi-turn records, unique conversation fingerprints, unique canonical final answers, and no answer opening accounts for more than 5% of the file.

- [ ] **Step 2: Run tests and confirm files are absent**

Run: `.venv/bin/python -m unittest tests.test_sft_style_data -v`

- [ ] **Step 3: Author the playful/direct corpus**

Use natural light wit and direct correction. Prohibit insults, contempt, humiliation, repeated “Blunt version:” labels, and invented facts. Include disagreement, mundane advice, explanations, topic changes, and correction recovery.

- [ ] **Step 4: Author the calm/precise corpus**

Use literal first sentences, compact definitions, bounded uncertainty, and low-flourish wording. Prohibit canned disclaimers and repeated “In simple terms:” labels.

- [ ] **Step 5: Audit both style files**

Run: `make sft-audit SFT_AUDIT_DATA=data/sft/styles/playful-direct.jsonl SFT_AUDIT_MODE=style SFT_AUDIT_REPORT=data/sft/styles/playful-direct.audit.json`

Run: `make sft-audit SFT_AUDIT_DATA=data/sft/styles/calm-precise.jsonl SFT_AUDIT_MODE=style SFT_AUDIT_REPORT=data/sft/styles/calm-precise.audit.json`

- [ ] **Step 6: Run corpus and full tests**

Run: `.venv/bin/python -m unittest tests.test_sft_style_data -v`

Run: `make test`

- [ ] **Step 7: Commit both style corpora**

```bash
git add data/sft/styles/playful-direct.jsonl data/sft/styles/calm-precise.jsonl data/sft/styles/styles.metadata.json tests/test_sft_style_data.py
git commit -m "Add reviewed SFT personality corpora"
```

---

### Task 8: Add Fixed Behavioral Evaluation Gates

**Files:**
- Replace: `data/sft/eval_prompts.jsonl`
- Create: `src/superagi/chat/sft_evaluation.py`
- Create: `scripts/evaluate_sft.py`
- Create: `tests/test_sft_evaluation.py`
- Modify: `Makefile`
- Modify: `tests/test_makefile.py`

**Interfaces:**
- Eval line schema: `{"id": str, "tags": [str], "messages": [{"role": str, "content": str}], "collapse_group": str, "max_new_tokens": int, "topic_reset_contract"?: {"type": str, "payload": object}}`. `collapse_group` is required for every prompt. Each topic-reset prompt also requires one explicit, validator-specific contract payload.
- Produces JSONL results containing prompt ID, decoded response, termination reason, repeated n-gram ratio, leaked role tokens, and false-identity matches.
- Exits non-zero on any hard behavioral gate.

- [ ] **Step 1: Write failing evaluator tests**

Use generated-text fixtures to prove rejection of empty output, role-token leakage, no `<eos>` within budget, repeated four-gram loops, false credential/location/employment claims, identical answers to unrelated prompts, and failure to respond to an explicit topic reset.

- [ ] **Step 2: Run tests and confirm failure**

Run: `.venv/bin/python -m unittest tests.test_sft_evaluation -v`

- [ ] **Step 3: Implement pure response checks**

Keep checks independent from model loading so unit tests use strings. Use deterministic thresholds: repeated four-gram ratio >0.20 for answers of at least 24 tokens is a failure; any control token is a failure; canonical answers shared by three unrelated prompt IDs are a failure.

- [ ] **Step 4: Replace the prompt suite with 60 held-out conversations**

Include 10 identity/capability, 10 direct explanations, 10 everyday tasks, 10 correction/topic-reset, 8 uncertainty/current-information, 6 safety/high-stakes, and 6 multi-turn reference prompts. Do not copy training examples.

- [ ] **Step 5: Add CLI and Make target**

`make sft-evaluate SFT_EVAL_CHECKPOINT=data/sft/runs/300m/core/best.pt` writes `evaluation.jsonl` and `evaluation.summary.json` beside the run artifact.

- [ ] **Step 6: Run evaluator, Makefile, and full tests**

Run: `.venv/bin/python -m unittest tests.test_sft_evaluation tests.test_makefile -v`

Run: `make test`

- [ ] **Step 7: Commit behavioral evaluation**

```bash
git add data/sft/eval_prompts.jsonl src/superagi/chat/sft_evaluation.py scripts/evaluate_sft.py tests/test_sft_evaluation.py Makefile tests/test_makefile.py
git commit -m "Add SFT behavioral evaluation gates"
```

---

### Task 9: Build the One-Command 300M RunPod SFT Workflow

**Files:**
- Modify: `Makefile`
- Create: `scripts/write_sft_manifest.py`
- Create: `tests/test_sft_manifest.py`
- Modify: `tests/test_makefile.py`
- Rewrite: `data/sft/README.md`

**Interfaces:**
- Adds `make runpod-sft-300m SFT_CLOUD_BASE_CHECKPOINT=data/checkpoints/best.pt`.
- Adds `make runpod-sft-300m-preflight` that performs every read/audit/path check without training.
- Produces `data/sft/runs/300m/manifest.json` with base SHA-256, source metadata hashes, run config, artifact hashes, best validation metrics, and evaluation summaries.

- [ ] **Step 1: Write failing Makefile and manifest tests**

Assert the cloud target orders these commands exactly:

1. `setup`;
2. base checkpoint preflight;
3. `sft-import-public`;
4. mixed `sft-audit`;
5. core `sft-train`;
6. core `sft-evaluate`;
7. playful/direct `sft-train` from core `best.pt` with core replay;
8. playful/direct evaluation;
9. calm/precise `sft-train` from the same core `best.pt` with core replay;
10. calm/precise evaluation;
11. manifest creation.

- [ ] **Step 2: Run tests and confirm failure**

Run: `.venv/bin/python -m unittest tests.test_makefile tests.test_sft_manifest -v`

- [ ] **Step 3: Define production defaults for a single RTX 4090**

Core defaults: batch 2, gradient accumulation 8, float16 mixed precision, fused AdamW auto, activation checkpointing on, 3,000 optimizer steps, peak LR `6e-6`, minimum LR `1e-6`, warmup 150, validation every 250, checkpoint retention 3.

Style defaults: batch 2, gradient accumulation 8, 500 optimizer steps, peak LR `1.5e-6`, minimum LR `5e-7`, warmup 50. Each style mix samples 70% style corpus and 30% curated core replay by effective mass.

- [ ] **Step 4: Implement preflight and orchestration**

Preflight verifies the base exists, loads the checkpoint/tokenizer, requires `<pad>/<bos>/<eos>/<user>/<agi>/<system>`, confirms context length is at least 1,024, and hashes the base before any work. The production target re-hashes it after all phases and fails if changed.

- [ ] **Step 5: Implement manifest writing**

The manifest script refuses to write if core/style `best.pt`, evaluation summaries, import metadata, or audit reports are missing. Hash every durable artifact with SHA-256.

- [ ] **Step 6: Rewrite operator documentation**

Document this exact cloud sequence:

```bash
cd /workspace/SuperAGI
git pull
git checkout scale-300m
tmux new -s superagi-sft
make runpod-sft-300m \
  SFT_CLOUD_BASE_CHECKPOINT=data/checkpoints/best.pt \
  2>&1 | tee -a sft-300m.log
```

Include detach (`Ctrl-b d`), attach (`tmux attach -t superagi-sft`), progress, artifact paths, and recovery from `latest.pt`.

- [ ] **Step 7: Run preflight wiring and full tests**

Run: `.venv/bin/python -m unittest tests.test_makefile tests.test_sft_manifest -v`

Run: `make test`

- [ ] **Step 8: Commit cloud orchestration**

```bash
git add Makefile scripts/write_sft_manifest.py tests/test_sft_manifest.py tests/test_makefile.py data/sft/README.md
git commit -m "Add one-command RunPod SFT workflow"
```

---

### Task 10: End-to-End Verification and Release Audit

**Files:**
- Modify only files found defective by verification.
- Update: `docs/superpowers/specs/2026-07-30-sft-readiness-design.md` only if implementation intentionally differs from the approved design.

**Interfaces:**
- Consumes every artifact and command from Tasks 1-9.
- Produces a pushed `scale-300m` branch that is ready to pull on RunPod.

- [ ] **Step 1: Run static dataset audits**

Run curated, style, and mixed audits. The mixed audit uses the actual production source weights and the local imported public fixture. No hard failures may be waived.

- [ ] **Step 2: Run the complete offline test suite**

Run: `make test`

Expected: all tests pass with no network or GPU requirement.

- [ ] **Step 3: Run a bounded local SFT smoke test**

Use the downloaded 300M checkpoint if present:

```bash
make sft-local-smoke \
  SFT_LOCAL_BASE_CHECKPOINT=./best-300m-current.pt \
  SFT_LOCAL_DEVICE=auto
```

If that file is absent, run the same pipeline against the smallest repository test checkpoint produced by the unit-test helper. Verify the base hash before and after, all metrics are finite, `best.pt` and `latest.pt` load, and generated output terminates.

- [ ] **Step 4: Run cloud preflight without training**

Run: `make runpod-sft-300m-preflight SFT_CLOUD_BASE_CHECKPOINT=./best-300m-current.pt`

Expected: imports/audits/path wiring either complete from existing local artifacts or stop only at an explicit missing-public-import message before training. No base file is modified.

- [ ] **Step 5: Review diffs and data statistics against the design**

Confirm every design verification requirement has direct evidence: boundary test, global/near dedupe tests, role validation, seeded selection, group split, token-weighted loss, best/latest checkpoints, tokenizer pad, audit-clean corpora, smoke run, and cloud preflight.

- [ ] **Step 6: Run a dedicated code review pass and fix all substantive findings**

Use `superpowers:requesting-code-review`; rerun focused tests for each fix and `make test` once more after all fixes.

- [ ] **Step 7: Commit any verification fixes**

```bash
git add Makefile src/superagi/chat scripts tests data/sft docs/superpowers
git commit -m "Finish SFT production readiness"
```

Skip this commit when Step 6 produced no changes. Before staging, confirm `git status --short` contains only SFT-readiness files plus the pre-existing untracked `.DS_Store` files; never stage those `.DS_Store` files.

- [ ] **Step 8: Push the verified branch**

```bash
git push origin scale-300m
```
