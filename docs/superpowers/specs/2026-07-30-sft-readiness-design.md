# SuperAGI SFT Readiness Design

## Objective

Prepare the completed 300M pretraining checkpoint for a reliable, reproducible supervised fine-tuning run on the existing RunPod RTX 4090. The result must be a functional neutral chat model plus two independently trained personality variants. The pipeline must preserve the pretrained checkpoint, reject low-quality training data before GPU work begins, select checkpoints using held-out data, and require one operator command after the repository is updated.

## Observed Failure Modes

The design addresses failures observed in prior SFT experiments:

- answers ignored the question or continued an obsolete topic;
- the model adopted a single training phrase and repeated it across unrelated prompts;
- synthetic metadata and numbered prompt templates appeared in output;
- identity, caution, and generic advice examples dominated ordinary answers;
- later SFT steps could be worse than earlier checkpoints, but only the last checkpoint was retained;
- train and validation examples were randomly separated without grouping near-duplicates;
- public-data deduplication restarted for every source;
- chat inference ended the prompt with `<agi>`, while SFT examples placed a space after `<agi>` before the response.

The tracked `broad-mixed.jsonl` and `style-playful-direct.jsonl` are not production-quality inputs. They contain repeated canonical answers, templated rewrites, and synthetic numbered prompts. They will be replaced rather than patched incrementally.

## Training Architecture

The pipeline has one behavior phase and two optional style branches:

```text
pretrained best.pt
       |
       v
core SFT best.pt
       |
       +-------------------------+
       |                         |
       v                         v
playful/direct best.pt      calm/precise best.pt
```

### Core SFT

Core SFT teaches:

- answer the latest question directly;
- preserve useful context across short multi-turn conversations;
- recover cleanly after correction or topic changes;
- distinguish explanation from personal advice;
- admit missing live information without turning every answer into a disclaimer;
- avoid invented identity, experience, tools, browsing, location, salary, credentials, or private access;
- use `<eos>` to end a complete answer;
- produce concise, concrete prose rather than generic process language.

Core SFT mixes curated examples with filtered public instruction and chat data in one phase. Curated examples target 15-25% of actual sampling mass. Identity and limitation examples are capped so they cannot become the default response mode.

### Style Branches

Each style branch starts from the same core-best checkpoint and uses a low learning rate for a short run. Style datasets include core-replay examples so tone does not replace task behavior.

- **Playful/direct:** concise, lightly teasing, willing to challenge weak premises, never hostile or demeaning.
- **Calm/precise:** concise, literal, measured, and low-flourish.

The neutral core checkpoint remains a first-class output and is never overwritten.

## Chat Contract

Training and inference use exactly the same serialized boundary:

```text
<bos><user> question
<agi> answer<eos>
```

Generation prompts must end in `<agi> `, including the same trailing space used during SFT. Only AGI response content and its `<eos>` token contribute to loss. User, system, and role-prefix tokens remain masked. Inference stops on `<eos>` or another role token.

The pad token ID is resolved from the checkpoint tokenizer rather than assumed to be zero.

## Public Data Policy

The production importer supports the existing sources but applies source-specific trust:

- `no_robots`: primary high-quality instruction source;
- `openassistant`: useful human multi-turn data after strict quality filtering;
- `dolly`: lower-weight single-turn instruction data;
- `ultrachat`: lower-weight synthetic breadth;
- `wildchat`: excluded from the default production mixture; it remains opt-in for experiments.

Selection is deterministic but not "first N accepted." Candidates are shuffled or reservoir-sampled with a seed so dataset ordering does not define the training distribution.

Filtering occurs globally across all sources and rejects:

- malformed or non-alternating role sequences;
- conversations not ending in an AGI response;
- empty, overlong, or context-overflow examples;
- exact duplicate conversations, prompts, or AGI answers;
- normalized near-duplicate prompt/answer pairs above a configurable similarity threshold;
- repeated n-gram loops and low lexical-diversity answers;
- replacement characters, leaked synthetic tags, dataset metadata, and known generation artifacts;
- assistant identity claims inconsistent with SuperAGI;
- claims of live browsing, private access, credentials, employment, location, personal history, or professional status;
- generic refusal/disclaimer answers when the prompt is harmless;
- answers that merely restate the prompt without adding information.

The importer writes a machine-readable audit containing accepted counts, rejection reasons, token counts, duplicate statistics, and selected counts per source.

## Curated Data Policy

The curated core corpus is rewritten as intentionally authored examples, not mechanical permutations. Target size is 1,200-2,000 conversations with approximately equal single-turn and multi-turn coverage. Independent topic batches are reviewed before being merged.

Coverage includes:

- identity and capability boundaries, kept to at most 3% of conversations;
- everyday food, travel, household, study, work, relationships, and planning questions;
- clear explanations of AI, computing, math, science, economics, finance, politics, and history;
- short transformations such as summarizing, rewriting, comparing, and outlining;
- follow-ups, pronoun/context resolution, correction, contradiction, topic switching, and requests for simpler explanations;
- uncertainty, current-information limits, and high-stakes boundaries;
- adversarial prompts that try to induce false identity or fabricated capabilities;
- answer-length variety from one sentence to short structured responses.

Every curated response must answer the actual prompt in its first sentence. Generic phrases such as "use that to choose the next step," "start smaller than feels necessary," and repeated framing labels are prohibited. Synthetic sequence numbers, placeholder metadata, and topic-name substitutions are prohibited.

Style corpora target 400-800 reviewed conversations per personality. They express tone through natural wording, not repeated prefixes such as "Blunt version:" or "Useful answer:".

## Dataset Audit Gates

`make sft-audit` runs before training and fails on any hard violation:

- invalid JSON or role sequence;
- empty response or missing supervised response tokens;
- exact duplicate conversation or AGI answer in curated data;
- train/validation canonical overlap;
- special-token/control-character leakage;
- known synthetic metadata patterns;
- configured response/context limit violations;
- disallowed template repetition above threshold;
- missing required behavioral coverage;
- source mixture outside configured bounds.

The audit also reports non-fatal corpus statistics: source distribution, turn distribution, token quantiles, response-length quantiles, most frequent openings, repeated n-grams, and identity/disclaimer frequency.

## Validation and Checkpointing

Splitting happens before tokenization and is deterministic, source-stratified, and group-aware. Canonically similar prompts and answers stay in the same partition. A small tracked behavioral evaluation set is never included in training.

SFT training retains:

- `latest.pt` for recovery;
- `best.pt` selected by held-out token-weighted loss;
- periodic metrics with train loss, validation loss, learning rate, elapsed time, examples per second, and supervised tokens per second.

The final named phase checkpoint is copied from `best.pt`, not the last update. Core and style phases use separate run directories. Existing checkpoints are never deleted by the SFT target.

## Cloud Training Behavior

The RunPod command performs:

1. dependency setup;
2. base-checkpoint and tokenizer preflight;
3. public-data import with deterministic source limits;
4. full corpus audit and mixture report;
5. core SFT training;
6. fixed-prompt generation from the core-best checkpoint;
7. both optional style branches;
8. fixed-prompt generation for each branch;
9. a final artifact manifest with paths, hashes, metrics, and source metadata.

The production target supports CUDA mixed precision, gradient accumulation, fused AdamW when available, and configurable activation checkpointing. Defaults are chosen for a single RTX 4090 and can be overridden from the command line.

The command aborts before GPU training if import or audit fails. It prints resumable commands and output paths. It never invokes `clean-generated` and never overwrites the pretrained base checkpoint.

## Behavioral Evaluation

Automated generation checks use a tracked prompt suite covering identity, direct factual explanation, everyday advice, topic change, correction, uncertainty, multi-turn reference, unsafe certainty, and answer termination. Hard checks reject:

- empty output;
- role-token leakage;
- missing termination within the token budget;
- excessive repeated n-grams;
- false first-person identity/credential claims;
- failure to change topic after an explicit reset;
- identical answers to unrelated prompts.

These checks are regression gates, not a claim of factual correctness. Final model quality still requires human review of fixed prompts from core and both style variants.

## Verification Requirements

Implementation is complete only when:

- unit tests reproduce and then fix the chat-boundary mismatch;
- importer tests cover global and near-duplicate filtering, role validation, seeded selection, and artifact rejection;
- training tests cover group-aware source-stratified splits, token-weighted validation, best/latest checkpoint behavior, and tokenizer-derived padding;
- curated corpora pass the new audit with no waived hard failures;
- the full test suite passes;
- a bounded CPU/MPS smoke run completes without changing the base checkpoint;
- the RunPod target has a dry-run/preflight test proving command wiring and artifact separation.

