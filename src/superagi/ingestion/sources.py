from __future__ import annotations

import html
import re
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from superagi.ingestion.streaming import (
    TokenShardBuildResult,
    build_token_shards_from_stream,
    normalize_stream_document_text,
)


@dataclass(frozen=True)
class SourceSpec:
    name: str
    dataset_name: str
    dataset_config: str | None = None
    split: str = "train"
    cleaner: str | None = None


DEFAULT_SOURCE_SPECS: dict[str, SourceSpec] = {
    "fineweb": SourceSpec(
        name="fineweb",
        dataset_name="HuggingFaceFW/fineweb",
        dataset_config="sample-10BT",
        cleaner="fineweb",
    ),
    "wikipedia": SourceSpec(
        name="wikipedia",
        dataset_name="wikimedia/wikipedia",
        dataset_config="20231101.en",
        cleaner="wikipedia",
    ),
    "dolma": SourceSpec(
        name="dolma",
        dataset_name="emozilla/dolma-v1_7-3B",
        cleaner="dolma",
    ),
    "openwebmath": SourceSpec(
        name="openwebmath",
        dataset_name="open-web-math/open-web-math",
        cleaner="openwebmath",
    ),
    "arxiv": SourceSpec(
        name="arxiv",
        dataset_name="ccdv/arxiv-summarization",
        dataset_config="document",
        cleaner="arxiv",
    ),
    "pmc": SourceSpec(
        name="pmc",
        dataset_name="casperhansen/pmc-oa-markdown",
        cleaner="pmc",
    ),
    "stackexchange": SourceSpec(
        name="stackexchange",
        dataset_name="HuggingFaceH4/stack-exchange-preferences",
        cleaner="stackexchange",
    ),
    "gutenberg": SourceSpec(
        name="gutenberg",
        dataset_name="manu/project_gutenberg",
        split="en",
        cleaner="gutenberg",
    ),
}


def parse_source_names(source_names: str) -> tuple[SourceSpec, ...]:
    names = [name.strip().lower() for name in source_names.split(",") if name.strip()]
    if not names:
        raise ValueError("sources must contain at least one source name")

    specs = []
    for name in names:
        spec = DEFAULT_SOURCE_SPECS.get(name)
        if spec is None:
            raise ValueError(f"unknown source: {name}")
        specs.append(spec)
    return tuple(specs)


def build_multi_source_token_shards(
    *,
    processed_dir: Path | str = Path("data/processed"),
    specs: Iterable[SourceSpec] | None = None,
    sources: str | None = None,
    max_documents_per_source: int = 1000,
    tokenizer_sample_documents: int = 500,
    shard_token_count: int = 1_000_000,
    validation_token_count: int = 100_000,
    target_train_tokens: int | None = None,
    min_chars: int = 500,
    bpe_vocab_size: int = 8000,
    bpe_min_frequency: int = 2,
    source_examples: Mapping[str, Iterable[Mapping[str, Any]]] | None = None,
) -> TokenShardBuildResult:
    resolved_specs = tuple(specs or parse_source_names(sources or "fineweb"))
    if not resolved_specs:
        raise ValueError("at least one source spec is required")

    def examples_factory() -> Iterable[dict[str, Any]]:
        return iter_mixed_source_examples(
            specs=resolved_specs,
            max_documents_per_source=max_documents_per_source,
            source_examples=source_examples,
        )

    return build_token_shards_from_stream(
        examples_factory=examples_factory,
        processed_dir=processed_dir,
        max_documents=max_documents_per_source * len(resolved_specs),
        tokenizer_sample_documents=tokenizer_sample_documents,
        shard_token_count=shard_token_count,
        validation_token_count=validation_token_count,
        target_train_tokens=target_train_tokens,
        min_chars=min_chars,
        bpe_vocab_size=bpe_vocab_size,
        bpe_min_frequency=bpe_min_frequency,
        metadata={
            "source": "mixed",
            "sources": [spec.name for spec in resolved_specs],
            "source_specs": [
                {
                    "name": spec.name,
                    "dataset": spec.dataset_name,
                    "dataset_config": spec.dataset_config,
                    "split": spec.split,
                    "cleaner": spec.cleaner or spec.name,
                }
                for spec in resolved_specs
            ],
            "max_documents_per_source": max_documents_per_source,
        },
    )


def iter_mixed_source_examples(
    *,
    specs: Iterable[SourceSpec],
    max_documents_per_source: int,
    source_examples: Mapping[str, Iterable[Mapping[str, Any]]] | None = None,
) -> Iterator[dict[str, Any]]:
    if max_documents_per_source <= 0:
        raise ValueError("max_documents_per_source must be positive")

    iterators = [
        _iter_clean_source_examples(
            spec,
            max_documents=max_documents_per_source,
            examples=None if source_examples is None else source_examples.get(spec.name),
        )
        for spec in specs
    ]
    active = [True for _ in iterators]
    while any(active):
        for index, iterator in enumerate(iterators):
            if not active[index]:
                continue
            try:
                yield next(iterator)
            except StopIteration:
                active[index] = False


def clean_source_example(source: str, example: Mapping[str, Any]) -> str:
    cleaner = source.lower()
    if cleaner in {"fineweb", "dolma", "generic"}:
        return _clean_generic_web_text(_first_text_field(example))
    if cleaner == "wikipedia":
        return _clean_wikipedia_text(_first_text_field(example))
    if cleaner == "openwebmath":
        return _clean_openwebmath_text(_first_text_field(example))
    if cleaner == "arxiv":
        return _clean_arxiv_example(example)
    if cleaner == "pmc":
        return _clean_pmc_text(_first_text_field(example))
    if cleaner == "stackexchange":
        return _clean_stackexchange_example(example)
    if cleaner == "gutenberg":
        return _clean_gutenberg_text(_first_text_field(example))
    return _clean_generic_web_text(_first_text_field(example))


def _iter_clean_source_examples(
    spec: SourceSpec,
    *,
    max_documents: int,
    examples: Iterable[Mapping[str, Any]] | None,
) -> Iterator[dict[str, Any]]:
    stream = examples if examples is not None else _load_hf_stream(spec)
    cleaner = spec.cleaner or spec.name
    yielded = 0
    for raw_example in stream:
        text = clean_source_example(cleaner, raw_example)
        if not text:
            continue
        yield {
            "text": text,
            "source": spec.name,
        }
        yielded += 1
        if yielded >= max_documents:
            break


def _load_hf_stream(spec: SourceSpec) -> Iterable[Mapping[str, Any]]:
    from datasets import load_dataset

    if spec.dataset_config:
        return load_dataset(
            spec.dataset_name,
            spec.dataset_config,
            split=spec.split,
            streaming=True,
        )
    return load_dataset(
        spec.dataset_name,
        split=spec.split,
        streaming=True,
    )


def _first_text_field(example: Mapping[str, Any]) -> str:
    for field_name in (
        "text",
        "content",
        "document",
        "article",
        "body",
        "page",
        "book",
        "completion",
    ):
        value = example.get(field_name)
        if isinstance(value, str) and value.strip():
            return value

    for value in example.values():
        if isinstance(value, str) and len(value.strip()) > 0:
            return value
    return ""


def _clean_generic_web_text(text: str) -> str:
    text = _strip_html(text)
    text = _drop_lines_matching(
        text,
        patterns=(
            r"^\s*(cookie|cookies|privacy policy|terms of use|subscribe|sign up)\b",
            r"^\s*(advertisement|sponsored|share this|follow us)\b",
            r"^\s*(javascript|enable javascript)\b",
        ),
    )
    return normalize_stream_document_text(text)


def _clean_wikipedia_text(text: str) -> str:
    text = re.sub(r"\[(?:\d+|citation needed|edit)\]", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^\s*=+\s*(.*?)\s*=+\s*$", r"\1", text, flags=re.MULTILINE)
    text = _truncate_at_heading(text, ("See also", "References", "External links"))
    text = _drop_lines_matching(
        text,
        patterns=(
            r"^\s*Category:",
            r"^\s*File:",
            r"^\s*Retrieved from\b",
        ),
    )
    return normalize_stream_document_text(text)


def _clean_openwebmath_text(text: str) -> str:
    text = _strip_html(text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return normalize_stream_document_text(text)


def _clean_arxiv_example(example: Mapping[str, Any]) -> str:
    abstract = _first_present_field(example, ("abstract", "summary"))
    article = _first_present_field(example, ("article", "document", "text"))
    conclusion = _extract_named_section(
        article,
        headings=("Conclusion", "Conclusions", "Concluding remarks", "Discussion"),
        stop_headings=("References", "Acknowledgements", "Appendix"),
    )
    parts = []
    if abstract:
        parts.append(f"Abstract\n{normalize_stream_document_text(abstract)}")
    if conclusion:
        parts.append(f"Conclusion\n{normalize_stream_document_text(conclusion)}")
    if not parts:
        return _clean_generic_web_text(article)
    return normalize_stream_document_text("\n\n".join(parts))


def _clean_pmc_text(text: str) -> str:
    text = _strip_html(text)
    text = _truncate_at_heading(text, ("References", "Bibliography", "Funding"))
    text = _drop_lines_matching(
        text,
        patterns=(
            r"^\s*(Figure|Fig\.|Table)\s+\d+[\.:]",
            r"^\s*Supplementary\b",
            r"^\s*Competing interests\b",
        ),
    )
    return normalize_stream_document_text(text)


def _clean_stackexchange_example(example: Mapping[str, Any]) -> str:
    title = _first_present_field(example, ("title", "question_title"))
    question = _first_present_field(example, ("question", "body", "prompt"))
    answer = _first_present_field(
        example,
        ("accepted_answer", "answer", "response", "completion", "chosen"),
    )

    parts = []
    if title:
        parts.append(f"Question: {_strip_html(title)}")
    if question:
        parts.append(_strip_html(question))
    if answer:
        parts.append(f"Answer: {_strip_html(answer)}")
    return normalize_stream_document_text("\n\n".join(part for part in parts if part))


def _clean_gutenberg_text(text: str) -> str:
    text = _strip_gutenberg_markers(text)
    text = _drop_lines_matching(
        text,
        patterns=(
            r"^\s*Project Gutenberg\b",
            r"^\s*Produced by\b",
            r"^\s*Transcribed from\b",
        ),
    )
    return normalize_stream_document_text(text)


def _first_present_field(example: Mapping[str, Any], fields: Iterable[str]) -> str:
    for field in fields:
        value = example.get(field)
        if isinstance(value, str) and value.strip():
            return value
    return ""


def _strip_html(text: str) -> str:
    text = re.sub(
        r"<(pre|code)\b[^>]*>.*?</\1>",
        " ",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"</p\s*>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    text = html.unescape(text)
    text = re.sub(r"[ \t]+", " ", text)
    return normalize_stream_document_text(text)


def _drop_lines_matching(text: str, *, patterns: Iterable[str]) -> str:
    compiled_patterns = [re.compile(pattern, flags=re.IGNORECASE) for pattern in patterns]
    kept_lines = []
    for line in text.splitlines():
        if any(pattern.search(line) for pattern in compiled_patterns):
            continue
        kept_lines.append(line)
    return "\n".join(kept_lines)


def _truncate_at_heading(text: str, headings: Iterable[str]) -> str:
    heading_pattern = "|".join(re.escape(heading) for heading in headings)
    match = re.search(
        rf"^\s*(?:=+\s*)?(?:{heading_pattern})(?:\s*=+)?\s*$",
        text,
        flags=re.IGNORECASE | re.MULTILINE,
    )
    if match is None:
        return text
    return text[: match.start()]


def _extract_named_section(
    text: str,
    *,
    headings: Iterable[str],
    stop_headings: Iterable[str],
) -> str:
    if not text:
        return ""
    heading_pattern = "|".join(re.escape(heading) for heading in headings)
    match = re.search(
        rf"^\s*(?:\d+(?:\.\d+)*\s+)?(?:{heading_pattern})\s*$",
        text,
        flags=re.IGNORECASE | re.MULTILINE,
    )
    if match is None:
        return ""

    section = text[match.end() :]
    stop_pattern = "|".join(re.escape(heading) for heading in stop_headings)
    stop = re.search(
        rf"^\s*(?:\d+(?:\.\d+)*\s+)?(?:{stop_pattern})\s*$",
        section,
        flags=re.IGNORECASE | re.MULTILINE,
    )
    if stop is not None:
        section = section[: stop.start()]
    return normalize_stream_document_text(section)


def _strip_gutenberg_markers(text: str) -> str:
    start = re.search(
        r"\*\*\*\s*START OF (?:THE )?PROJECT GUTENBERG EBOOK .*?\*\*\*",
        text,
        flags=re.IGNORECASE,
    )
    if start is not None:
        text = text[start.end() :]
    end = re.search(
        r"\*\*\*\s*END OF (?:THE )?PROJECT GUTENBERG EBOOK .*?\*\*\*",
        text,
        flags=re.IGNORECASE,
    )
    if end is not None:
        text = text[: end.start()]
    return text
