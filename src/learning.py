from __future__ import annotations

import json
import re
from pydantic import ValidationError
from .config import settings
from .llm import invoke_llm
from .rag import fetch_all_chunks, format_citations, render_prompt, retrieve
from .schemas import Flashcard, FlashcardSet, QuizItem, QuizSet, Summary

SUMMARY_SINGLE_TEMPLATE = "summary_single.jinja2"
SUMMARY_MAP_TEMPLATE = "summary_map.jinja2"
SUMMARY_REDUCE_TEMPLATE = "summary_reduce.jinja2"
QUIZ_TEMPLATE = "quiz.jinja2"
FLASHCARDS_TEMPLATE = "flashcards.jinja2"


def _resolve_target(document, query, filters, k, retrieval_k):
    effective_filters = dict(filters or {})
    if document:
        effective_filters["filename"] = document
    if query:
        return retrieve(query, k=k or retrieval_k, filters=effective_filters), "query", query
    if effective_filters:
        chunks = fetch_all_chunks(filters=effective_filters)
        scope = "document" if document else "filter"
        target = ", ".join(f"{key}={value}" for key, value in effective_filters.items())
        return chunks, scope, target
    return fetch_all_chunks(filters=None), "corpus", None


def _parse_json(text: str):
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.I)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    obj = json.loads(cleaned)
    if not isinstance(obj, (dict, list)):
        raise RuntimeError("Expected JSON object or array")
    return obj


def _validate_summary_payload(payload: dict) -> tuple[str, list[str]]:
    summary = str(payload.get("summary") or "").strip()
    key_points = [str(x).strip() for x in (payload.get("key_points") or []) if str(x).strip()]
    if not summary:
        raise RuntimeError("Summary is empty")
    return summary, key_points


def _validate_items(payload, key, model_class, dedup_field, label, valid_markers):
    raw_items = payload.get(key) if isinstance(payload, dict) else None
    if not isinstance(raw_items, list):
        raise RuntimeError(f"Missing list field '{key}'")
    items, seen = [], set()
    for raw in raw_items:
        try:
            item = model_class.model_validate(raw)
        except ValidationError:
            continue
        norm = str(getattr(item, dedup_field, "")).strip().lower()
        if not norm or norm in seen:
            continue
        seen.add(norm)
        markers = [m for m in item.source_markers if m in valid_markers]
        items.append(item.model_copy(update={"source_markers": markers}))
    if not items:
        raise RuntimeError(f"No valid {label} produced")
    return items


def summarize(document=None, query=None, filters=None, k=None) -> Summary:
    chunks, scope, target = _resolve_target(document, query, filters, k, settings.summarize_retrieval_k)
    if not chunks:
        raise RuntimeError("No indexed content matched the requested scope")
    if len(chunks) <= settings.summarize_batch_size:
        payload = _parse_json(invoke_llm(render_prompt(SUMMARY_SINGLE_TEMPLATE, chunks=chunks)))
        summary_text, key_points = _validate_summary_payload(payload)
    else:
        partials = []
        for start in range(0, len(chunks), settings.summarize_batch_size):
            batch = chunks[start:start + settings.summarize_batch_size]
            payload = _parse_json(invoke_llm(render_prompt(SUMMARY_MAP_TEMPLATE, chunks=batch)))
            s, p = _validate_summary_payload(payload)
            partials.append({"summary": s, "key_points": p})
        payload = _parse_json(invoke_llm(render_prompt(SUMMARY_REDUCE_TEMPLATE, partials=partials)))
        summary_text, key_points = _validate_summary_payload(payload)
    return Summary(scope=scope, target=target, summary=summary_text, key_points=key_points, citations=format_citations(chunks), chunks=chunks)


def generate_quiz(document=None, query=None, filters=None, count=None, k=None) -> QuizSet:
    chunks, scope, target = _resolve_target(document, query, filters, k, settings.generation_retrieval_k)
    if not chunks:
        raise RuntimeError("No indexed content matched the requested scope")
    n = count or settings.quiz_default_count
    markers = {f"S{i}" for i in range(1, len(chunks) + 1)}
    payload = _parse_json(invoke_llm(render_prompt(QUIZ_TEMPLATE, chunks=chunks, count=n)))
    items = _validate_items(payload, "items", QuizItem, "question", "quiz items", markers)
    return QuizSet(scope=scope, target=target, items=items[:n], chunks=chunks, citations=format_citations(chunks))


def generate_flashcards(document=None, query=None, filters=None, count=None, k=None) -> FlashcardSet:
    chunks, scope, target = _resolve_target(document, query, filters, k, settings.generation_retrieval_k)
    if not chunks:
        raise RuntimeError("No indexed content matched the requested scope")
    n = count or settings.flashcards_default_count
    markers = {f"S{i}" for i in range(1, len(chunks) + 1)}
    payload = _parse_json(invoke_llm(render_prompt(FLASHCARDS_TEMPLATE, chunks=chunks, count=n)))
    cards = _validate_items(payload, "cards", Flashcard, "front", "flashcards", markers)
    return FlashcardSet(scope=scope, target=target, cards=cards[:n], chunks=chunks, citations=format_citations(chunks))
