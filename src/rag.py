from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from jinja2 import Environment, FileSystemLoader, StrictUndefined
from .config import settings
from .filters import filters_to_qdrant
from .llm import invoke_llm
from .schemas import ChunkMetadata, Citation, RagAnswer, RetrievedChunk
from .store import get_client, get_vector_store, scroll_all

PROMPTS_DIR = Path(__file__).parent / "prompts"
ANSWER_TEMPLATE = "answer.jinja2"

@lru_cache(maxsize=1)
def _jinja_env() -> Environment:
    return Environment(loader=FileSystemLoader(str(PROMPTS_DIR)), autoescape=False, undefined=StrictUndefined, trim_blocks=True, lstrip_blocks=True)

def render_prompt(template_name: str, **context) -> str:
    return _jinja_env().get_template(template_name).render(**context)

def retrieve(query: str, k: int | None = None, filters=None, collection_name: str | None = None) -> list[RetrievedChunk]:
    if not query or not query.strip():
        raise ValueError("query must not be empty")
    name = collection_name or settings.qdrant_collection
    if not get_client().collection_exists(name):
        return []
    hits = get_vector_store(collection_name).similarity_search_with_score(query=query, k=k or settings.top_k, filter=filters_to_qdrant(filters))
    return [RetrievedChunk(text=doc.page_content, score=float(score), metadata=ChunkMetadata(**doc.metadata)) for doc, score in hits]

def fetch_all_chunks(filters=None, collection_name: str | None = None) -> list[RetrievedChunk]:
    name = collection_name or settings.qdrant_collection
    if not get_client().collection_exists(name):
        return []
    results = []
    for page in scroll_all(name, scroll_filter=filters_to_qdrant(filters)):
        for point in page:
            payload = point.payload or {}
            meta = payload.get("metadata") or {}
            text = payload.get("page_content") or payload.get("text") or ""
            if meta and text:
                results.append(RetrievedChunk(text=text, score=0.0, metadata=ChunkMetadata(**meta)))
    def key(c):
        try: idx = int(c.metadata.chunk_id.rsplit(":", 1)[-1])
        except Exception: idx = 0
        return c.metadata.filename, c.metadata.page, idx
    return sorted(results, key=key)

def format_citations(chunks: list[RetrievedChunk]) -> list[Citation]:
    return [Citation(source_index=i, source_marker=f"S{i}", filename=c.metadata.filename, page=c.metadata.page, section=c.metadata.section, chunk_id=c.metadata.chunk_id) for i, c in enumerate(chunks, 1)]

def answer(question: str, k: int | None = None, filters=None, collection_name: str | None = None) -> RagAnswer:
    chunks = retrieve(question, k=k, filters=filters, collection_name=collection_name)
    if not chunks:
        return RagAnswer(question=question, answer="Tôi không có đủ thông tin trong ngữ cảnh được cung cấp để trả lời.")
    text = invoke_llm(render_prompt(ANSWER_TEMPLATE, question=question, chunks=chunks))
    return RagAnswer(question=question, answer=text.strip(), citations=format_citations(chunks), chunks=chunks)
