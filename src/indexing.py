from __future__ import annotations

import hashlib
import uuid
from collections import defaultdict
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import settings
from .schemas import ChunkMetadata, DocumentInfo, UploadResponse
from .store import ensure_collection, get_client, get_vector_store, scroll_all


def discover_pdfs() -> list[Path]:
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    return sorted(p for p in settings.data_dir.glob("*.pdf") if p.is_file())


def _document_id(path: Path) -> str:
    raw = f"{path.name}:{path.stat().st_size}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def _chunk_id(doc_id: str, page: int, index: int) -> str:
    return f"{doc_id}:{page}:{index}"


def _load_pdf(path: Path):
    pages = PyPDFLoader(str(path)).load()
    doc_id = _document_id(path)
    for doc in pages:
        original = dict(doc.metadata)
        page_number = int(original.get("page", 0)) + 1
        doc.metadata = {"document_id": doc_id, "filename": path.name, "source": str(path.resolve()), "page": page_number, "section": original.get("section")}
    return pages


def _splitter(chunk_size: int | None = None, chunk_overlap: int | None = None):
    return RecursiveCharacterTextSplitter(chunk_size=chunk_size or settings.chunk_size, chunk_overlap=settings.chunk_overlap if chunk_overlap is None else chunk_overlap, separators=["\n\n", "\n", ". ", " ", ""], keep_separator=False)


def build_chunks(pdf_paths: list[Path], chunk_size: int | None = None, chunk_overlap: int | None = None, chunker=None):
    page_docs = []
    for path in pdf_paths:
        page_docs.extend(_load_pdf(path))
    splitter = chunker or _splitter(chunk_size, chunk_overlap)
    chunks = splitter.split_documents(page_docs)
    per_doc_page_counter: dict[tuple[str, int], int] = defaultdict(int)
    for chunk in chunks:
        doc_id = chunk.metadata["document_id"]
        page = int(chunk.metadata["page"])
        key = (doc_id, page)
        idx = per_doc_page_counter[key]
        per_doc_page_counter[key] += 1
        meta = ChunkMetadata(document_id=doc_id, filename=chunk.metadata["filename"], source=chunk.metadata["source"], page=page, chunk_id=_chunk_id(doc_id, page, idx), section=chunk.metadata.get("section"))
        chunk.metadata = meta.model_dump()
    return chunks


def index_chunks(chunks, collection_name: str | None = None) -> int:
    if not chunks:
        return 0
    ids = [str(uuid.uuid5(uuid.NAMESPACE_DNS, c.metadata["chunk_id"])) for c in chunks]
    get_vector_store(collection_name).add_documents(chunks, ids=ids)
    return len(chunks)


def ingest(recreate: bool = False, collection_name: str | None = None, chunker=None, chunk_size: int | None = None, chunk_overlap: int | None = None) -> int:
    pdfs = discover_pdfs()
    ensure_collection(recreate=recreate, collection_name=collection_name)
    chunks = build_chunks(pdfs, chunker=chunker, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return index_chunks(chunks, collection_name=collection_name)


def save_and_ingest_pdf(file_bytes: bytes, filename: str) -> UploadResponse:
    safe_name = Path(filename).name
    if not safe_name.lower().endswith(".pdf"):
        raise ValueError("Only PDF files are supported")
    if not file_bytes:
        raise ValueError("Uploaded PDF is empty")
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    dest = settings.data_dir / safe_name
    dest.write_bytes(file_bytes)
    ensure_collection(recreate=False)
    chunks = build_chunks([dest])
    return UploadResponse(filename=safe_name, chunks_indexed=index_chunks(chunks))


def list_documents(collection_name: str | None = None) -> list[DocumentInfo]:
    name = collection_name or settings.qdrant_collection
    client = get_client()
    if not client.collection_exists(name):
        return []
    aggregate: dict[str, dict] = {}
    for page in scroll_all(name):
        for point in page:
            payload = point.payload or {}
            meta = payload.get("metadata") or {}
            doc_id = meta.get("document_id")
            filename = meta.get("filename")
            page_num = meta.get("page")
            if not doc_id or not filename:
                continue
            item = aggregate.setdefault(doc_id, {"document_id": doc_id, "filename": filename, "pages": set(), "chunks": 0})
            item["chunks"] += 1
            if isinstance(page_num, int):
                item["pages"].add(page_num)
    return sorted([DocumentInfo(document_id=v["document_id"], filename=v["filename"], pages=len(v["pages"]), chunks=v["chunks"]) for v in aggregate.values()], key=lambda x: x.filename.lower())
