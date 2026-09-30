from __future__ import annotations

from dataclasses import dataclass
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from ..store import get_embeddings

DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]
RECURSIVE_CONFIGS = [("rc_500_50",500,50),("rc_800_100",800,100),("rc_1000_150",1000,150),("rc_1500_200",1500,200)]
SEMANTIC_CONFIGS = [("semantic_percentile","percentile"),("semantic_std_dev","standard_deviation"),("semantic_interquartile","interquartile")]

@dataclass(frozen=True)
class ChunkingStrategy:
    strategy_id: str
    chunker: object
    params: dict[str, object]

@dataclass(frozen=True)
class RecursiveChunker:
    chunk_size: int = 500
    chunk_overlap: int = 50
    separators: list[str] | None = None
    def _splitter(self):
        return RecursiveCharacterTextSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap, separators=self.separators or DEFAULT_SEPARATORS, is_separator_regex=False)
    def split_documents(self, documents: list[Document]) -> list[Document]:
        return self._splitter().split_documents(documents) if documents else []

@dataclass(frozen=True)
class SemanticChunkerWrapper:
    breakpoint_type: str = "percentile"
    def _splitter(self):
        from langchain_experimental.text_splitter import SemanticChunker
        return SemanticChunker(embeddings=get_embeddings(), breakpoint_threshold_type=self.breakpoint_type)
    def split_documents(self, documents: list[Document]) -> list[Document]:
        return self._splitter().split_documents(documents) if documents else []

def get_strategies(include_semantic: bool = True) -> list[ChunkingStrategy]:
    result = [ChunkingStrategy(sid, RecursiveChunker(size, overlap), {"chunk_size": size, "chunk_overlap": overlap}) for sid,size,overlap in RECURSIVE_CONFIGS]
    if include_semantic:
        result.extend(ChunkingStrategy(sid, SemanticChunkerWrapper(kind), {"breakpoint_type": kind}) for sid,kind in SEMANTIC_CONFIGS)
    return result
