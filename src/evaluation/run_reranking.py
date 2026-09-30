from __future__ import annotations

import argparse
import json
from pathlib import Path
from sentence_transformers import CrossEncoder
from ..config import settings
from ..indexing import ingest
from ..llm import invoke_llm
from ..rag import ANSWER_TEMPLATE, format_citations, render_prompt, retrieve
from ..schemas import RagAnswer
from .ragas_evaluator import load_benchmark, run_evaluation, summary_metrics

RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"


def answer_with_reranker(question: str, collection_name: str, reranker: CrossEncoder, initial_k: int = 15, rerank_k: int = 5, filters: dict[str, object] | None = None) -> RagAnswer:
    chunks = retrieve(question, k=initial_k, filters=filters, collection_name=collection_name)
    if not chunks:
        return RagAnswer(question=question, answer="Tôi không có đủ thông tin trong ngữ cảnh được cung cấp để trả lời.")
    scores = reranker.predict([[question, chunk.text] for chunk in chunks])
    for chunk, score in zip(chunks, scores):
        chunk.score = float(score)
    reranked = sorted(chunks, key=lambda c: c.score, reverse=True)[:rerank_k]
    text = invoke_llm(render_prompt(ANSWER_TEMPLATE, question=question, chunks=reranked))
    return RagAnswer(question=question, answer=text.strip(), citations=format_citations(reranked), chunks=reranked)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", default="src/evaluation/benchmark_rag.csv")
    parser.add_argument("--output", default="evaluation_results/reranking.json")
    parser.add_argument("--judge-provider", default="gemini")
    parser.add_argument("--initial-k", type=int, default=15)
    parser.add_argument("--rerank-k", type=int, default=5)
    args = parser.parse_args()
    collection = f"{settings.qdrant_collection}__reranking"
    chunk_count = ingest(recreate=True, collection_name=collection, chunk_size=1000, chunk_overlap=150)
    reranker = CrossEncoder(RERANKER_MODEL)
    cases = load_benchmark(Path(args.benchmark))
    def answer_fn(q: str): return answer_with_reranker(q, collection, reranker, args.initial_k, args.rerank_k)
    result = run_evaluation(cases, answer_fn=answer_fn, llm_provider=args.judge_provider)
    payload = {"chunk_count": chunk_count, "initial_k": args.initial_k, "rerank_k": args.rerank_k, "metrics": summary_metrics(result.to_pandas())}
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))

if __name__ == "__main__": main()
