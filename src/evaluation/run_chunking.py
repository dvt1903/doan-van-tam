from __future__ import annotations

import argparse
import json
from pathlib import Path
from ..config import settings
from ..indexing import ingest
from ..rag import answer
from ..schemas import RagAnswer
from .chunking_strategies import get_strategies
from .ragas_evaluator import load_benchmark, run_evaluation, summary_metrics


def _write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _evaluate_strategy(strategy, output_dir: Path, test_cases, judge_provider: str):
    collection_name = f"{settings.qdrant_collection}__{strategy.strategy_id}"
    chunk_count = ingest(recreate=True, collection_name=collection_name, chunker=strategy.chunker)
    out = {"strategy_id": strategy.strategy_id, "params": strategy.params, "chunk_count": chunk_count, "summary_metrics": {}}
    try:
        def answer_fn(q: str) -> RagAnswer: return answer(q, collection_name=collection_name)
        result = run_evaluation(test_cases, answer_fn=answer_fn, llm_provider=judge_provider)
        out["summary_metrics"] = summary_metrics(result.to_pandas())
    except Exception as exc:
        out["error"] = str(exc)
    _write_json(output_dir / f"{strategy.strategy_id}.json", out)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", default="src/evaluation/benchmark_rag.csv")
    parser.add_argument("--output-dir", default="evaluation_results/chunking")
    parser.add_argument("--judge-provider", default="gemini")
    parser.add_argument("--no-semantic", action="store_true")
    args = parser.parse_args()
    cases = load_benchmark(Path(args.benchmark))
    results = [_evaluate_strategy(s, Path(args.output_dir), cases, args.judge_provider) for s in get_strategies(not args.no_semantic)]
    _write_json(Path(args.output_dir) / "summary.json", results)

if __name__ == "__main__": main()
