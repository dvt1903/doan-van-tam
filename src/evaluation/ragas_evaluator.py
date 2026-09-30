from __future__ import annotations

import csv
from collections.abc import Callable
from pathlib import Path
from datasets import Dataset
from ragas import evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness
from ragas.run_config import RunConfig
from ..llm import get_llm
from ..rag import answer
from ..schemas import RagAnswer
from ..store import get_embeddings


def load_benchmark(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))
    return [r for r in rows if r.get("question") and r.get("ground_truth")]


def get_ragas_metrics(llm, embeddings):
    faithfulness.llm = llm
    answer_relevancy.llm = llm
    context_precision.llm = llm
    context_precision.embeddings = embeddings
    context_recall.llm = llm
    context_recall.embeddings = embeddings
    return [faithfulness, answer_relevancy, context_precision, context_recall]


def run_evaluation(test_cases: list[dict[str, str]], *, answer_fn: Callable[[str], RagAnswer] = answer, llm_provider: str | None = None, timeout_s: int = 180, max_retries: int = 3, max_workers: int = 4):
    if llm_provider == "mock":
        raise ValueError("Ragas requires a real judge LLM; use gemini, vllm or hf_local")
    data = {"user_input": [], "response": [], "retrieved_contexts": [], "reference": []}
    for case in test_cases:
        response = answer_fn(case["question"])
        data["user_input"].append(case["question"])
        data["response"].append(response.answer)
        data["retrieved_contexts"].append([c.text for c in response.chunks])
        data["reference"].append(case["ground_truth"])
    dataset = Dataset.from_dict(data)
    llm = LangchainLLMWrapper(get_llm(provider=llm_provider))
    embeddings = LangchainEmbeddingsWrapper(get_embeddings())
    config = RunConfig(timeout=timeout_s, max_retries=max_retries, max_workers=max_workers)
    return evaluate(dataset=dataset, metrics=get_ragas_metrics(llm, embeddings), llm=llm, embeddings=embeddings, run_config=config)


def summary_metrics(df):
    cols = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    return {c: float(df[c].mean()) for c in cols if c in df.columns}
