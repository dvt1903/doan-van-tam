from __future__ import annotations

import json
import re
from functools import lru_cache

from .config import settings


def _build_hf_local():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
    from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline

    tokenizer = AutoTokenizer.from_pretrained(settings.hf_model)
    kwargs = {}
    if torch.cuda.is_available() and settings.hf_device >= 0:
        kwargs["torch_dtype"] = torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(settings.hf_model, **kwargs)
    text_gen = pipeline(
        task="text-generation",
        model=model,
        tokenizer=tokenizer,
        device=settings.hf_device,
        return_full_text=False,
        max_new_tokens=settings.hf_max_new_tokens,
        do_sample=settings.llm_temperature > 0,
        temperature=max(settings.llm_temperature, 1e-5),
    )
    return ChatHuggingFace(llm=HuggingFacePipeline(pipeline=text_gen))


def _build_gemini():
    from langchain_google_genai import ChatGoogleGenerativeAI

    return ChatGoogleGenerativeAI(
        model=settings.gemini_model,
        temperature=settings.llm_temperature,
        google_api_key=settings.google_api_key,
    )


def _build_vllm():
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=settings.vllm_model,
        api_key=settings.vllm_api_key,
        base_url=settings.vllm_api_base,
        temperature=settings.llm_temperature,
    )


def _extract_sources(prompt: str) -> list[tuple[str, str]]:
    pattern = re.compile(r"\[(S\d+)\][^\n]*\n(.*?)(?=\n\[S\d+\]|\Z)", re.S)
    return [(m.group(1), re.sub(r"\s+", " ", m.group(2)).strip()) for m in pattern.finditer(prompt)]


def _mock_invoke(prompt: str) -> str:
    """Deterministic fallback so the whole project can be demoed without API keys."""
    sources = _extract_sources(prompt)
    source_texts = [text for _, text in sources if text]
    first = source_texts[0] if source_texts else "Không có ngữ cảnh phù hợp."

    if "TASK=ANSWER" in prompt:
        sentence = re.split(r"(?<=[.!?])\s+", first)[0][:900]
        return f"{sentence} [S1]" if sources else sentence

    if "TASK=SUMMARY" in prompt or "TASK=SUMMARY_MAP" in prompt:
        text = " ".join(source_texts)[:3000]
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
        summary = " ".join(sentences[:3]) or first[:700]
        points = sentences[:5] or [first[:300]]
        return json.dumps({"summary": summary, "key_points": points}, ensure_ascii=False)

    if "TASK=SUMMARY_REDUCE" in prompt:
        partials = re.findall(r"SUMMARY:\s*(.*)", prompt)
        summary = " ".join(partials)[:1800] or "Tóm tắt tổng hợp từ các phần đã xử lý."
        return json.dumps(
            {"summary": summary, "key_points": partials[:5] or [summary]},
            ensure_ascii=False,
        )

    count_match = re.search(r"COUNT=(\d+)", prompt)
    count = max(1, min(int(count_match.group(1)) if count_match else 5, 20))
    marker = sources[0][0] if sources else "S1"

    if "TASK=QUIZ" in prompt:
        base = re.split(r"(?<=[.!?])\s+", first)[0][:180]
        items = []
        for i in range(count):
            items.append(
                {
                    "question": f"Câu {i+1}: Ý nào phù hợp nhất với nội dung nguồn?",
                    "options": [base or "Nội dung đúng", "Phương án nhiễu A", "Phương án nhiễu B", "Phương án nhiễu C"],
                    "correct_index": 0,
                    "explanation": "Đáp án dựa trực tiếp trên đoạn nguồn được truy xuất.",
                    "source_markers": [marker],
                    "difficulty": "medium",
                    "topic": "Tài liệu",
                }
            )
        return json.dumps({"items": items}, ensure_ascii=False)

    if "TASK=FLASHCARDS" in prompt:
        sentence = re.split(r"(?<=[.!?])\s+", first)[0][:300]
        cards = []
        for i in range(count):
            cards.append(
                {
                    "front": f"Thẻ {i+1}: Nội dung chính của phần này là gì?",
                    "back": sentence,
                    "hint": "Xem lại nguồn được trích dẫn.",
                    "topic": "Tài liệu",
                    "source_markers": [marker],
                }
            )
        return json.dumps({"cards": cards}, ensure_ascii=False)

    return first[:1000]


@lru_cache(maxsize=4)
def get_llm(provider: str | None = None):
    provider = provider or settings.llm_provider
    if provider == "hf_local":
        return _build_hf_local()
    if provider == "gemini":
        return _build_gemini()
    if provider == "vllm":
        return _build_vllm()
    if provider == "mock":
        return None
    raise ValueError(f"Unknown llm_provider '{provider}'")


def invoke_llm(prompt: str, provider: str | None = None) -> str:
    provider = provider or settings.llm_provider
    if provider == "mock":
        return _mock_invoke(prompt)
    from langchain_core.messages import HumanMessage

    response = get_llm(provider=provider).invoke([HumanMessage(content=prompt)])
    return response.content if isinstance(response.content, str) else str(response.content)
