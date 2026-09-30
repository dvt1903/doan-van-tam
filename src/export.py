from __future__ import annotations

from pathlib import Path
from typing import Literal
from .schemas import FlashcardSet, QuizSet, Summary

ExportFormat = Literal["text", "md", "json"]


def _source_lines(model) -> list[str]:
    return [f"- [{c.source_marker}] {c.filename}, trang {c.page}" for c in model.citations]


def _to_markdown(model) -> str:
    if isinstance(model, Summary):
        lines = ["# Tóm tắt", "", model.summary, "", "## Ý chính"] + [f"- {point}" for point in model.key_points]
    elif isinstance(model, QuizSet):
        lines = ["# Quiz", ""]
        for i, item in enumerate(model.items, 1):
            lines.append(f"## Câu {i}. {item.question}")
            for j, option in enumerate(item.options):
                mark = "**" if j == item.correct_index else ""
                lines.append(f"- {chr(65+j)}. {mark}{option}{mark}")
            lines += [f"- Giải thích: {item.explanation}", f"- Nguồn: {', '.join(item.source_markers)}", ""]
    elif isinstance(model, FlashcardSet):
        lines = ["# Flashcards", ""]
        for i, card in enumerate(model.cards, 1):
            lines += [f"## Thẻ {i}", f"**Mặt trước:** {card.front}", f"**Mặt sau:** {card.back}"]
            if card.hint:
                lines.append(f"**Gợi ý:** {card.hint}")
            lines += [f"**Nguồn:** {', '.join(card.source_markers)}", ""]
    else:
        return model.model_dump_json(indent=2) + "\n"
    lines += ["", "## Nguồn", *_source_lines(model)]
    return "\n".join(lines).strip() + "\n"


def export(model, *, fmt: ExportFormat = "text", output: Path | None = None):
    if fmt == "json":
        text = model.model_dump_json(indent=2) + "\n"
    elif fmt in {"text", "md"}:
        text = _to_markdown(model)
    else:
        raise ValueError(f"Unknown fmt '{fmt}'. Expected 'text' | 'md' | 'json'.")
    if output is None:
        return text
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")
    return output
