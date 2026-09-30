from __future__ import annotations

import json
from pathlib import Path
import typer
from ..export import export
from ..indexing import ingest as ingest_data_dir, list_documents
from ..learning import generate_flashcards, generate_quiz, summarize as summarize_learning
from ..rag import answer, retrieve

app = typer.Typer(help="Simple NotebookLM - RAG Learning CLI")

def _parse_filters(value: str | None):
    if not value: return None
    obj = json.loads(value)
    if not isinstance(obj, dict): raise typer.BadParameter("filters must be a JSON object")
    return obj

def _emit(model, output: str | None, fmt: str):
    path = Path(output) if output else None
    result = export(model, fmt=fmt, output=path)
    typer.echo(f"Saved: {path}" if path else result)

@app.command()
def ingest(recreate: bool = typer.Option(False, help="Recreate vector collection")):
    typer.echo(f"Done. {ingest_data_dir(recreate=recreate)} chunks indexed.")

@app.command()
def documents():
    typer.echo(json.dumps([d.model_dump() for d in list_documents()], ensure_ascii=False, indent=2))

@app.command()
def ask(question: str, k: int | None = None, filters: str | None = typer.Option(None, help='JSON, e.g. {"filename":"doc.pdf"}')):
    result = answer(question, k=k, filters=_parse_filters(filters))
    typer.echo(result.answer)
    typer.echo("\nNguồn:")
    for c in result.citations: typer.echo(f"- [{c.source_marker}] {c.filename}, trang {c.page}")

@app.command("debug-retrieval")
def debug_retrieval(question: str, k: int | None = None, filters: str | None = None):
    chunks = retrieve(question, k=k, filters=_parse_filters(filters))
    typer.echo(json.dumps([c.model_dump() for c in chunks], ensure_ascii=False, indent=2))

@app.command("summarize")
def summarize_cmd(document: str | None = None, query: str | None = None, filters: str | None = None, k: int | None = None, output: str | None = None, fmt: str = "text"):
    _emit(summarize_learning(document=document, query=query, filters=_parse_filters(filters), k=k), output, fmt)

@app.command()
def quiz(document: str | None = None, query: str | None = None, filters: str | None = None, count: int | None = None, k: int | None = None, output: str | None = None, fmt: str = "text"):
    _emit(generate_quiz(document=document, query=query, filters=_parse_filters(filters), count=count, k=k), output, fmt)

@app.command()
def flashcards(document: str | None = None, query: str | None = None, filters: str | None = None, count: int | None = None, k: int | None = None, output: str | None = None, fmt: str = "text"):
    _emit(generate_flashcards(document=document, query=query, filters=_parse_filters(filters), count=count, k=k), output, fmt)

if __name__ == "__main__": app()
