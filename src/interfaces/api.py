from __future__ import annotations

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from ..filters import MetadataFilter, filters_to_dict
from ..indexing import list_documents, save_and_ingest_pdf
from ..learning import generate_flashcards, generate_quiz, summarize as summarize_learning
from ..rag import answer, retrieve
from ..schemas import DocumentInfo, FlashcardSet, QuizSet, RagAnswer, Summary, UploadResponse

class AskRequest(BaseModel):
    question: str = Field(min_length=1)
    k: int | None = Field(default=None, ge=1, le=64)
    filters: MetadataFilter | None = None

class SummarizeRequest(BaseModel):
    document: str | None = None
    query: str | None = None
    filters: MetadataFilter | None = None
    k: int | None = Field(default=None, ge=1, le=64)

class QuizRequest(BaseModel):
    document: str | None = None
    query: str | None = None
    filters: MetadataFilter | None = None
    count: int | None = Field(default=None, ge=1, le=50)
    k: int | None = Field(default=None, ge=1, le=64)

class FlashcardsRequest(QuizRequest):
    pass

app = FastAPI(title="Simple NotebookLM - RAG Learning API", description="Grounded Q&A, summaries, quizzes and flashcards over indexed PDFs.", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.get("/health")
def health(): return {"status": "ok"}

@app.get("/documents", response_model=list[DocumentInfo])
def documents(): return list_documents()

@app.post("/upload", response_model=UploadResponse)
async def upload(file: UploadFile = File(...)):
    try:
        return save_and_ingest_pdf(await file.read(), file.filename or "")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

@app.post("/ask", response_model=RagAnswer)
def ask(req: AskRequest):
    try: return answer(req.question, k=req.k, filters=filters_to_dict(req.filters))
    except Exception as exc: raise HTTPException(status_code=500, detail=str(exc)) from exc

@app.post("/debug-retrieval")
def debug_retrieval(req: AskRequest):
    try: return [c.model_dump() for c in retrieve(req.question, k=req.k, filters=filters_to_dict(req.filters))]
    except Exception as exc: raise HTTPException(status_code=500, detail=str(exc)) from exc

@app.post("/summarize", response_model=Summary)
def summarize(req: SummarizeRequest):
    try: return summarize_learning(document=req.document, query=req.query, filters=filters_to_dict(req.filters), k=req.k)
    except Exception as exc: raise HTTPException(status_code=500, detail=str(exc)) from exc

@app.post("/quiz", response_model=QuizSet)
def quiz(req: QuizRequest):
    try: return generate_quiz(document=req.document, query=req.query, filters=filters_to_dict(req.filters), count=req.count, k=req.k)
    except Exception as exc: raise HTTPException(status_code=500, detail=str(exc)) from exc

@app.post("/flashcards", response_model=FlashcardSet)
def flashcards(req: FlashcardsRequest):
    try: return generate_flashcards(document=req.document, query=req.query, filters=filters_to_dict(req.filters), count=req.count, k=req.k)
    except Exception as exc: raise HTTPException(status_code=500, detail=str(exc)) from exc
