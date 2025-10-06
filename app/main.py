\
import os
from typing import List, Dict, Any
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from app.db import get_conn
from app import rag
from utils.text import load_pdf_bytes, load_txt_bytes, paragraph_chunk

app = FastAPI(title="Cloud AI Control Plane")

class IngestTextBody(BaseModel):
    source: str
    text: str

@app.post("/ingest_text")
def ingest_text(body: IngestTextBody):
    chunks = paragraph_chunk(body.text)
    embs = rag.embed(chunks)
    _insert_chunks(body.source, chunks, embs)
    return {"ok": True, "chunks": len(chunks)}

@app.post("/ingest_file")
def ingest_file(file: UploadFile = File(...), source: str = ""):
    name = file.filename or "uploaded"
    ext = (name.split(".")[-1] or "").lower()
    if ext == "pdf":
        text = load_pdf_bytes(file.file)
    else:
        text = load_txt_bytes(file.file)
    src = source or name
    chunks = paragraph_chunk(text)
    embs = rag.embed(chunks)
    _insert_chunks(src, chunks, embs)
    return {"ok": True, "chunks": len(chunks), "source": src}

class AskBody(BaseModel):
    question: str
    k: int = 5
    probes: int = 10

@app.post("/ask")
def ask(body: AskBody):
    qemb = rag.embed_one(body.question)
    hits = _search(qemb, k=body.k, probes=body.probes)
    answer = rag.build_answer(body.question, hits)
    return {"answer": answer, "hits": hits}

# --- helpers ---
def _vec_literal(v: List[float]) -> str:
    return "[" + ",".join(f"{x:.8f}" for x in v) + "]"

def _insert_chunks(source: str, chunks: List[str], embs: List[List[float]]):
    with get_conn() as conn:
        with conn.cursor() as cur:
            for chunk, emb in zip(chunks, embs):
                cur.execute(
                    "insert into documents (source, chunk, embedding) values (%s, %s, %s::vector)",
                    (source, chunk, _vec_literal(emb))
                )
        conn.commit()

def _search(query_emb: List[float], k: int = 5, probes: int = 10) -> List[Dict[str,Any]]:
    q = _vec_literal(query_emb)
    with get_conn() as conn:
        with conn.cursor() as cur:
            try:
                cur.execute("set ivfflat.probes = %s;", (probes,))
            except Exception:
                pass
            cur.execute(
                """
                select id, source, chunk, (1 - (embedding <=> %s::vector)) as score
                from documents
                order by embedding <=> %s::vector
                limit %s;
                """,
                (q, q, k)
            )
            rows = cur.fetchall()
    return [{"id": r[0], "source": r[1], "chunk": r[2], "score": float(r[3])} for r in rows]
