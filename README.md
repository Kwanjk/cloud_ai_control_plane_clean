# Cloud AI Control Plane (FastAPI + Supabase pgvector)

Cloud-first backbone for your personal AI:
- Ingest notes/docs as text or PDFs
- Store embeddings in **Supabase Postgres (pgvector)**
- Ask questions and get **RAG answers with citations**
- Ready to deploy on **Render**

## 0) Prereqs
- OpenAI account + API key
- Supabase project
- GitHub account (for Render deploy)

## 1) Supabase: enable pgvector & create table
Run this in the SQL editor:
```sql
create extension if not exists vector;
create extension if not exists pgcrypto;

create table if not exists documents (
  id uuid primary key default gen_random_uuid(),
  source text,
  chunk text not null,
  embedding vector(1536),
  created_at timestamptz default now()
);

create index if not exists documents_embedding_ivfflat
  on documents using ivfflat (embedding vector_cosine_ops) with (lists = 100);
```
> If you choose a different embedding model/dimension, change both the `vector(…)` size and `.env EMBED_DIM`.

## 2) Local test
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env && nano .env
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

## 3) Deploy to Render
- Push to GitHub
- New → Web Service
- Build: `pip install -r requirements.txt`
- Start: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
- Add env vars from `.env`

## API

### POST /ingest_text
```json
{ "source": "journal/2025-10-05.md", "text": "I ran 3 miles..." }
```
→ `{ "ok": true, "chunks": 4 }`

### POST /ingest_file  (multipart)
- key: `file` (.pdf/.txt)
- optional: `source`
→ `{ "ok": true, "chunks": 12, "source": "..." }`

### POST /ask
```json
{ "question": "What themes recur across my journals?", "k": 5 }
```
→ `{ "answer": "... [1][2] ...", "hits": [ ... ] }`
