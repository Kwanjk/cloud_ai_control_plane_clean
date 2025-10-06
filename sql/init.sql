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
