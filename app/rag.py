import os
from typing import List, Dict, Any
from openai import OpenAI

# Create a reusable client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))

CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o")
EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")

def embed(texts: list[str]) -> list[list[float]]:
    """Create embeddings for multiple text chunks."""
    resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
    return [d.embedding for d in resp.data]

def embed_one(text: str) -> list[float]:
    """Embed a single text string."""
    return embed([text])[0]

ANSWER_PROMPT = """You are a careful assistant for a personal knowledge base.
Use ONLY the provided context snippets to answer. Cite sources as [1], [2], etc. (mapping to 'source').
If context is insufficient, say what is missing and ask a follow-up question.

Question:
{question}

Context:
{snippets}

Answer with citations.
"""

def build_answer(question: str, hits: List[Dict[str, Any]]) -> str:
    """Builds an answer to the question using retrieved text snippets."""
    numbered = []
    for i, h in enumerate(hits, start=1):
        src = h.get("source", "unknown")
        txt = (h.get("chunk", "") or "").replace("\n", " ").strip()
        numbered.append(f"[{i}] ({src}): {txt[:700]}")
    prompt = ANSWER_PROMPT.format(question=question, snippets="\n\n".join(numbered))

    chat = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": "You write concise, accurate answers with citations."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )
    return chat.choices[0].message.content.strip()
