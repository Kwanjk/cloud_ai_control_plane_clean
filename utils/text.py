\
from typing import List
import PyPDF2

def load_pdf_bytes(file_obj) -> str:
    reader = PyPDF2.PdfReader(file_obj)
    text_parts = []
    for page in reader.pages:
        t = page.extract_text() or ""
        if t.strip():
            text_parts.append(t)
    return "\n".join(text_parts)

def load_txt_bytes(file_obj) -> str:
    data = file_obj.read()
    try:
        return data.decode("utf-8", errors="ignore")
    except Exception:
        return data.decode("latin1", errors="ignore")

def paragraph_chunk(text: str, max_chars: int = 900, overlap: int = 120) -> List[str]:
    paras = [p.strip() for p in text.split("\n") if p.strip()]
    chunks: List[str] = []
    buf = ""
    for p in paras:
        if len(buf) + len(p) + 1 <= max_chars:
            buf = (buf + "\n" + p).strip()
        else:
            if buf:
                chunks.append(buf)
            buf = p
    if buf:
        chunks.append(buf)

    if overlap > 0 and len(chunks) > 1:
        merged = []
        for i, c in enumerate(chunks):
            if i == 0:
                merged.append(c)
            else:
                tail = chunks[i-1][-overlap:]
                merged.append(tail + "\n" + c)
        return merged
    return chunks
