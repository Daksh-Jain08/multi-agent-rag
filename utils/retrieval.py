from __future__ import annotations

from typing import Iterable, List

from utils.schemas import Document


def simple_lexical_retrieve(query: str, documents: Iterable[Document], top_k: int = 5) -> List[Document]:
    query_terms = {t.strip().lower() for t in query.split() if t.strip()}
    scored = []
    for doc in documents:
        text_terms = set(doc.text.lower().split())
        score = len(query_terms.intersection(text_terms))
        scored.append((score, doc))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [doc for _, doc in scored[:top_k]]
