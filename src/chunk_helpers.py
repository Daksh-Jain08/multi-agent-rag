"""Helper functions for embedding-based chunk selection."""
from typing import Dict, List
from sentence_transformers import SentenceTransformer
import numpy as np

embedding_model = None

def get_embedding_model():
    """Lazy load embedding model once."""
    global embedding_model
    if embedding_model is None:
        print("[CHUNK-SELECTION] Loading embedding model: all-MiniLM-L6-v2")
        embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    return embedding_model

def split_into_chunks(text: str, max_sentences: int = 3) -> List[str]:
    """Split text into chunks of sentences."""
    if not text or not isinstance(text, str):
        return []
    sentences = text.split(". ")
    chunks = []
    for i in range(0, len(sentences), max_sentences):
        chunk = ". ".join(sentences[i:i+max_sentences])
        if chunk.strip():
            chunks.append(chunk.strip())
    return chunks

def get_top_k_chunks(query: str, chunks: List[str], k: int = 3) -> List[str]:
    """Select top-k most relevant chunks using embedding similarity."""
    if not chunks or not query:
        return chunks
    
    if len(chunks) <= k:
        return chunks
    
    model = get_embedding_model()
    query_emb = model.encode(query)
    chunk_embs = model.encode(chunks)
    
    scores = np.dot(chunk_embs, query_emb)
    top_indices = np.argsort(scores)[-k:][::-1]
    
    top_chunks = [chunks[i] for i in top_indices]
    return top_chunks

def reduce_document_with_chunks(query: str, doc: Dict[str, str], k: int = 3) -> Dict[str, str]:
    """Reduce document to top-k relevant chunks based on query."""
    text = doc.get("text", "")
    if not text or len(text) < 100:
        return doc
    
    chunks = split_into_chunks(text)
    if len(chunks) <= k:
        return doc
    
    top_chunks = get_top_k_chunks(query, chunks, k=k)
    reduced_text = " ".join(top_chunks)
    
    reduced_doc = dict(doc)
    reduced_doc["text"] = reduced_text
    return reduced_doc

def reduce_documents(query: str, docs: List[Dict[str, str]], k: int = 3) -> List[Dict[str, str]]:
    """Reduce all documents to top-k relevant chunks per document."""
    reduced_docs = []
    for doc in docs:
        reduced_doc = reduce_document_with_chunks(query, doc, k=k)
        reduced_docs.append(reduced_doc)
    return reduced_docs
