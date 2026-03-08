"""
Minimal RAG: embed a small corpus and a query, retrieve top-k by similarity, then answer with context.
Uses embedding model and chat model from .env (EMBEDDING_MODEL, BASE_URL, AI_GRID_KEY; OSS_MODEL or QWEN_MODEL for chat).
"""
import math
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

AI_GRID_KEY = os.getenv("AI_GRID_KEY")
BASE_URL = os.getenv("BASE_URL")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
CHAT_MODEL = os.getenv("OSS_MODEL") or os.getenv("QWEN_MODEL", "gpt-oss-120b")

client = OpenAI(base_url=BASE_URL, api_key=AI_GRID_KEY)

DOCS = [
    "LangGraph is a library for building stateful, multi-actor applications with LLMs.",
    "Embeddings are dense vectors that represent the meaning of text; similar texts have similar vectors.",
    "RAG stands for Retrieval-Augmented Generation: retrieve relevant documents, then generate an answer using them.",
    "OpenAI-compatible APIs expose endpoints like /v1/chat/completions and /v1/embeddings.",
]


def get_embedding(text: str) -> list[float]:
    """Return the embedding vector for the given text."""
    r = client.embeddings.create(model=EMBEDDING_MODEL, input=text)
    return r.data[0].embedding


def cosine(a: list[float], b: list[float]) -> float:
    """Return cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def retrieve(query: str, docs: list[str], k: int = 2) -> list[str]:
    """Embed query and docs, return the top-k docs by cosine similarity."""
    q_emb = get_embedding(query)
    doc_embs = [get_embedding(d) for d in docs]
    scores = [(cosine(q_emb, d_emb), d) for d_emb, d in zip(doc_embs, docs)]
    scores.sort(key=lambda x: -x[0])
    return [d for _, d in scores[:k]]


def answer_with_context(query: str, context_docs: list[str]) -> str:
    """Build a prompt with context and return the chat model reply."""
    context = "\n".join(f"- {d}" for d in context_docs)
    prompt = f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer briefly using the context."
    r = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    return r.choices[0].message.content or ""


def main():
    """Run RAG: retrieve top-2 docs for a query, then generate an answer."""
    query = "What is RAG and how do embeddings help?"
    print("Query:", query)
    print("\nRetrieving...")
    top = retrieve(query, DOCS, k=2)
    print("Top docs:", [d[:60] + "..." if len(d) > 60 else d for d in top])
    print("\nAnswer:")
    answer = answer_with_context(query, top)
    print(answer)


if __name__ == "__main__":
    main()
