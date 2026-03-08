"""
Graph RAG: RAG pipeline as a LangGraph workflow — query expansion -> retrieve -> generate.
The graph defines the retrieval pipeline; optional rewrite step improves retrieval recall.
Uses .env: BASE_URL, AI_GRID_KEY, EMBEDDING_MODEL, OSS_MODEL (or QWEN_MODEL).
"""
import math
import os
from pathlib import Path
from typing import TypedDict

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from openai import OpenAI

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

AI_GRID_KEY = os.getenv("AI_GRID_KEY")
BASE_URL = os.getenv("BASE_URL")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
CHAT_MODEL = os.getenv("OSS_MODEL") or os.getenv("QWEN_MODEL", "gpt-oss-120b")

openai_client = OpenAI(base_url=BASE_URL, api_key=AI_GRID_KEY)

DOCS = [
    "LangGraph is a library for building stateful, multi-actor applications with LLMs.",
    "Embeddings are dense vectors that represent the meaning of text; similar texts have similar vectors.",
    "RAG stands for Retrieval-Augmented Generation: retrieve relevant documents, then generate an answer.",
    "Graph RAG structures the RAG pipeline as a graph of steps: e.g. rewrite query, retrieve, generate.",
    "Query expansion improves retrieval by generating alternative phrasings of the user question.",
    "OpenAI-compatible APIs expose /v1/chat/completions and /v1/embeddings.",
]


class GraphRAGState(TypedDict):
    """State: query, expanded_queries, retrieved_docs, and final answer."""
    query: str
    expanded_queries: list[str]
    retrieved_docs: list[str]
    answer: str


def get_embedding(text: str) -> list[float]:
    """Return the embedding vector for the given text."""
    r = openai_client.embeddings.create(model=EMBEDDING_MODEL, input=text)
    return r.data[0].embedding


def cosine(a: list[float], b: list[float]) -> float:
    """Return cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def retrieve_for_query(query: str, k: int = 2) -> list[str]:
    """Embed a single query and return top-k docs by cosine similarity."""
    q_emb = get_embedding(query)
    doc_embs = [get_embedding(d) for d in DOCS]
    scores = [(cosine(q_emb, d_emb), d) for d_emb, d in zip(doc_embs, DOCS)]
    scores.sort(key=lambda x: -x[0])
    return [d for _, d in scores[:k]]


def rewrite_query_node(state: GraphRAGState) -> dict:
    """
    Graph node: expand the user query into 1–2 search queries to improve retrieval recall.
    """
    query = state["query"]
    llm = ChatOpenAI(
        model=CHAT_MODEL,
        openai_api_key=AI_GRID_KEY,
        openai_api_base=BASE_URL,
        temperature=0.3,
    )
    prompt = (
        f"Original question: {query}\n\n"
        "Output 1 or 2 short search queries (one per line) that would help find relevant documents. "
        "No numbering, no explanation."
    )
    response = llm.invoke([HumanMessage(content=prompt)])
    text = response.content.strip() if isinstance(response, AIMessage) else str(response)
    lines = [q.strip() for q in text.splitlines() if q.strip()][:2]
    expanded = [query] + [q for q in lines if q != query]
    return {"expanded_queries": expanded}


def retrieve_node(state: GraphRAGState) -> dict:
    """
    Graph node: for each expanded query, retrieve top-k docs; merge and dedupe by order.
    """
    queries = state.get("expanded_queries") or [state["query"]]
    all_docs = []
    seen = set()
    for q in queries:
        for d in retrieve_for_query(q, k=2):
            if d not in seen:
                seen.add(d)
                all_docs.append(d)
    return {"retrieved_docs": all_docs[:6]}


def generate_node(state: GraphRAGState) -> dict:
    """
    Graph node: build a prompt from retrieved_docs and query, return final answer.
    """
    docs = state.get("retrieved_docs") or []
    query = state["query"]
    context = "\n".join(f"- {d}" for d in docs) if docs else "No documents retrieved."
    llm = ChatOpenAI(
        model=CHAT_MODEL,
        openai_api_key=AI_GRID_KEY,
        openai_api_base=BASE_URL,
        temperature=0.3,
    )
    prompt = (
        f"Context:\n{context}\n\nQuestion: {query}\n\n"
        "Answer based on the context. Be concise."
    )
    response = llm.invoke([HumanMessage(content=prompt)])
    answer = response.content if isinstance(response, AIMessage) else str(response)
    return {"answer": answer}


def build_graph_rag_graph():
    """
    Build the Graph RAG workflow: START -> rewrite_query -> retrieve -> generate -> END.
    """
    graph = StateGraph(GraphRAGState)
    graph.add_node("rewrite_query", rewrite_query_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("generate", generate_node)
    graph.add_edge(START, "rewrite_query")
    graph.add_edge("rewrite_query", "retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)
    return graph.compile()


def run_graph_rag(query: str) -> GraphRAGState:
    """Run the Graph RAG pipeline for one query; returns state with answer and retrieved_docs."""
    app = build_graph_rag_graph()
    initial: GraphRAGState = {
        "query": query,
        "expanded_queries": [],
        "retrieved_docs": [],
        "answer": "",
    }
    return app.invoke(initial)


def main():
    """Run Graph RAG on a sample query and print expanded queries, doc count, and answer."""
    query = "How does query expansion help in RAG?"
    print("Query:", query)
    print()
    result = run_graph_rag(query)
    expanded = result.get("expanded_queries") or []
    docs = result.get("retrieved_docs") or []
    print("Expanded queries:", expanded)
    print(f"Retrieved {len(docs)} doc(s).")
    print("Answer:", result.get("answer", ""))


if __name__ == "__main__":
    main()
