"""
Agentic RAG: LangGraph agent that retrieves (optionally multiple times) then generates an answer.
State holds messages, retrieved docs, and query; the graph can loop retrieve -> generate until satisfied.
Uses .env: BASE_URL, AI_GRID_KEY, EMBEDDING_MODEL, OSS_MODEL (or QWEN_MODEL).
"""
import math
import os
from pathlib import Path
from typing import Annotated, Literal, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
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
    "RAG stands for Retrieval-Augmented Generation: retrieve relevant documents, then generate an answer using them.",
    "Agentic RAG uses an agent that decides when to retrieve and when to answer; it can call retrieve multiple times.",
    "OpenAI-compatible APIs expose /v1/chat/completions and /v1/embeddings.",
]


class AgenticRAGState(TypedDict):
    """State: messages, latest query, accumulated retrieved docs, and step count."""
    messages: Annotated[list, add_messages]
    query: str
    retrieved_docs: list[str]
    step: int


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


def retrieve_node(state: AgenticRAGState) -> dict:
    """
    Agent tool node: embed the current query, get top-k docs by similarity, append to retrieved_docs.
    """
    query = state["query"]
    k = 2
    q_emb = get_embedding(query)
    doc_embs = [get_embedding(d) for d in DOCS]
    scores = [(cosine(q_emb, d_emb), d) for d_emb, d in zip(doc_embs, DOCS)]
    scores.sort(key=lambda x: -x[0])
    new_docs = [d for _, d in scores[:k]]
    existing = state.get("retrieved_docs") or []
    combined = list(dict.fromkeys(existing + new_docs))[:5]
    return {
        "retrieved_docs": combined,
        "step": state.get("step", 0) + 1,
    }


def generate_node(state: AgenticRAGState) -> dict:
    """
    Generate an answer using the current messages and retrieved_docs; append assistant message.
    """
    docs = state.get("retrieved_docs") or []
    query = state["query"]
    context = "\n".join(f"- {d}" for d in docs) if docs else "(No documents retrieved yet.)"
    llm = ChatOpenAI(
        model=CHAT_MODEL,
        openai_api_key=AI_GRID_KEY,
        openai_api_base=BASE_URL,
        temperature=0.3,
    )
    prompt = (
        f"Context:\n{context}\n\nQuestion: {query}\n\n"
        "Answer based on the context. If the context is insufficient, say so briefly."
    )
    response = llm.invoke([HumanMessage(content=prompt)])
    return {"messages": [response], "step": state.get("step", 0) + 1}


def should_retrieve_again(state: AgenticRAGState) -> Literal["retrieve", "end"]:
    """
    After generate: end if we already retrieved at least once and have docs; else retrieve once then end.
    """
    step = state.get("step", 0)
    docs = state.get("retrieved_docs") or []
    if step >= 2 and docs:
        return "end"
    if not docs:
        return "retrieve"
    return "end"


def build_agentic_rag_graph():
    """
    Build the agentic RAG graph: START -> retrieve -> generate -> (retrieve again or end).
    """
    graph = StateGraph(AgenticRAGState)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("generate", generate_node)
    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_conditional_edges("generate", should_retrieve_again, {"retrieve": "retrieve", "end": END})
    return graph.compile()


def run_agentic_rag(query: str) -> AgenticRAGState:
    """Run the agentic RAG graph for one query; returns final state with messages and retrieved_docs."""
    app = build_agentic_rag_graph()
    initial: AgenticRAGState = {
        "messages": [HumanMessage(content=query)],
        "query": query,
        "retrieved_docs": [],
        "step": 0,
    }
    return app.invoke(initial)


def main():
    """Run agentic RAG on a sample query and print the answer and doc count."""
    query = "What is agentic RAG and how does it differ from simple RAG?"
    print("Query:", query)
    print()
    result = run_agentic_rag(query)
    docs = result.get("retrieved_docs") or []
    print(f"Retrieved {len(docs)} doc(s). Steps: {result.get('step', 0)}")
    for msg in result["messages"]:
        if isinstance(msg, AIMessage):
            print("Answer:", msg.content)
            break


if __name__ == "__main__":
    main()
