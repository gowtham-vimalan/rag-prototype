from __future__ import annotations

import json
from typing import TypedDict

from langgraph.graph import StateGraph, END

from rag import get_collection, get_llm_client, SYSTEM_PROMPT, TOP_K


class AgentState(TypedDict):
    query: str
    original_query: str
    documents: list[dict]
    generation: str
    retry_count: int
    trace: list[str]
    chat_history: list[dict]


def retrieve(state: AgentState) -> dict:
    collection = get_collection()
    results = collection.query(query_texts=[state["query"]], n_results=TOP_K)
    docs = []
    for i in range(len(results["documents"][0])):
        docs.append({
            "text": results["documents"][0][i],
            "source": results["metadatas"][0][i]["source"],
            "distance": results["distances"][0][i],
        })
    return {"documents": docs, "trace": state["trace"] + [f"retrieve(query='{state['query']}')"]}


def grade_documents(state: AgentState) -> dict:
    client = get_llm_client()
    query = state["query"]
    docs = state["documents"]

    relevant_docs = []
    for doc in docs:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": (
                    "You are a relevance grader. Given a user question and a document, "
                    "determine if the document contains information that could help answer the question. "
                    "Be LENIENT — if the document mentions any related topics, facts, numbers, or policies "
                    "that could partially address the question, mark it as relevant. "
                    "Only mark irrelevant if the document is completely unrelated. "
                    "Respond with JSON: {\"relevant\": true} or {\"relevant\": false}"
                )},
                {"role": "user", "content": f"Question: {query}\n\nDocument: {doc['text'][:2000]}"},
            ],
            temperature=0,
        )
        try:
            result = json.loads(response.choices[0].message.content)
            if result.get("relevant"):
                relevant_docs.append(doc)
        except (json.JSONDecodeError, KeyError):
            relevant_docs.append(doc)

    return {
        "documents": relevant_docs,
        "trace": state["trace"] + [f"grade_documents({len(relevant_docs)}/{len(docs)} relevant)"],
    }


def rewrite_query(state: AgentState) -> dict:
    client = get_llm_client()
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": (
                "You are a query rewriter. Given an original question that didn't retrieve "
                "good results, rewrite it to be more specific and likely to match relevant documents. "
                "Return ONLY the rewritten query, nothing else."
            )},
            {"role": "user", "content": f"Original question: {state['original_query']}\nPrevious query: {state['query']}"},
        ],
        temperature=0.3,
    )
    new_query = response.choices[0].message.content.strip()
    return {
        "query": new_query,
        "retry_count": state["retry_count"] + 1,
        "trace": state["trace"] + [f"rewrite_query('{new_query}')"],
    }


def generate(state: AgentState) -> dict:
    docs = state["documents"]
    parts = [f"[Source: {doc['source']}]\n{doc['text']}" for doc in docs]
    context = "\n\n---\n\n".join(parts)

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if state.get("chat_history"):
        messages.extend(state["chat_history"])

    user_message = f"Context from company policies:\n\n{context}\n\nQuestion: {state['original_query']}"
    messages.append({"role": "user", "content": user_message})

    client = get_llm_client()
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.1,
    )
    return {
        "generation": response.choices[0].message.content,
        "trace": state["trace"] + ["generate"],
    }


def should_rewrite_or_generate(state: AgentState) -> str:
    if len(state["documents"]) >= 1:
        return "generate"
    if state["retry_count"] >= 2:
        return "generate"
    return "rewrite_query"


def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("retrieve", retrieve)
    graph.add_node("grade_documents", grade_documents)
    graph.add_node("rewrite_query", rewrite_query)
    graph.add_node("generate", generate)

    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "grade_documents")
    graph.add_conditional_edges("grade_documents", should_rewrite_or_generate, {
        "generate": "generate",
        "rewrite_query": "rewrite_query",
    })
    graph.add_edge("rewrite_query", "retrieve")
    graph.add_edge("generate", END)

    return graph.compile()


agent = build_graph()


def ask(query: str, chat_history: list[dict] | None = None) -> tuple[str, list[str]]:
    result = agent.invoke({
        "query": query,
        "original_query": query,
        "documents": [],
        "generation": "",
        "retry_count": 0,
        "trace": [],
        "chat_history": chat_history or [],
    })
    return result["generation"], result["trace"]


def stream_ask(query: str, chat_history: list[dict] | None = None):
    docs = _retrieve_with_grading(query, chat_history)
    trace = docs["trace"]

    context_parts = [f"[Source: {doc['source']}]\n{doc['text']}" for doc in docs["documents"]]
    context = "\n\n---\n\n".join(context_parts)

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if chat_history:
        messages.extend(chat_history)
    messages.append({"role": "user", "content": f"Context from company policies:\n\n{context}\n\nQuestion: {query}"})

    client = get_llm_client()
    stream = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.1,
        stream=True,
    )
    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content

    yield {"__trace__": trace + ["generate(streamed)"]}


def _retrieve_with_grading(query: str, chat_history: list[dict] | None = None) -> dict:
    """Run retrieve → grade → rewrite loop without the final generate step."""
    state = {
        "query": query,
        "original_query": query,
        "documents": [],
        "generation": "",
        "retry_count": 0,
        "trace": [],
        "chat_history": chat_history or [],
    }

    for _ in range(3):
        state.update(retrieve(state))
        state.update(grade_documents(state))
        if len(state["documents"]) >= 1 or state["retry_count"] >= 2:
            break
        state.update(rewrite_query(state))

    return state
