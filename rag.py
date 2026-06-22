import os
from pathlib import Path

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

CHROMA_DIR = Path(__file__).parent / "chroma_db"
COLLECTION_NAME = "company_policies"
TOP_K = 5

SYSTEM_PROMPT = """\
You are a helpful company policy assistant. Answer the user's question based ONLY on the provided context from company policy documents.

Before answering, follow this reasoning process step by step:
1. Identify the user's situation — their role, tenure, employment status, or any details they have shared.
2. Break the question into its individual parts if it asks about multiple things.
3. For each part, find the relevant policy conditions and eligibility requirements in the context.
4. Check whether the user meets ALL conditions before saying something is allowed or available.
5. If two policies relate to the same topic, reconcile them — do not treat them in isolation.

Rules:
- If the answer is found in the context, provide a clear and specific answer with relevant details (numbers, dates, limits, etc.).
- Cite which policy document the information comes from.
- If the context does not contain enough information to answer the question, say "I don't have enough information in the company policies to answer that question."
- Do not make up information or use knowledge outside the provided context.
- Do not give an opinion or compare policies with industry standards."""


def get_collection():
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    embedding_fn = SentenceTransformerEmbeddingFunction(model_name="BAAI/bge-base-en-v1.5")
    return client.get_collection(name=COLLECTION_NAME, embedding_function=embedding_fn)


def get_llm_client() -> OpenAI:
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        raise RuntimeError("GITHUB_TOKEN environment variable is not set.")
    return OpenAI(
        base_url="https://models.inference.ai.azure.com",
        api_key=token,
    )


def retrieve(query: str, top_k: int = TOP_K) -> list[dict]:
    collection = get_collection()
    results = collection.query(query_texts=[query], n_results=top_k)
    docs = []
    for i in range(len(results["documents"][0])):
        docs.append({
            "text": results["documents"][0][i],
            "source": results["metadatas"][0][i]["source"],
            "distance": results["distances"][0][i],
        })
    return docs


def build_context(docs: list[dict]) -> str:
    parts = []
    for i, doc in enumerate(docs, 1):
        parts.append(f"[Source: {doc['source']}]\n{doc['text']}")
    return "\n\n---\n\n".join(parts)


def ask(query: str, chat_history: list[dict] | None = None) -> str:
    docs = retrieve(query)
    context = build_context(docs)

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    if chat_history:
        messages.extend(chat_history)

    user_message = f"Context from company policies:\n\n{context}\n\nQuestion: {query}"
    messages.append({"role": "user", "content": user_message})

    client = get_llm_client()
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.1,
    )
    return response.choices[0].message.content


def stream_ask(query: str, chat_history: list[dict] | None = None):
    docs = retrieve(query)
    context = build_context(docs)

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    if chat_history:
        messages.extend(chat_history)

    user_message = f"Context from company policies:\n\n{context}\n\nQuestion: {query}"
    messages.append({"role": "user", "content": user_message})

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
