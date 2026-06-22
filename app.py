import streamlit as st
from agentic_rag import stream_ask, ask
from rag import retrieve

st.set_page_config(page_title="Company Policy Assistant", page_icon="📋", layout="centered")
st.title("📋 Company Policy Assistant")
st.caption("Ask questions about company policies — powered by Agentic RAG with LangGraph")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "traces" not in st.session_state:
    st.session_state.traces = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask a question about company policies..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    history = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.messages[:-1]
    ]

    with st.chat_message("assistant"):
        trace = []
        chunks = []
        for token in stream_ask(prompt, history):
            if isinstance(token, dict) and "__trace__" in token:
                trace = token["__trace__"]
            else:
                chunks.append(token)

        response = "".join(chunks)
        st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
    st.session_state.traces.append(trace)

with st.sidebar:
    st.header("About")
    st.markdown(
        "This assistant uses **Agentic RAG** with LangGraph to answer "
        "questions based on company policy documents.\n\n"
        "The agent can:\n"
        "- **Retrieve** relevant documents\n"
        "- **Grade** document relevance\n"
        "- **Rewrite** queries for better results\n"
        "- **Generate** answers from verified sources"
    )

    if st.session_state.traces:
        st.header("Agent Trace")
        latest_trace = st.session_state.traces[-1]
        for step in latest_trace:
            st.code(step, language=None)

    if st.button("🔍 Show source documents for last query"):
        if st.session_state.messages:
            last_user = next(
                (m["content"] for m in reversed(st.session_state.messages) if m["role"] == "user"),
                None,
            )
            if last_user:
                docs = retrieve(last_user)
                for doc in docs:
                    with st.expander(f"📄 {doc['source']} (similarity: {1 - doc['distance']:.2f})"):
                        st.text(doc["text"])
