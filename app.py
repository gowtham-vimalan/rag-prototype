import streamlit as st
from rag import stream_ask, retrieve

st.set_page_config(page_title="Company Policy Assistant", page_icon="📋", layout="centered")
st.title("📋 Company Policy Assistant")
st.caption("Ask questions about company policies — leave, remote work, expenses, code of conduct")

if "messages" not in st.session_state:
    st.session_state.messages = []

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
        response = st.write_stream(stream_ask(prompt, history))

    st.session_state.messages.append({"role": "assistant", "content": response})

with st.sidebar:
    st.header("About")
    st.markdown(
        "This assistant uses **RAG** (Retrieval-Augmented Generation) to answer "
        "questions based on company policy documents.\n\n"
        "**Documents loaded:**\n"
        "- Leave Policy\n"
        "- Remote Work Policy\n"
        "- Expense Reimbursement Policy\n"
        "- Code of Conduct"
    )
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
