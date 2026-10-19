import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from agent import ask_agent

st.set_page_config(page_title="agentic RAG system", page_icon="💬")
st.title("ReAct Agent")
st.caption(
    "Ask a question. Your existing LangGraph agent can search indexed PDFs and the web."
)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

with st.sidebar:
    st.subheader("Controls")
    if st.button("Clear conversation", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()
    st.caption(
        "The agent logic is imported from agent.py; it is not redefined in this UI file."
    )

for item in st.session_state.chat_history:
    with st.chat_message(item["role"]):
        st.markdown(item["content"])

prompt = st.chat_input("Ask your question...")
if prompt:
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Convert visible chat history to LangChain messages. The graph handles its own tool loop.
    lc_messages = []
    for item in st.session_state.chat_history:
        if item["role"] == "user":
            lc_messages.append(HumanMessage(content=item["content"]))
        else:
            lc_messages.append(AIMessage(content=item["content"]))

    with st.chat_message("assistant"):
        with st.spinner("Agent is thinking and using tools if needed..."):
            try:
                answer = ask_agent(lc_messages)
                st.markdown(answer)
            except Exception as exc:
                answer = f"**Error:** {exc}\n\nCheck your API keys, FAISS index path, and installed dependencies."
                st.error(answer)

    st.session_state.chat_history.append({"role": "assistant", "content": answer})
