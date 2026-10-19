"""Reusable ReAct RAG agent extracted from RAG_Model(1).ipynb.

The Streamlit UI imports `graph` from this module instead of duplicating agent code.
"""

import os
from pathlib import Path
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage
from langchain_tavily import TavilySearch
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

# Keep the model configured in your working notebook.
llm = ChatOpenAI(
    model="prism-ml/ternary-bonsai-2-27b",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    max_tokens=1000,
)

# TavilySearch reads this environment variable.
if os.getenv("TAVILY_API_KEY"):
    os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY", "")

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Uses the FAISS index already created by your notebook; does not rebuild it.
FAISS_INDEX_DIR = Path(os.getenv("FAISS_INDEX_DIR", str(BASE_DIR / "faiss_pdf_index")))
vectorstore = FAISS.load_local(
    str(FAISS_INDEX_DIR),
    embeddings,
    allow_dangerous_deserialization=True,
)


def rag_answer(query: str) -> str:
    """Answer from indexed PDFs, or return a sentinel if the PDFs cannot answer."""
    retrieved_docs = vectorstore.similarity_search(query, k=4)

    if not retrieved_docs:
        return "NO_RELEVANT_DOCUMENTS_FOUND"

    context_parts = []
    references = []
    seen = set()

    for doc in retrieved_docs:
        source = doc.metadata.get("source", "Unknown")
        page_index = doc.metadata.get("page")
        # PyPDFLoader page metadata is zero-based; display human-readable page numbers.
        page_number = page_index + 1 if isinstance(page_index, int) else page_index

        context_parts.append(
            f"Source: {source}\nPage: {page_number}\nContent: {doc.page_content}"
        )

        ref = (source, page_number)
        if ref not in seen:
            seen.add(ref)
            references.append(f"- {source}, page {page_number}")

    context = "\n\n".join(context_parts)
    prompt = f"""
You answer questions using ONLY the supplied PDF context.
If the context does not contain enough information to answer the question,
return exactly this token and nothing else: NO_RELEVANT_DOCUMENTS_FOUND
Do not guess or use outside knowledge.
Do not write a Sources/References section; the application adds references.

PDF context:
{context}

Question: {query}

Answer:
"""

    response = llm.invoke(prompt)
    answer = response.content.strip()

    if "NO_RELEVANT_DOCUMENTS_FOUND" in answer:
        return "NO_RELEVANT_DOCUMENTS_FOUND"

    if not answer:
        return "NO_RELEVANT_DOCUMENTS_FOUND"

    return f"{answer}\n\nSources:\n" + "\n".join(references)


from langchain_core.tools import tool


@tool
def rag_search(query: str) -> str:
    """Search the indexed PDFs first. If the PDFs contain an answer, return it with source filename and PDF page number. If the PDFs cannot answer, return exactly NO_RELEVANT_DOCUMENTS_FOUND. Use this tool before web search."""
    return rag_answer(query)


# Initialize Tavily and both tools, as in the notebook.
tavily_search = TavilySearch(max_results=2)
tools = [rag_search, tavily_search]


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


SYSTEM_PROMPT = """
You are a ReAct assistant with two tools:
- rag_search: answers using the user's indexed PDF documents and returns PDF filenames/page numbers.
- tavily_search: searches the web and returns web results and URLs.

Tool-use rules:
1. For each new user request, call rag_search first. Do not answer before doing so.
2. Inspect the complete RAG observation. If it returns NO_RELEVANT_DOCUMENTS_FOUND, or does not answer every part of the request, use tavily_search.
3. After EVERY tool result, reason again. Decide whether the original request is fully answered or another tool call is needed.
4. For multi-step questions, do not stop after answering only the first part. Use observations to formulate a useful follow-up query when needed.
5. You may call Tavily more than once with a refined query if the first results are insufficient. Do not repeat the same search without a reason.
6. Use RAG and Tavily results as evidence; do not invent facts, document names, page numbers, or URLs.
7. If the evidence is insufficient after reasonable tool calls, clearly say what is missing.
8. If the PDF result answers the whole question, preserve its Sources section with filenames and page numbers. If web results are used, include relevant URLs.
9. Return a concise final answer only when the user's complete request has been addressed.
"""

llm_with_tools = llm.bind_tools(tools)


def agent_node(state: AgentState):
    messages = [SystemMessage(content=SYSTEM_PROMPT), *state["messages"]]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}


builder = StateGraph(AgentState)
builder.add_node("agent", agent_node)
builder.add_node("tools", ToolNode(tools))
builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", tools_condition)
builder.add_edge("tools", "agent")
graph = builder.compile()


def ask_agent(messages):
    """Run the graph with a list of LangChain messages and return its final text."""
    result = graph.invoke({"messages": messages})
    for message in reversed(result["messages"]):
        # The final response is an AI message without pending tool calls.
        if getattr(message, "type", None) == "ai" and not getattr(
            message, "tool_calls", None
        ):
            content = message.content
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                return "\n".join(
                    str(part.get("text", part)) if isinstance(part, dict) else str(part)
                    for part in content
                )
            return str(content)
    return "The agent did not return a final answer."
