import os
from dotenv import load_dotenv

from src.vectorstore import FaissVectorStore
from langchain_openai import ChatOpenAI


load_dotenv()


class RAGSearch:
    def __init__(
        self,
        persist_dir: str = "faiss_store",
        embedding_model: str = "all-MiniLM-L6-v2",
        llm_model: str = "prism-ml/ternary-bonsai-2-27b",
    ):

        self.vectorstore = FaissVectorStore(persist_dir, embedding_model)

        # Load or build vectorstore
        faiss_path = os.path.join(persist_dir, "faiss.index")

        meta_path = os.path.join(persist_dir, "metadata.pkl")

        if not (os.path.exists(faiss_path) and os.path.exists(meta_path)):
            from src.data_loader import load_all_documents

            docs = load_all_documents("data")

            self.vectorstore.build_from_documents(docs)

        else:
            self.vectorstore.load()

        # OpenRouter LLM
        openrouter_api_key = os.getenv("OPENROUTER_API_KEY")

        self.llm = ChatOpenAI(
            model=llm_model,
            api_key=openrouter_api_key,
            base_url="https://openrouter.ai/api/v1",
            max_tokens=1000,
        )

        print(f"[INFO] OpenRouter LLM initialized: {llm_model}")

    def search_and_summarize(self, query: str, top_k: int = 5) -> str:

        results = self.vectorstore.query(query, top_k=top_k)

        texts = [r["metadata"].get("text", "") for r in results if r["metadata"]]

        context = "\n\n".join(texts)

        if not context:
            return "No relevant documents found."

        prompt = f"""
Summarize the following context for the query:

Query:
{query}

Context:
{context}

Summary:
"""

        response = self.llm.invoke([prompt])

        return response.content


# Example usage
if __name__ == "__main__":
    rag_search = RAGSearch()

    query = "What is attention mechanism?"

    summary = rag_search.search_and_summarize(query, top_k=3)

    print("\nSummary:")
    print(summary)
