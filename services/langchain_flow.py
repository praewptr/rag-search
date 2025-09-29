from typing import List

from langchain.docstore.document import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableMap, RunnablePassthrough

from config import azure_search_index_doc, azure_search_index_txt
from services.client import llm, retriever_pdf, retriever_text

prompt = ChatPromptTemplate.from_template(
    """You are an expert assistant providing answers based strictly on document context. 
    **CRITICAL: Respond in the SAME language as the user's question.**

    **Language Rules:**
    - English question → English answer (clear, professional)
    - Thai question → Thai answer (natural, conversational Thai - not direct translation)
    - Focus on main language

    **Format:**
    - headings for clarity
    - Be comprehensive but concise
    - Professional yet friendly tone

    **Strict Rules:**
    - ONLY use information from provided documents
    - NO citation markers [doc1], [source], URLs
    - NO assumptions beyond document content



    Context: {context}

    Question: {question}"""
)


def format_docs(docs: List[Document]) -> str:
    """
    Format documents for prompt context, including source information.
    """
    if not docs:
        return ""

    return "\n\n--- Document ---\n\n".join(
        f"Source: {doc.metadata.get('source_index', 'unknown')}\n{doc.page_content}"
        for doc in docs
    )


def retrieve_and_rank(question: str, top_k: int = 5) -> List[Document]:
    """
    Retrieve documents from both retrievers, merge and sort by score.
    """
    try:
        results_pdf = retriever_pdf.invoke(question)
        results_text = retriever_text.invoke(question)

        combined_results = []

        for doc in results_pdf:
            doc.metadata["source_index"] = azure_search_index_doc
            combined_results.append(doc)

        for doc in results_text:
            doc.metadata["source_index"] = azure_search_index_txt
            combined_results.append(doc)

        ranked = sorted(
            combined_results,
            key=lambda d: d.metadata.get("@search.score", 0),
            reverse=True,
        )

        return ranked[:top_k]

    except Exception as e:
        print(f"[Error][retrieve_and_rank] {e}")
        return []


def generate_answer(question: str) -> str:
    """
    Generate an answer for the question using retrieved documents and LLM.
    """
    try:
        ranked_docs = retrieve_and_rank(question)
        context = format_docs(ranked_docs)

        # If no context, return insufficient info message directly
        if not context.strip():
            if any(
                "\u0e00" <= c <= "\u0e7f" for c in question
            ):  # Thai unicode range approx
                return []
            else:
                return []

        # Build a static input dict for RunnableMap to avoid redundant retrievals
        inputs = {
            "context": context,
            "question": question,
        }

        chain = (
            RunnableMap(
                {
                    "context": RunnablePassthrough(),
                    "question": RunnablePassthrough(),
                }
            )
            | prompt
            | llm
            | StrOutputParser()
        )

        answer = chain.invoke(inputs)

        print("\n--- Answer ---\n")
        print(answer)

        return answer

    except Exception as e:
        error_msg = f"[Error][generate_answer] {e}"
        print(error_msg)
        return error_msg


if __name__ == "__main__":
    q = "วิธีการตั้งค่า machine"
    print(generate_answer(q))
