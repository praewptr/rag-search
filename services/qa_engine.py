import json
import re
from typing import Any, Dict, List

from azure.search.documents.models import VectorizedQuery
from fastapi import HTTPException
from openai import AzureOpenAI

from config import (
    azure_oai_deployment,
    azure_search_endpoint,
    azure_search_index_doc,
    azure_search_index_txt,
    azure_search_key,
)
from services.client import embeddings, search_client_pdf, search_client_text

extension_config = dict(
    dataSources=[
        {
            "type": "AzureCognitiveSearch",
            "parameters": {
                "endpoint": azure_search_endpoint,
                "key": azure_search_key,
                "indexName": azure_search_index_doc,
            },
            "fields_mapping": {
                "content_field": "chunk",
                "filepath_field": "text_parent_id",
                "title_field": "title",
                "url_field": "",
                "vector_field": "text_vector",
            },
        },
        {
            "type": "AzureCognitiveSearch",
            "parameters": {
                "endpoint": azure_search_endpoint,
                "key": azure_search_key,
                "indexName": azure_search_index_txt,
            },
            "fields_mapping": {
                "content_field": "content",
                "filepath_field": "source",
                "title_field": "title",
                "url_field": "",
                "vector_field": "text_vector",
            },
        },
    ]
)


def get_response(
    text: str,
    client: AzureOpenAI,
    context: str = "",
):
    response = client.chat.completions.create(
        model=azure_oai_deployment,
        temperature=0.3,
        max_tokens=700,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an expert assistant providing accurate answers based **strictly on document context**. "
                    "**CRITICAL: Respond in the SAME language as the user's question.**\n\n"
                    "**Language Rules:**\n"
                    "- English question → English answer (clear, professional)\n"
                    "- Thai question → Thai answer (natural, conversational Thai - not direct translation)\n"
                    "- Focus on main language, ignore technical terms in parentheses\n\n"
                    "**Format:**\n"
                    "- Use bullet points (•) and headings for clarity\n"
                    "- Be comprehensive but concise\n"
                    "- Professional yet friendly tone\n\n"
                    "**Strict Rules:**\n"
                    "- ONLY use information from provided documents\n"
                    "- NO assumptions beyond document content\n\n"
                    "**If insufficient info:**\n"
                    "- Thai: 'ขออภัย ไม่มีข้อมูลเพียงพอในเอกสารสำหรับคำถามนี้'\n"
                    "- English: 'I don't have sufficient information in the documents to answer this question.'\n"
                ),
            },
            {
                "role": "user",
                "content": f"""
            Answer this question using ONLY the provided documents.

            **IMPORTANT: Respond in the same language as the question.**
            Context : {context}
            Question: {text}

            Provide a detailed answer with clear formatting.
            """,
            },
        ],
        extra_body=extension_config,
    )

    answer = response.choices[0].message.content
    citations = []

    try:
        context_messages = response.choices[0].message.context.get("messages", [])
        if context_messages:
            citation_json = json.loads(context_messages[0].get("content", "{}"))
            if "citations" in citation_json:
                citations = citation_json["citations"]
    except Exception:
        # No citations or context — skip silently
        pass

    return answer, citations


# --- Helper functions (remain unchanged) ---
def fix_line_breaks(text: str) -> str:
    if not isinstance(text, str):
        return ""
    return re.sub(r"(?<!\n)\n(?!\n)", " ", text).strip()


def _process_search_results(
    results: List[Dict[str, Any]], content_field: str
) -> List[Dict[str, Any]]:
    return [
        {"score": result["@search.score"], "content": result[content_field]}
        for result in results
    ]


def get_llm_answer(query: str, context: str, openai_client: AzureOpenAI):
    """Gets a final answer from the LLM based on the query and retrieved context."""

    # If no context or empty context, return None to indicate no answer
    if not context or context.strip() == "":
        return None

    system_prompt = """
    You are a helpful AI assistant. Answer the user's question using ONLY the provided CONTEXT below.
    - Focus on providing an answer that is directly relevant to the user's question.
    - If the context contains unrelated or extra information, IGNORE it and do not include it in your answer.
    - If the information in the context is not sufficient to answer the question, respond with "NO_ANSWER_FOUND".
    - Be concise and professional. Do not cite the source file for the information you use.
    - Always answer in the SAME LANGUAGE as the question. If the question is in Thai, answer in Thai. If in English, answer in English.
    """

    user_prompt = f"CONTEXT:\n---\n{context}\n---\n\nQUESTION: {query}\n\nANSWER:"

    try:
        response = openai_client.chat.completions.create(
            model=azure_oai_deployment,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
            max_tokens=700,
        )

        answer = response.choices[0].message.content.strip()

        # Check if LLM indicates no answer found
        if "NO_ANSWER_FOUND" in answer or not answer:
            return []

        return answer

    except Exception as e:
        return f"Error generating answer: {str(e)}"


async def rag_pipeline(question: str, openai_client: AzureOpenAI) -> str:
    """
    An improved, asynchronous RAG pipeline that retrieves context, processes it,
    and generates a final answer using an LLM.
    """
    try:
        question_emb = embeddings.embed_query(question)
        vector_query = VectorizedQuery(
            vector=question_emb, k_nearest_neighbors=3, fields="text_vector"
        )

        # --- Step 1: Search documents concurrently ---
        text_results = search_client_text.search(
            search_text=None, vector_queries=[vector_query]
        )
        pdf_results = search_client_pdf.search(
            search_text=None, vector_queries=[vector_query]
        )

        # --- Step 2: Combine and process results ---
        combined_results = _process_search_results(text_results, "content")
        combined_results.extend(_process_search_results(pdf_results, "chunk"))

        sorted_results = sorted(
            combined_results, key=lambda x: x["score"], reverse=True
        )

        top_results = [res for res in sorted_results if res["score"] >= 0.5][:3]

        if not top_results:
            return []

        # --- Step 3: Assemble and clean the context for the LLM ---
        context_for_llm = ""
        for i, result in enumerate(top_results, 1):
            cleaned_content = fix_line_breaks(result["content"])
            context_for_llm += f"--- Excerpt {i} ---\n{cleaned_content}\n\n"

        # print(f"Final Context for LLM:\n{context_for_llm}")

        # --- Step 4: Generate the final answer ---
        final_answer = get_llm_answer(question, context_for_llm, openai_client)
        if final_answer is None:
            return []
        else:
            return final_answer

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {e}")
