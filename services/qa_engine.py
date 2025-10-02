import json

from openai import AzureOpenAI
from azure.search.documents.models import VectorizedQuery
from config import (
    azure_oai_deployment,
    azure_search_endpoint,
    azure_search_index_doc,
    azure_search_key,
    azure_search_index_txt
)
from services.client import embeddings,search_client_text,search_client_pdf
import asyncio

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
        }
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

def get_llm_answer(query: str, context: str, openai_client: AzureOpenAI):
    """Gets a final answer from the LLM based on the query and retrieved context."""
    
    # If no context or empty context, return None to indicate no answer
    if not context or context.strip() == "":
        return None
        
    system_prompt = """
    You are a helpful AI assistant. Answer the user's question based ONLY on the provided information.
    If the information is not sufficient to answer the question, respond with "NO_ANSWER_FOUND".
    Be concise and professional. Do not cite the source file for the information you use.
    """
    
    user_prompt = f"""
    CONTEXT:
    ---
    {context}
    ---
    QUESTION: {query}
    
    ANSWER:
    """
    
    try:
        response = openai_client.chat.completions.create(
            model=azure_oai_deployment,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,
            max_tokens=500,
        )
        
        answer = response.choices[0].message.content.strip()
        
        # Check if LLM indicates no answer found
        if "NO_ANSWER_FOUND" in answer or not answer:
            return None
            
        return answer
        
    except Exception as e:
        print(f"Error calling LLM: {e}")
        return None



def rag_pipeline(question:str):
    question_emb = embeddings.embed_query(question)
    vector_query_text = VectorizedQuery(vector=question_emb, k_nearest_neighbors=3, fields="text_vector")
    vector_query_pdf = VectorizedQuery(vector=question_emb, k_nearest_neighbors=3, fields="text_vector")
    combined_results = []
    SCORE_THRESHOLD = 0.55
    try:
        # Run the search on the text index and wait for it to complete
        text_results = search_client_text.search(search_text=None, vector_queries=[vector_query_text])
        
        # Process results from the text index using a standard 'for' loop
        for result in text_results:
            combined_results.append({
                "score": result['@search.score'],
                "content": result['content']
            })
            
        # Now, run the search on the PDF index and wait for it to complete
        pdf_results = search_client_pdf.search(search_text=None, vector_queries=[vector_query_pdf])
        
        # Process results from the PDF index
        for result in pdf_results:
            combined_results.append({
                "score": result['@search.score'],
                "content": result['chunk']
            })
            
        sorted_results = sorted(combined_results, key=lambda x: x['score'], reverse=True)
        top_results = [res for res in sorted_results if res['score'] >= SCORE_THRESHOLD][:5]
        
        # If no results meet the threshold, return None
        if not top_results:
            return None
            
        context_for_llm = ""
        for i, result in enumerate(top_results):
            context_for_llm += f"Content: {result['content']}\n\n"

    except Exception as e:
        print(f"Error during search: {e}")
        return None

    return context_for_llm