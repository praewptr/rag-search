from services.config import (
    azure_search_endpoint,
    azure_search_key,
    azure_search_index,
    azure_oai_deployment,
    azure_oai_endpoint,
    azure_oai_key,
)
import json
from openai import AzureOpenAI
from services.vector_store import add_faq, find_similar_faq

# Configure your data source

extension_config = dict(
    dataSources=[
        {
            "type": "AzureCognitiveSearch",
            "parameters": {
                "endpoint": azure_search_endpoint,
                "key": azure_search_key,
                "indexName": azure_search_index,
            },
            "fields_mapping": {
                "content_field": "content",
                "filepath_field": "filepath",
                "title_field": "title",
                "url_field": "url",
                "vector_field": "contentVector",
            },
        }
    ]
)


def get_response(text: str, client: AzureOpenAI):
    """
    Uses Azure OpenAI with Azure Cognitive Search extensions to answer a question
    using retrieved documents automatically — no document IDs needed.
    """
    response = client.chat.completions.create(
        model=azure_oai_deployment,
        temperature=0.3,  # ลดลงเล็กน้อยเพื่อให้ภาษาสม่ำเสมอแต่ยังยืดหยุ่น
        max_tokens=600,
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
                    "- ⚠️ ONLY use information from provided documents\n"
                    "- ⚠️ NO citation markers [doc1], [source], URLs\n"
                    "- ⚠️ NO assumptions beyond document content\n\n"
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


def get_threshold_by_length(text, thresholds=None):
    if thresholds is None:
        thresholds = {
            (0, 5): 0.45,  # ลดค่าสำหรับคำสั้นๆ
            (6, 15): 0.65,  # ลดค่าสำหรับคำปานกลาง
            (16, 1000): 0.75,  # ลดค่าสำหรับคำยาว
        }
    length = len(text.split())
    for (min_len, max_len), thr in thresholds.items():
        if min_len <= length <= max_len:
            return thr
    return 0.65


async def get_response_with_fallback(text: str, client: AzureOpenAI) -> str:
    """
    Get an answer by searching ChromaDB first, then Azure Cognitive Search if not found.
    """
    # Step 1: Search in ChromaDB
    answer = find_similar_faq(text)
    if answer:
        print("finding similar answer from ChromaDB")
        return answer, []

    # Step 2: Search in Azure Cognitive Search
    response = client.chat.completions.create(
        model=azure_oai_deployment,
        temperature=0.3,
        max_tokens=500,
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
                    "- ⚠️ ONLY use information from provided documents\n"
                    "- ⚠️ NO citation markers [doc1], [source], URLs\n"
                    "- ⚠️ NO assumptions beyond document content\n\n"
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

                Question: {text}

                Provide a detailed answer with clear formatting.
                """,
            },
        ],
        extra_body=extension_config,
    )
    print("getting answer from Azure Cognitive Search")

    answer = response.choices[0].message.content

    try:
        context_messages = response.choices[0].message.context.get("messages", [])
        if context_messages:
            citation_json = json.loads(context_messages[0].get("content", "{}"))
            if "citations" in citation_json:
                citations = citation_json["citations"]
    except Exception:
        # No citations or context — skip silently
        pass

    # Step 3: Store the answer in ChromaDB
    if answer and (
        "ขออภัย ไม่มีข้อมูลเพียงพอในเอกสารสำหรับคำถามนี้" not in answer
        and "I couldn’t find that information" not in answer
    ):
        add_faq(text, answer)

    return answer, citations
