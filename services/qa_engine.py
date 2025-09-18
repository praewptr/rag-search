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


GROUNDED_PROMPT = """
You are a helpful assistant answering questions based strictly on the documents provided.

- Use only the information from the sources below to answer the query.
- Respond in clear, concise bullet points.
- Do NOT include any information that is not explicitly stated in the sources.
- If the answer is not available in the provided sources, reply with: "I couldn’t find that information in the documents provided."

Question:
{text}

"""


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
