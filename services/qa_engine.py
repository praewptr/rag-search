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


def get_response(text: str, client: AzureOpenAI):
    """
    Uses Azure OpenAI with Azure Cognitive Search extensions to answer a question
    using retrieved documents automatically — no document IDs needed.
    """
    response = client.chat.completions.create(
        model=azure_oai_deployment,
        temperature=0.5,
        max_tokens=1000,
        messages=[
            {
                "role": "system",
                "content": (
                    "คุณเป็นผู้ช่วยที่ให้ข้อมูลเป็นภาษาไทยโดยอิงจากบริบทที่ให้มาเท่านั้น "
                    "กรุณาตอบคำถามอย่างสุภาพและชัดเจน โดยจัดรูปแบบคำตอบให้อ่านง่าย เช่น:\n"
                    "- ใช้หัวข้อหรือหัวข้อย่อยในการแบ่งหมวดหมู่ข้อมูล\n"
                    "- ใช้ bullet points (เช่น - หรือ •) เพื่อแจกแจงรายละเอียด\n"
                    "- เว้นบรรทัดระหว่างหัวข้อเพื่อให้อ่านง่าย\n"
                    "- หลีกเลี่ยงการตอบเป็นข้อความยาวติดกัน\n\n"
                    " - หากมีการลำดับขั้นตอนหรือจำนวนข้อ เช่น 1., 2., 3..\n"
                    "ห้ามใส่ citation markers เช่น [doc1], [doc2] หรือ URL ใด ๆ ในคำตอบ\n"
                    "หากไม่สามารถตอบได้จากบริบท ให้ตอบว่า 'ไม่ทราบข้อมูลเพียงพอที่จะตอบคำถามนี้ได้'\n"
                ),
            },
            {
                "role": "user",
                "content": text,
            },
        ],
        extra_body=extension_config,  # Uses Azure Cognitive Search to get context
    )

    answer = response.choices[0].message.content
    citations = []

    # ✅ Optionally extract citations if available
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
