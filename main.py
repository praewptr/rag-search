from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from services.utils import remove_citation_markers
from services.qa_engine import get_response
from models.rag_search import QuestionRequest
import sys
import socket
import os
from dotenv import load_dotenv
from openai import AzureOpenAI


project_dir = os.path.dirname(os.path.abspath(__file__))
if project_dir not in sys.path:
    sys.path.append(project_dir)


# ------- Config and Environment Variables -------#
env_path = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(env_path)

azure_oai_endpoint = os.getenv("AZURE_OAI_ENDPOINT")
azure_oai_key = os.getenv("AZURE_OAI_KEY")
azure_oai_deployment = os.getenv("AZURE_OAI_DEPLOYMENT")
azure_search_endpoint = os.getenv("AZURE_SEARCH_ENDPOINT")
azure_search_key = os.getenv("AZURE_SEARCH_KEY")
azure_search_index = os.getenv("AZURE_SEARCH_INDEX")
# -----------------------------------------------#

# --------- Initialize Azure OpenAI client---------#
azure_openai_client = AzureOpenAI(
    base_url=f"{azure_oai_endpoint}/openai/deployments/{azure_oai_deployment}/extensions",
    api_key=azure_oai_key,
    api_version="2023-09-01-preview",
)
# --------------------------------------------------#

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
# -----------------------------------------------------------------
app = FastAPI(title="RAG PDF Question Answering API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/aisearch")
def ask_question(request: QuestionRequest):
    try:

        answer, citations = get_response(
            request.question,
            azure_openai_client,
            azure_oai_deployment,
            azure_oai_endpoint,
            extension_config,
        )

        cleaned_answer = remove_citation_markers(answer)

        sources = []
        for c in citations:
            title = c.get("title") or c.get("filepath") or "Unknown Source"
            if title not in sources:
                sources.append(title)

        response_payload = {
            "answer": cleaned_answer,
            "sources": sources,
            # "citations": citations
        }
        return response_payload

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error generating answer: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn

    hostname = socket.gethostname()
    ip_address = socket.gethostbyname(hostname)
    print(f"Server hostname: {hostname}")
    print(f"Server IP address: {ip_address}")

    uvicorn.run(app, host="0.0.0.0", port=5099)
