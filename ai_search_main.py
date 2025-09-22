from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from services.utils import remove_citation_markers

# from services.qa_engine import get_response
from services.client import azure_openai_client, chroma_client
from models.rag_search import QuestionRequest, SearchRequest, SearchResponse

from services.qa_engine import get_response_with_fallback, get_response
import os
from services.manual_search import RAGSearchService
from datetime import datetime


def ensure_chroma_db_exists(directory: str = "./chroma_db") -> None:
    """
    Ensure the chroma_db directory exists. If not, create it.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")
    else:
        print(f"Directory already exists: {directory}")


# Call the function to ensure chroma_db exists
ensure_chroma_db_exists()


app = FastAPI(title="RAG PDF Question Answering API", version="1.0.0")
search_service = RAGSearchService()
# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.post("/aisearch")
def ask_question(request: QuestionRequest):
    try:
        answer, citations = get_response(request.question, azure_openai_client)

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


@app.post("/aisearch_v2")
async def ask_question(request: QuestionRequest):
    """
    Ask a question and get an answer from ChromaDB or Azure Cognitive Search.
    """
    try:
        answer, citations = await get_response_with_fallback(
            request.question, azure_openai_client
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


@app.post("/search", response_model=SearchResponse)
async def search_documents(request: SearchRequest):
    """
    Search for PDF documents using natural language queries

    Returns the top K most relevant documents with scores and URLs
    """
    try:
        # Perform search
        documents, search_time = await search_service.search_documents(
            request.query, request.top_k
        )

        # Add URLs to documents
        document_results = search_service.add_urls_to_documents(documents)

        # Create response
        response = SearchResponse(
            query=request.query,
            total_results=len(document_results),
            documents=document_results,
            search_time_ms=search_time,
            timestamp=datetime.now().isoformat(),
        )

        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")
