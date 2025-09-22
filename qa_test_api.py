"""
FastAPI RAG Search API
======================

A FastAPI application for RAG PDF Question Answering System
that provides search endpoints to find top 3 documents with highest scores.

Author: RAG System Team
Date: September 2025
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import requests
import json
import os
from datetime import datetime

from services.config import (
    azure_search_endpoint,
    azure_search_key,
)


# Pydantic Models
class SearchRequest(BaseModel):
    query: str = Field(
        ..., description="Search query text", min_length=1, max_length=500
    )
    top_k: int = Field(
        default=3, description="Number of top results to return", ge=1, le=10
    )


class DocumentResult(BaseModel):
    title: str
    score: float
    url: Optional[str] = None
    id: Optional[str] = None
    rank: int


class SearchResponse(BaseModel):
    query: str
    total_results: int
    documents: List[DocumentResult]
    search_time_ms: float
    timestamp: str


class HealthResponse(BaseModel):
    status: str
    message: str
    timestamp: str
    configuration: Dict[str, bool]


# Initialize FastAPI app
app = FastAPI(
    title="RAG PDF Search API",
    description="Search for PDF documents using Azure Search with relevance scoring",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class RAGSearchService:
    """Service class for RAG search functionality"""

    def __init__(self):
        self.azure_search_endpoint = azure_search_endpoint
        self.azure_search_key = azure_search_key
        self.pdf_map_path = "pdf_map.json"
        self.pdf_map = self._load_pdf_map()

    def _load_pdf_map(self) -> List[Dict[str, str]]:
        """Load PDF mapping from JSON file"""
        try:
            with open(self.pdf_map_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"⚠️ Warning: {self.pdf_map_path} not found. Creating empty mapping.")
            return []
        except json.JSONDecodeError as e:
            print(f"❌ Error parsing PDF map: {e}")
            return []

    async def search_documents(
        self, query_text: str, top_k: int = 3
    ) -> tuple[List[Dict[str, Any]], float]:
        """
        Search for documents using Azure Search

        Returns:
            Tuple of (documents_list, search_time_ms)
        """
        import time

        start_time = time.time()

        # Create payload for vector search
        payload = {
            "count": True,
            "select": "title",
            "top": top_k,
            "vectorQueries": [
                {"kind": "text", "text": query_text, "fields": "text_vector"}
            ],
        }

        # Send request to Azure Search
        url = f"{self.azure_search_endpoint}/indexes/rag-manual/docs/search?api-version=2023-10-01-Preview"
        headers = {"Content-Type": "application/json", "api-key": self.azure_search_key}

        try:
            response = requests.post(
                url, headers=headers, data=json.dumps(payload), timeout=10
            )

            search_time = (time.time() - start_time) * 1000  # Convert to milliseconds

            if response.status_code == 200:
                results = response.json()
                documents = []
                for doc in results.get("value", []):
                    documents.append(
                        {"title": doc["title"], "score": doc.get("@search.score", 0.0)}
                    )
                # Sort by score descending
                documents.sort(key=lambda x: x["score"], reverse=True)
                return documents, search_time
            else:
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"Azure Search failed: {response.text}",
                )

        except requests.RequestException as e:
            raise HTTPException(
                status_code=500,
                detail=f"Network error connecting to Azure Search: {str(e)}",
            )

    def add_urls_to_documents(
        self, documents: List[Dict[str, Any]]
    ) -> List[DocumentResult]:
        """Add URLs to documents from PDF mapping"""
        results = []
        for i, doc in enumerate(documents, 1):
            title = doc["title"]
            score = doc["score"]
            match = next(
                (item for item in self.pdf_map if item["title"] == title), None
            )

            result = DocumentResult(
                title=title,
                score=score,
                rank=i,
                url=match["url"] if match else None,
                id=match.get("id") if match else None,
            )
            results.append(result)

        return results


# Initialize the service
search_service = RAGSearchService()


@app.get("/", response_class=HTMLResponse)
async def root():
    """Root endpoint with API documentation"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>RAG PDF Search API</title>
        <style>
            body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 40px; background: #f5f7fa; }
            .container { max-width: 800px; margin: 0 auto; background: white; padding: 40px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
            h1 { color: #2d3748; text-align: center; margin-bottom: 20px; }
            .endpoint { background: #f8f9fa; padding: 20px; margin: 20px 0; border-radius: 8px; border-left: 4px solid #3182ce; }
            .method { display: inline-block; padding: 4px 12px; border-radius: 4px; font-weight: bold; margin-right: 10px; }
            .post { background: #48bb78; color: white; }
            .get { background: #3182ce; color: white; }
            code { background: #edf2f7; padding: 2px 6px; border-radius: 4px; font-family: 'Courier New', monospace; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🚀 RAG PDF Search API</h1>
            <p>Welcome to the RAG PDF Search API. This API allows you to search for PDF documents and get the top 3 most relevant results with scores.</p>
            
            <div class="endpoint">
                <span class="method post">POST</span>
                <strong>/search</strong> - Search for documents
                <p>Search for PDF documents using natural language queries. Returns top 3 results with relevance scores and URLs.</p>
            </div>
            
            <div class="endpoint">
                <span class="method get">GET</span>
                <strong>/search/{query}</strong> - Quick search
                <p>Perform a quick search using URL parameters. Example: <code>/search/manual?top_k=3</code></p>
            </div>
            
            <div class="endpoint">
                <span class="method get">GET</span>
                <strong>/health</strong> - Health check
                <p>Check API health and configuration status.</p>
            </div>
            
            <div class="endpoint">
                <span class="method get">GET</span>
                <strong>/documents</strong> - List all documents
                <p>Get a list of all available PDF documents in the mapping.</p>
            </div>
            
            <p><strong>📚 Documentation:</strong></p>
            <ul>
                <li><a href="/docs">Interactive API Docs (Swagger)</a></li>
                <li><a href="/redoc">Alternative API Docs (ReDoc)</a></li>
            </ul>
        </div>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)


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


@app.get("/search/{query}", response_model=SearchResponse)
async def quick_search(
    query: str,
    top_k: int = Query(
        default=3, ge=1, le=10, description="Number of results to return"
    ),
):
    """
    Quick search endpoint using URL parameters

    Example: GET /search/manual?top_k=3
    """
    request = SearchRequest(query=query, top_k=top_k)
    return await search_documents(request)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""

    config_status = {
        "azure_search_configured": bool(
            search_service.azure_search_endpoint and search_service.azure_search_key
        ),
        "pdf_mapping_loaded": len(search_service.pdf_map) > 0,
        "pdf_mapping_count": len(search_service.pdf_map),
    }

    overall_status = (
        "healthy"
        if all(
            [
                config_status["azure_search_configured"],
                config_status["pdf_mapping_loaded"],
            ]
        )
        else "degraded"
    )

    return HealthResponse(
        status=overall_status,
        message="RAG Search API is running",
        timestamp=datetime.now().isoformat(),
        configuration=config_status,
    )


@app.get("/documents")
async def list_documents():
    """List all available PDF documents"""
    return {
        "total_documents": len(search_service.pdf_map),
        "documents": search_service.pdf_map,
        "timestamp": datetime.now().isoformat(),
    }


@app.get("/stats")
async def get_stats():
    """Get API statistics"""
    mapped_urls = sum(1 for doc in search_service.pdf_map if doc.get("url"))

    return {
        "total_documents": len(search_service.pdf_map),
        "documents_with_urls": mapped_urls,
        "documents_without_urls": len(search_service.pdf_map) - mapped_urls,
        "coverage_percentage": (
            (mapped_urls / len(search_service.pdf_map) * 100)
            if search_service.pdf_map
            else 0
        ),
        "azure_search_configured": bool(
            search_service.azure_search_endpoint and search_service.azure_search_key
        ),
        "timestamp": datetime.now().isoformat(),
    }


# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return {
        "error": "Endpoint not found",
        "message": f"The endpoint {request.url.path} was not found",
        "available_endpoints": [
            "/search (POST)",
            "/search/{query} (GET)",
            "/health (GET)",
            "/documents (GET)",
            "/stats (GET)",
            "/docs (GET)",
            "/redoc (GET)",
        ],
    }


if __name__ == "__main__":
    import uvicorn

    print("🚀 Starting RAG PDF Search API...")
    print("📚 Available endpoints:")
    print("  • POST /search - Search for documents")
    print("  • GET /search/{query} - Quick search")
    print("  • GET /health - Health check")
    print("  • GET /documents - List documents")
    print("  • GET /stats - API statistics")
    print("  • GET /docs - Interactive docs")
    print("  • GET /redoc - Alternative docs")
    print("\n🌐 Starting server on http://localhost:8000")

    uvicorn.run(
        "qa_test_api:app", host="0.0.0.0", port=8000, reload=True, log_level="info"
    )
