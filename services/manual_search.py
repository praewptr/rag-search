from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import requests
import json
import os
from datetime import datetime
from models.rag_search import DocumentResult, SearchRequest, SearchResponse


from services.config import (
    azure_search_endpoint,
    azure_search_key,
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
