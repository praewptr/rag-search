import json
import os
from datetime import datetime
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from services.azure_upload import process_and_upload

app = FastAPI(title="RAG Data Manager API")

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

# Mock database file
MOCK_DB_FILE = "rag_text.json"


# Pydantic models
class DocumentItem(BaseModel):
    content: str
    source: str
    timestamp: str
    id: Optional[int] = None


class DocumentResponse(BaseModel):
    value: List[DocumentItem]


class UploadRequest(BaseModel):
    value: List[DocumentItem]


class UploadResponse(BaseModel):
    message: str
    uploaded_count: int
    status: str


# Utility functions
def load_mock_data() -> dict:
    """Load data from the mock database file."""
    try:
        if not os.path.exists(MOCK_DB_FILE):
            return {"value": []}

        with open(MOCK_DB_FILE, "r", encoding="utf-8") as file:
            data = json.load(file)

        # Ensure the data has the correct structure
        if isinstance(data, list):
            # If it's a list, wrap it in the expected structure
            data = {"value": data}
        elif not isinstance(data, dict) or "value" not in data:
            # If it's not the expected structure, create empty
            data = {"value": []}

        # Add IDs if they don't exist
        for i, item in enumerate(data["value"]):
            if "id" not in item or item["id"] is None:
                item["id"] = i + 1

        return data
    except Exception as e:
        print(f"Error loading mock data: {e}")
        return {"value": []}


def save_mock_data(data: dict):
    """Save data to the mock database file."""
    try:
        with open(MOCK_DB_FILE, "w", encoding="utf-8") as file:
            json.dump(data, file, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"Error saving mock data: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to save data: {str(e)}")


# API Endpoints


@app.get("/")
async def serve_data_manager():
    """Serve the data manager UI."""
    return FileResponse("static/data_manager.html")


@app.get("/api/documents", response_model=DocumentResponse)
async def get_documents():
    """
    Retrieve all documents from the mock database.
    """
    try:
        data = load_mock_data()
        return data
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to load documents: {str(e)}"
        )


@app.delete("/api/documents/{doc_id}")
async def delete_document(doc_id: int):
    """
    Delete a document by its ID from the mock database.
    """
    try:
        data = load_mock_data()
        documents = data["value"]

        # Find and remove the document with the specified ID
        original_count = len(documents)
        data["value"] = [doc for doc in documents if doc.get("id") != doc_id]

        if len(data["value"]) == original_count:
            raise HTTPException(status_code=404, detail="Document not found")

        save_mock_data(data)
        return {"message": f"Document with ID {doc_id} deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to delete document: {str(e)}"
        )


@app.post("/api/upload-to-azure", response_model=UploadResponse)
async def upload_to_azure(request: UploadRequest):
    """
    Upload selected documents to Azure AI Search using existing infrastructure.
    """
    try:
        selected_docs = request.value

        if not selected_docs:
            raise HTTPException(
                status_code=400, detail="No documents selected for upload"
            )

        uploaded_count = 0
        error_count = 0
        errors = []

        print(f"🔄 Processing {len(selected_docs)} documents for Azure upload...")

        for i, doc in enumerate(selected_docs, 1):
            try:
                print(f"📄 Processing document {i}: {doc.source}")

                # Use existing process_and_upload function
                result = process_and_upload(
                    {
                        "content": doc.content,
                        "source": doc.source,
                        "timestamp": doc.timestamp,
                    }
                )

                if result:
                    uploaded_count += 1
                    print(f"✅ Successfully uploaded document {i}")
                else:
                    error_count += 1
                    errors.append(f"Document {i} ({doc.source}): Upload failed")

            except Exception as e:
                error_count += 1
                error_msg = f"Document {i} ({doc.source}): {str(e)}"
                errors.append(error_msg)
                print(f"❌ Error processing document {i}: {e}")

        # Prepare response
        if error_count == 0:
            message = f"🎉 Successfully uploaded all {uploaded_count} documents to Azure AI Search!"
            status = "success"
        elif uploaded_count > 0:
            message = f"⚠️ Uploaded {uploaded_count} documents, {error_count} failed. Errors: {'; '.join(errors[:3])}"
            status = "partial_success"
        else:
            message = (
                f"❌ Failed to upload any documents. Errors: {'; '.join(errors[:3])}"
            )
            status = "error"

        print(f"📊 Final result: {uploaded_count} uploaded, {error_count} failed")

        return UploadResponse(
            message=message, uploaded_count=uploaded_count, status=status
        )

    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Unexpected error in upload_to_azure: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to upload documents: {str(e)}"
        )


@app.post("/api/add-document")
async def add_document(document: DocumentItem):
    """
    Add a new document to the mock database.
    """
    try:
        data = load_mock_data()

        # Generate new ID
        max_id = max([doc.get("id", 0) for doc in data["value"]], default=0)
        document.id = max_id + 1

        # Add timestamp if not provided
        if not document.timestamp:
            document.timestamp = datetime.utcnow().isoformat() + "Z"

        data["value"].append(document.dict())
        save_mock_data(data)

        return {"message": "Document added successfully", "id": document.id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to add document: {str(e)}")


@app.get("/api/export-json")
async def export_selected_json(doc_ids: str = ""):
    """
    Export selected documents as JSON format for download.
    """
    try:
        data = load_mock_data()

        if doc_ids:
            # Parse comma-separated IDs
            selected_ids = [int(x.strip()) for x in doc_ids.split(",") if x.strip()]
            selected_docs = [
                doc for doc in data["value"] if doc.get("id") in selected_ids
            ]
        else:
            # Export all documents if no IDs specified
            selected_docs = data["value"]

        # Format for Azure upload
        export_data = {
            "value": [
                {
                    "content": doc["content"],
                    "source": doc["source"],
                    "timestamp": doc["timestamp"],
                }
                for doc in selected_docs
            ]
        }

        return export_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to export JSON: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    print("🚀 Starting RAG Data Manager...")
    print("📋 Data Manager UI: http://localhost:8000")
    print("📚 API docs: http://localhost:8000/docs")
    uvicorn.run(app, host="127.0.0.1", port=8000)
