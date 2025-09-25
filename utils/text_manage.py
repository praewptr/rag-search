import json
import os

from fastapi import HTTPException

from models.upload_txt import (
    DocumentItem,
)
from services.azure_upload import process_and_upload

MOCK_DB_FILE = "rag_text.json"


# Helper function for document validation
def validate_document(document: DocumentItem):
    if not document.content.strip():
        raise HTTPException(status_code=400, detail="Document content cannot be empty")
    if not document.source.strip():
        raise HTTPException(status_code=400, detail="Document source cannot be empty")


# Helper function for processing and uploading documents
def process_documents(documents):
    successful_uploads = 0
    failed_uploads = []

    for i, doc_item in enumerate(documents):
        try:
            validate_document(doc_item)

            json_data = {
                "content": doc_item.content,
                "source": doc_item.source,
                "timestamp": doc_item.timestamp,
            }

            result = process_and_upload(json_data)

            if result:
                successful_uploads += 1
            else:
                failed_uploads.append(
                    f"Document {i + 1} ({doc_item.source}): Upload failed"
                )

        except Exception as doc_error:
            failed_uploads.append(
                f"Document {i + 1} ({doc_item.source}): {str(doc_error)}"
            )

    return successful_uploads, failed_uploads


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
