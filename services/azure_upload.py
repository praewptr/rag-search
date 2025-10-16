import logging
import uuid
from typing import Dict, List

from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

from config import (
    azure_search_endpoint,
    azure_search_index_txt,
    azure_search_key,
)
from services.client import embeddings

logger = logging.getLogger(__name__)


# 1. Chunking
def chunk_content(content: str, source: str, timestamp: str = None) -> List[Document]:
    try:
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_text(content)
        documents = []
        for i, chunk in enumerate(chunks):
            documents.append(
                Document(
                    page_content=chunk,
                    metadata={
                        "id": str(uuid.uuid4()),
                        "chunk_id": str(uuid.uuid4()),
                        "source": source,
                        "timestamp": timestamp,
                    },
                )
            )
        return documents
    except Exception as e:
        logger.error(f"Error during chunking content from source {source}: {e}")
        raise e


# 2. Upload to Azure Search
def upload_to_azure_search(documents: List[Document], index_name: str = None):
    # Use provided index name or fall back to default
    target_index = index_name or azure_search_index_txt

    # Initialize search client for specified index
    search_client = SearchClient(
        endpoint=azure_search_endpoint,
        index_name=target_index,
        credential=AzureKeyCredential(azure_search_key),
    )

    # Prepare documents for Azure upload
    azure_docs = []
    for doc in documents:
        vector = embeddings.embed_query(doc.page_content)
        azure_docs.append(
            {
                "@search.action": "upload",
                "id": doc.metadata["id"],
                "content": doc.page_content,
                "source": doc.metadata["source"],
                "timestamp": doc.metadata["timestamp"],
                "chunk_id": str(doc.metadata["chunk_id"]),
                "text_vector": vector,
            }
        )

    # Upload in batches
    try:
        result = search_client.upload_documents(documents=azure_docs)
        logger.info(
            f"Successfully uploaded {len(azure_docs)} documents to Azure Search index '{target_index}'"
        )

        # Log any failures in detail
        for r in result:
            if not r.succeeded:
                logger.error(f"Failed to upload document ID {r.key}: {r.error_message}")

        return result
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Error uploading to Azure Search: {e}")
        if "not found" in error_msg.lower() or "404" in error_msg:
            logger.error(f"Index '{target_index}' not found in Azure Search.")
            raise Exception(
                f"Index '{target_index}' not found in Azure Search. Please verify the index exists."
            )
        elif "bad request" in error_msg.lower() or "400" in error_msg:
            logger.error(f"Bad request for index '{target_index}': {error_msg}")
            raise Exception(
                f"Index '{target_index}' field structure mismatch. Expected fields: id, content, source, timestamp, chunk_id, text_vector. Original error: {error_msg}"
            )
        elif "unauthorized" in error_msg.lower() or "401" in error_msg:
            logger.error(f"Authentication failed for Azure Search: {error_msg}")
            raise Exception(
                f"Authentication failed for Azure Search. Please check API key. Original error: {error_msg}"
            )
        elif "forbidden" in error_msg.lower() or "403" in error_msg:
            logger.error(f"Access denied to index '{target_index}': {error_msg}")
            raise Exception(
                f"Access denied to index '{target_index}'. Please check API key permissions. Original error: {error_msg}"
            )
        else:
            logger.error(
                f"Failed to upload documents to index '{target_index}': {error_msg}"
            )
            raise Exception(
                f"Upload to index '{target_index}' failed. This may indicate field structure incompatibility or configuration issues. Original error: {error_msg}"
            )


# 3. Orchestration function
def process_and_upload(data: Dict, index_name: str = None):
    if "content" not in data or "source" not in data:
        raise ValueError("JSON data must include 'content' and 'source' fields.")

    documents = chunk_content(
        content=data["content"],
        source=data["source"],
        timestamp=data.get("timestamp"),
    )

    result = upload_to_azure_search(documents, index_name)
    return result


def process_and_upload_batch(documents: List[Dict], index_name: str = None):
    all_chunks = []
    for data in documents:
        # Handle both Oracle format and standardized format
        content = data.get("content") or data.get("CONTENT") or data.get("Content")
        source = (
            data.get("source")
            or data.get("USER_NAME")
            or data.get("user_name")
            or data.get("SOURCE")
        )
        timestamp = (
            data.get("timestamp") or data.get("CREATED_DATE") or data.get("TIMESTAMP")
        )

        if not content:
            raise ValueError(
                f"Document missing content field. Available fields: {list(data.keys())}"
            )
        if not source:
            raise ValueError(
                f"Document missing source field. Available fields: {list(data.keys())}"
            )

        # Use provided 'id' for tracking but keep unique chunk IDs
        doc_chunks = chunk_content(
            content=content,
            source=source,
            timestamp=timestamp,
        )
        # Add source document reference in metadata for tracking
        doc_id = data.get("id") or data.get("ID")
        if doc_id:
            for chunk in doc_chunks:
                # Store original document ID in source for tracking
                original_source = chunk.metadata.get("source", "")
                chunk.metadata["source"] = f"{original_source} (Doc ID: {doc_id})"
        all_chunks.extend(doc_chunks)

    return upload_to_azure_search(all_chunks, index_name)


if __name__ == "__main__":
    sample_data = {
        "content": "This is a sample document content to be chunked and uploaded to Azure Search.",
        "source": "sample.txt",
        "timestamp": "2023-10-01T12:00:00Z",
    }
    process_and_upload(sample_data)
