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


# 1. Chunking
def chunk_content(content: str, source: str, timestamp: str = None) -> List[Document]:
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


# 2. Upload to Azure Search
def upload_to_azure_search(documents: List[Document]):
    # Initialize search client for text index
    search_client = SearchClient(
        endpoint=azure_search_endpoint,
        index_name=azure_search_index_txt,
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
        print(f"✅ Successfully uploaded {len(azure_docs)} documents to Azure Search")
        return result
    except Exception as e:
        print(f"❌ Error uploading to Azure Search: {e}")
        raise e


# 3. Orchestration function
def process_and_upload(data: Dict):
    if "content" not in data or "source" not in data:
        raise ValueError("JSON data must include 'content' and 'source' fields.")

    documents = chunk_content(
        content=data["content"],
        source=data["source"],
        timestamp=data.get("timestamp"),
    )

    result = upload_to_azure_search(documents)
    return result


def process_and_upload_batch(documents: List[Dict]):
    all_chunks = []
    for data in documents:
        if "content" not in data or "source" not in data:
            raise ValueError("Each document must include 'content' and 'source'.")

        # Use provided 'id' or generate new UUIDs for chunk_id
        doc_chunks = chunk_content(
            content=data["content"],
            source=data["source"],
            timestamp=data.get("timestamp"),
        )
        # Overwrite chunk metadata 'id' with provided id if available
        if "id" in data:
            for chunk in doc_chunks:
                chunk.metadata["id"] = str(data["id"])
        all_chunks.extend(doc_chunks)

    return upload_to_azure_search(all_chunks)


if __name__ == "__main__":
    sample_data = {
        "content": "This is a sample document content to be chunked and uploaded to Azure Search.",
        "source": "sample.txt",
        "timestamp": "2023-10-01T12:00:00Z",
    }
    process_and_upload(sample_data)
