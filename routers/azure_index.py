import logging
import os
from typing import Optional, Union

from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex,
)
from dotenv import load_dotenv
from fastapi import APIRouter, HTTPException, Path, Query

from config import (
    azure_search_endpoint,
    azure_search_index_txt,
    azure_search_key,
)
from models.azure_index import (
    CreateIndexRequest,
    CreateIndexResponse,
    DeleteByTitleResponse,
    DeleteDocumentResponse,
    DeleteIndexResponse,
    IndexInfo,
    ListIndexesResponse,
)
from models.upload import (
    IndexStats,
    SearchResponse,
    SearchResult,
    TitleListResponse,
)
from utils.azure_index import (
    count_documents,
    create_search_fields,
    create_vector_search_config,
    get_index_client,
    get_search_client,
    get_select_field,
    handle_index_error,
    paginate_results,
)

router = APIRouter()
logger = logging.getLogger(__name__)


env_path = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(env_path)


@router.post("/create-index", response_model=CreateIndexResponse)
async def create_index(request: CreateIndexRequest):
    """
    Create a new Azure Search index with standard RAG schema.

    Args:
        request: CreateIndexRequest containing index name and options

    Returns:
        CreateIndexResponse: Result of index creation
    """
    try:
        index_client = get_index_client()
        index_name = request.name

        logger.info(f"Creating index: {index_name}")

        # Check if index already exists
        try:
            existing_index = index_client.get_index(index_name)
            if existing_index and not request.force_recreate:
                return CreateIndexResponse(
                    status="exists",
                    message=f"Index '{index_name}' already exists. Use force_recreate=true to recreate it.",
                    index_name=index_name,
                    created=False,
                )
            elif existing_index and request.force_recreate:
                # Delete existing index
                logger.info(f"Deleting existing index: {index_name}")
                index_client.delete_index(index_name)
                logger.info(f"Successfully deleted existing index: {index_name}")

        except Exception:
            logger.info(f"Index '{index_name}' doesn't exist, proceeding with creation")

        # Define the index schema
        fields = create_search_fields()
        vector_search = create_vector_search_config()

        # Create the index
        search_index = SearchIndex(
            name=index_name,
            fields=fields,
            similarity={"@odata.type": "#Microsoft.Azure.Search.BM25Similarity"},
            vector_search=vector_search,
        )

        index_client.create_index(search_index)

        logger.info(f"Successfully created index: {index_name}")

        return CreateIndexResponse(
            status="success",
            message=f"Index '{index_name}' created successfully",
            index_name=index_name,
            created=True,
        )

    except Exception as e:
        logger.error(f"An error occurred in the azure-index/create_index endpoint: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to create index '{request.name}': {str(e)}"
        )


@router.get("/indexes", response_model=ListIndexesResponse)
async def list_available_indexes():
    """
    Get list of available Azure Search indexes.
    """
    try:
        index_client = get_index_client()
        indexes = list(index_client.list_indexes())
        available_indexes = [IndexInfo(name=index.name) for index in indexes]
        return ListIndexesResponse(
            status="success",
            indexes=available_indexes,
            default_index=azure_search_index_txt,
        )
    except Exception as e:
        logger.error(
            f"An error occurred in the azure-index/list_indexes endpoint: {str(e)}"
        )
        raise HTTPException(status_code=500, detail=f"Failed to list indexes: {str(e)}")


@router.delete("/index/{index_name}", response_model=DeleteIndexResponse)
async def delete_index(
    index_name: str = Path(..., description="Name of the Azure Search index to delete"),
):
    """
    Delete an Azure Search index.
    """
    try:
        await handle_index_error(index_name, "deletion")
        index_client = get_index_client()
        logger.info(f"Deleting index: {index_name}")
        index_client.delete_index(index_name)
        logger.info(f"Successfully deleted index: {index_name}")
        return DeleteIndexResponse(
            status="success",
            message=f"Index '{index_name}' deleted successfully",
            index_name=index_name,
        )

    except Exception as e:
        logger.error(f"An error occurred in the azure-index/delete_index endpoint: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to delete index '{index_name}': {str(e)}"
        )


@router.get("/index-stats", response_model=IndexStats)
async def get_index_statistics(
    index_name: Optional[str] = Query(None, description="Index name to get stats for"),
):
    """
    Get basic statistics about the Azure Search text index.
    """
    try:
        search_client = get_search_client(index_name)
        selected_index = index_name or azure_search_index_txt

        try:
            index_client_instance = SearchIndexClient(
                endpoint=azure_search_endpoint,
                credential=AzureKeyCredential(azure_search_key),
            )
            index = index_client_instance.get_index(name=selected_index)
            field_names = [field.name for field in index.fields]
            logger.info(f"Available fields in index '{selected_index}': {field_names}")

            select_field = get_select_field(index.fields)
            logger.info(f"Using field '{select_field}' for counting documents")

        except Exception as e:
            logger.warning(f"Could not get index schema: {e}")
            select_field = None

        document_count = count_documents(search_client, select_field)
        logger.info(f"Total document count: {document_count}")

        return IndexStats(
            total_documents=document_count,
            index_name=selected_index,
            status="success",
        )

    except Exception as e:
        logger.error(f"An error occurred in the azure-index/index-stats endpoint: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get index statistics: {str(e)}"
        )


@router.get("/browse", response_model=Union[SearchResponse, TitleListResponse])
async def browse_all_documents(
    page: int = Query(1, description="Page number", ge=1),
    page_size: int = Query(20, description="Documents per page", le=50),
    index_name: Optional[str] = Query(None, description="Index name to browse"),
):
    """
    Browse all documents in the Azure Search text index with pagination.
    Uses different method if index has 'title' field - returns list of unique titles.
    """
    try:
        search_client = get_search_client(index_name)
        selected_index = index_name or azure_search_index_txt
        logger.info(
            f"Browsing index '{selected_index}' - Page {page}, Size {page_size}"
        )

        # Check if index has title field
        has_title_field = False
        try:
            index_client_instance = get_index_client()
            index = index_client_instance.get_index(name=selected_index)
            field_names = [field.name for field in index.fields]
            has_title_field = "title" in field_names
            logger.info(f"Index has title field: {has_title_field}")
        except Exception as e:
            logger.warning(f"Could not check index schema: {e}")
            has_title_field = False

        if has_title_field:
            logger.info("Using title-based method - returning unique document titles")
            search_results = search_client.search(
                search_text="*", select=["title"], include_total_count=True
            )
            unique_titles = {
                doc.get("title") for doc in search_results if doc.get("title")
            }
            title_list = sorted(unique_titles)
            paginated_titles = await paginate_results(title_list, page, page_size)
            return TitleListResponse(
                total_count=len(title_list),
                titles=paginated_titles,
                query="*",
                status="success",
            )
        else:
            logger.info("Using standard method for document retrieval")
            skip = (page - 1) * page_size
            results = search_client.search(
                search_text="*",
                top=page_size,
                skip=skip,
                include_total_count=True,
            )
            search_results = [
                SearchResult(
                    id=result.get("id", ""),
                    content=result.get("content", ""),
                    source=result.get("source", ""),
                    timestamp=result.get("timestamp", ""),
                    chunk_id=result.get("chunk_id"),
                    score=result.get("@search.score"),
                )
                for result in results
            ]
            total_count = results.get_count() or len(search_results)
            return SearchResponse(
                total_count=total_count,
                results=search_results,
                query="*",
                status="success",
            )
    except Exception as e:
        logger.error(f"An error occurred in the azure-index/browse endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to browse index: {str(e)}")


@router.delete("/documents/{doc_id}", response_model=DeleteDocumentResponse)
async def delete_document_from_index(
    doc_id: str = Path(..., description="ID of the document to delete"),
    index_name: Optional[str] = Query(None, description="Index name to delete from"),
):
    """
    Delete a specific document by its ID from the Azure Search index.
    """
    try:
        search_client = get_search_client(index_name)
        selected_index = index_name or azure_search_index_txt
        logger.info(f"Deleting document with ID: {doc_id} from index: {selected_index}")
        check_results = search_client.search(
            search_text="*", filter=f"id eq '{doc_id}'", select=["id"], top=1
        )
        if not list(check_results):
            raise HTTPException(
                status_code=404, detail=f"Document with ID '{doc_id}' not found"
            )
        search_client.delete_documents([{"id": doc_id}])
        logger.info(f"Successfully deleted document with ID: {doc_id}")
        return DeleteDocumentResponse(
            message=f"Successfully deleted document with ID: {doc_id}",
            deleted_id=doc_id,
            status="success",
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"An error occurred in the azure-index/document endpoint: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to delete document: {str(e)}"
        )


@router.delete("/delete-documents-by-title/", response_model=DeleteByTitleResponse)
async def delete_documents_by_title(
    index_name: str = Path(..., description="Index name to delete from"),
    title: str = Query(..., description="Title of the document to delete"),
):
    try:
        index_client = SearchIndexClient(
            endpoint=azure_search_endpoint,
            credential=AzureKeyCredential(azure_search_key),
        )
        index_schema = index_client.get_index(index_name)
        field_names = [f.name for f in index_schema.fields]
        has_image_vector = "image_vector" in field_names
        search_client = SearchClient(
            endpoint=azure_search_endpoint,
            index_name=index_name,
            credential=AzureKeyCredential(azure_search_key),
        )
        if has_image_vector:
            select_fields = ["text_parent_id", "title", "image_parent_id"]
        else:
            select_fields = ["parent_id", "title"]
        results = search_client.search(
            search_text=title,
            select=select_fields,
            top=100,
        )
        parent_ids = set()
        if has_image_vector:
            for doc in results:
                if doc.get("title") == title and doc.get("text_parent_id"):
                    parent_ids.add(doc["text_parent_id"])
        else:
            for doc in results:
                if doc.get("title") == title and doc.get("parent_id"):
                    parent_ids.add(doc["parent_id"])
        if not parent_ids:
            raise HTTPException(
                status_code=404, detail=f"No documents found with title '{title}'"
            )
        total_deleted = 0
        documents_to_delete = []
        for target_parent_id in parent_ids:
            if has_image_vector:
                results_text = search_client.search(
                    search_text="*",
                    filter=f"text_parent_id eq '{target_parent_id}'",
                    select=["*"],
                )
                results_image = search_client.search(
                    search_text="*",
                    filter=f"image_parent_id eq '{target_parent_id}'",
                    select=["*"],
                )
            else:
                results_text = search_client.search(
                    search_text="*",
                    filter=f"parent_id eq '{target_parent_id}'",
                    select=["*"],
                )
                results_image = []
            for doc in results_text:
                logger.warning(f"results_text doc keys: {list(doc.keys())}")
                if "chunk_id" in doc:
                    documents_to_delete.append({"chunk_id": doc["chunk_id"]})
                else:
                    logger.error(f"No chunk_id in doc: {doc}")
            for doc in results_image:
                logger.warning(f"results_image doc keys: {list(doc.keys())}")
                if "chunk_id" in doc:
                    documents_to_delete.append({"chunk_id": doc["chunk_id"]})
                else:
                    logger.error(f"No chunk_id in doc: {doc}")
        if documents_to_delete:
            search_client.delete_documents(documents=documents_to_delete)
            total_deleted = len(documents_to_delete)
        else:
            raise HTTPException(status_code=404, detail="No documents found to delete.")
        return DeleteByTitleResponse(
            message=f"Deleted {total_deleted} documents.",
            deleted_count=total_deleted,
            title=title,
        )
    except Exception as e:
        logger.error(
            f"An error occurred in the azure-index/delete_by_title endpoint: {e}"
        )
        raise HTTPException(status_code=500, detail=f"Internal error: {e}")
