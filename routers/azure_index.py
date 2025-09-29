import logging
import os
from typing import Optional

from azure.core.credentials import AzureKeyCredential
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex,
)
from dotenv import load_dotenv
from fastapi import APIRouter, HTTPException, Query

from config import (
    azure_search_endpoint,
    azure_search_index_txt,
    azure_search_key,
)
from models.azure_index import CreateIndexRequest, CreateIndexResponse
from models.upload_txt import (
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
            # Index doesn't exist, which is what we want
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
        logger.error(f"Error creating index: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to create index '{request.name}': {str(e)}"
        )


@router.get("/indexes")
async def list_available_indexes():
    """
    Get list of available Azure Search indexes.

    Returns:
        List of available indexes
    """
    try:
        index_client = get_index_client()
        indexes = list(index_client.list_indexes())

        available_indexes = []
        for index in indexes:
            available_indexes.append({"name": index.name})

        return {
            "status": "success",
            "indexes": available_indexes,
            "default_index": azure_search_index_txt,
        }

    except Exception as e:
        logger.error(f"Error listing indexes: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to list indexes: {str(e)}")


@router.delete("/index/{index_name}")
async def delete_index(index_name: str):
    """
    Delete an Azure Search index.

    Args:
        index_name: Name of the index to delete

    Returns:
        Success message
    """
    try:
        await handle_index_error(index_name, "deletion")
        index_client = get_index_client()
        logger.info(f"Deleting index: {index_name}")
        index_client.delete_index(index_name)
        logger.info(f"Successfully deleted index: {index_name}")
        return {
            "status": "success",
            "message": f"Index '{index_name}' deleted successfully",
            "index_name": index_name,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting index: {e}")
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
        logger.error(f"Error getting index stats: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get index statistics: {str(e)}"
        )


@router.get("/search", response_model=SearchResponse)
async def search_index(
    query: str = Query(..., description="Search query"),
    top: int = Query(50, description="Number of results to return", le=100),
    skip: int = Query(0, description="Number of results to skip", ge=0),
):
    """
    Search the Azure Search text index.

    Args:
        query: Search query string
        top: Number of results to return (max 100)
        skip: Number of results to skip for pagination

    Returns:
        SearchResponse: Search results with metadata
    """
    try:
        search_client = get_search_client()

        logger.info(f"Searching index '{azure_search_index_txt}' for: '{query}'")

        # Perform search
        results = search_client.search(
            search_text=query if query != "*" else "",
            top=top,
            skip=skip,
            include_total_count=True,
            select=["id", "content", "source", "timestamp", "chunk_id"],
        )

        # Process results
        search_results = []
        total_count = 0

        for result in results:
            search_results.append(
                SearchResult(
                    id=result.get("id", ""),
                    content=result.get("content", ""),
                    source=result.get("source", ""),
                    timestamp=result.get("timestamp", ""),
                    chunk_id=result.get("chunk_id"),
                    score=result.get("@search.score"),
                )
            )

        # Get total count from results
        try:
            total_count = results.get_count()
        except Exception:
            total_count = len(search_results)

        logger.info(
            f"Found {total_count} total results, returning {len(search_results)} results"
        )

        return SearchResponse(
            total_count=total_count,
            results=search_results,
            query=query,
            status="success",
        )

    except Exception as e:
        logger.error(f"Error searching index: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to search index: {str(e)}")


@router.get("/browse")
async def browse_all_documents(
    page: int = Query(1, description="Page number", ge=1),
    page_size: int = Query(20, description="Documents per page", le=50),
    index_name: Optional[str] = Query(None, description="Index name to browse"),
):
    """
    Browse all documents in the Azure Search text index with pagination.
    Uses different method if index has 'title' field - returns list of unique titles.

    Args:
        page: Page number (1-based)
        page_size: Number of documents per page (max 50)
        index_name: Optional index name to browse (defaults to text index)

    Returns:
        Union[SearchResponse, TitleListResponse]: Document results or title list
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
        logger.error(f"Error browsing index: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to browse index: {str(e)}")


@router.get("/document/{doc_id}")
async def get_document_by_id(doc_id: str):
    """
    Get a specific document by its ID from the Azure Search index.

    Args:
        doc_id: Document ID to retrieve

    Returns:
        Document data
    """
    try:
        search_client = get_search_client()

        logger.info(f"Fetching document with ID: {doc_id}")

        # Search for the specific document
        results = search_client.search(
            search_text="*",
            filter=f"id eq '{doc_id}'",
            select=["id", "content", "source", "timestamp", "chunk_id"],
            top=1,
        )

        result_list = list(results)

        if not result_list:
            raise HTTPException(
                status_code=404, detail=f"Document with ID '{doc_id}' not found"
            )

        document = result_list[0]

        return SearchResult(
            id=document.get("id", ""),
            content=document.get("content", ""),
            source=document.get("source", ""),
            timestamp=document.get("timestamp", ""),
            chunk_id=document.get("chunk_id"),
            score=document.get("@search.score"),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching document: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to fetch document: {str(e)}"
        )


@router.delete("/documents/{doc_id}")
async def delete_document_from_index(
    doc_id: str,
    index_name: Optional[str] = Query(None, description="Index name to delete from"),
):
    """
    Delete a specific document by its ID from the Azure Search index.

    Args:
        doc_id: Document ID to delete
        index_name: Optional index name to delete from (defaults to text index)

    Returns:
        Success message
    """
    try:
        search_client = get_search_client(index_name)
        selected_index = index_name or azure_search_index_txt

        logger.info(f"Deleting document with ID: {doc_id} from index: {selected_index}")

        # First, check if document exists
        check_results = search_client.search(
            search_text="*", filter=f"id eq '{doc_id}'", select=["id"], top=1
        )

        if not list(check_results):
            raise HTTPException(
                status_code=404, detail=f"Document with ID '{doc_id}' not found"
            )

        # Delete the document
        search_client.delete_documents([{"id": doc_id}])

        logger.info(f"Successfully deleted document with ID: {doc_id}")

        return {
            "message": f"Successfully deleted document with ID: {doc_id}",
            "deleted_id": doc_id,
            "status": "success",
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting document: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to delete document: {str(e)}"
        )
