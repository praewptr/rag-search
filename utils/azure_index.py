from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchField,
    SearchFieldDataType,
)

from config import (
    azure_emb_oai_deployment,
    azure_emb_oai_key,
    azure_search_endpoint,
    azure_search_index_txt,
    azure_search_key,
)


def get_search_client(index_name: str = None):
    """Initialize and return Azure Search client."""
    # Use provided index_name or default to txt index
    selected_index = index_name or azure_search_index_txt
    return SearchClient(
        endpoint=azure_search_endpoint,
        index_name=selected_index,
        credential=AzureKeyCredential(azure_search_key),
    )


def get_index_client():
    """Initialize and return Azure Search Index client for managing indexes."""
    return SearchIndexClient(
        endpoint=azure_search_endpoint,
        credential=AzureKeyCredential(azure_search_key),
    )


def create_search_fields() -> list:
    """
    Define the fields for the Azure Search index.

    Returns:
        list: A list of SearchField objects defining the schema.
    """
    return [
        SearchField(
            name="id",
            type=SearchFieldDataType.String,
            key=True,
            searchable=False,
            filterable=True,
            retrievable=True,
            sortable=True,
            facetable=True,
        ),
        SearchField(
            name="content",
            type=SearchFieldDataType.String,
            searchable=True,
            filterable=True,
            retrievable=True,
            sortable=True,
            facetable=True,
            analyzer_name="th.microsoft",
        ),
        SearchField(
            name="source",
            type=SearchFieldDataType.String,
            searchable=True,
            filterable=True,
            retrievable=True,
            sortable=True,
            facetable=True,
        ),
        SearchField(
            name="timestamp",
            type=SearchFieldDataType.String,
            searchable=False,
            filterable=True,
            retrievable=True,
            sortable=True,
            facetable=True,
        ),
        SearchField(
            name="chunk_id",
            type=SearchFieldDataType.String,
            searchable=False,
            filterable=True,
            retrievable=True,
            sortable=True,
            facetable=True,
        ),
        SearchField(
            name="text_vector",
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            searchable=True,
            retrievable=True,
            vector_search_dimensions=3072,
            vector_search_configuration="vector-config-1758767420214",
            vector_search_profile_name="vector-profile-1758767419468",
        ),
    ]


def create_vector_search_config() -> dict:
    """
    Define the vector search configuration for the Azure Search index.

    Returns:
        dict: A dictionary defining the vector search configuration.
    """
    return {
        "algorithms": [
            {
                "name": "vector-config-1758767420214",
                "kind": "hnsw",
                "hnswParameters": {
                    "metric": "cosine",
                    "m": 4,
                    "efConstruction": 400,
                    "efSearch": 500,
                },
            }
        ],
        "profiles": [
            {
                "name": "vector-profile-1758767419468",
                "algorithm": "vector-config-1758767420214",
                "vectorizer": "vectorizer-1758767424582",
            }
        ],
        "vectorizers": [
            {
                "name": "vectorizer-1758767424582",
                "kind": "azureOpenAI",
                "azureOpenAIParameters": {
                    "resourceUri": "https://garmi-openai.openai.azure.com",
                    "deploymentId": "text-embedding-3-large",
                    "apiKey": azure_emb_oai_key,
                    "modelName": azure_emb_oai_deployment,
                },
            }
        ],
    }
