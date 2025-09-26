import requests
from azure.search.documents.indexes.models import (
    SearchField,
    SearchFieldDataType,
    SearchIndex,
)

from config import (
    azure_ai_service_key,
    azure_emb_oai_deployment,
    azure_emb_oai_endpoint,
    azure_emb_oai_key,
    azure_oai_deployment,
    azure_oai_endpoint,
    azure_oai_key,
    azure_search_endpoint,
    azure_search_index_doc,
    azure_search_key,
    azure_storage_connection_str,
)
from services.client import index_client

headers = {
    "Content-Type": "application/json",
    "api-key": azure_search_key,
}


def create_search_fields(name: str) -> list:
    """
    Define the fields for the Azure Search index.

    Returns:
        list: A list of SearchField objects defining the schema.
    """
    return [
        SearchField(
            name="chunk_id",
            type=SearchFieldDataType.String,
            key=True,
            searchable=True,
            filterable=False,
            retrievable=True,
            sortable=True,
            facetable=False,
            analyzer_name="keyword",
        ),
        SearchField(
            name="parent_id",
            type=SearchFieldDataType.String,
            searchable=False,
            filterable=True,
            retrievable=True,
            sortable=False,
            facetable=False,
            key=False,
        ),
        SearchField(
            name="chunk",
            type=SearchFieldDataType.String,
            searchable=True,
            filterable=False,
            retrievable=True,
            sortable=False,
            facetable=False,
        ),
        SearchField(
            name="title",
            type=SearchFieldDataType.String,
            searchable=True,
            filterable=False,
            retrievable=True,
            sortable=False,
            facetable=False,
        ),
        SearchField(
            name="text_vector",
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            filterable=False,
            searchable=True,
            retrievable=True,
            vector_search_dimensions=3072,
            vector_search_configuration=f"vector-config-{name}",
            vector_search_profile_name=f"vector-profile-{name}",
        ),
    ]


def create_vector_search_config(name: str) -> dict:
    """
    Define the vector search configuration for the Azure Search index.

    Returns:
        dict: A dictionary defining the vector search configuration.
    """
    return {
        "algorithms": [
            {
                "name": f"vector-config-{name}",
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
                "name": f"vector-profile-{name}",
                "algorithm": f"vector-config-{name}",
                "vectorizer": f"vectorizer-{name}",
            }
        ],
        "vectorizers": [
            {
                "name": f"vectorizer-{name}",
                "kind": "azureOpenAI",
                "azureOpenAIParameters": {
                    "resourceUri": azure_emb_oai_endpoint,
                    "deploymentId": azure_emb_oai_deployment,
                    "apiKey": azure_emb_oai_key,
                    "modelName": azure_emb_oai_deployment,
                },
            }
        ],
    }


def create_datasource(name: str, container_name: str):
    """Create a data source for Azure Search."""
    datasource_payload = {
        "name": name,
        "type": "azureblob",
        "credentials": {"connectionString": azure_storage_connection_str},
        "container": {"name": container_name},
    }
    response = requests.put(
        f"{azure_search_endpoint}/datasources/{name}?api-version=2023-10-01-Preview",
        headers=headers,
        json=datasource_payload,
    )
    return response


def create_skillset(name: str):
    """Create a skillset for Azure Search."""
    skillset_payload = {
        "name": name,
        "description": "Skillset to chunk documents and generate embeddings",
        "skills": [
            {
                "@odata.type": "#Microsoft.Skills.Vision.OcrSkill",
                "context": "/document/normalized_images/*",
                "lineEnding": "Space",
                "defaultLanguageCode": "en",
                "detectOrientation": True,
                "inputs": [
                    {"name": "image", "source": "/document/normalized_images/*"}
                ],
                "outputs": [{"name": "text", "targetName": "text"}],
            },
            {
                "@odata.type": "#Microsoft.Skills.Text.MergeSkill",
                "context": "/document",
                "insertPreTag": " ",
                "insertPostTag": " ",
                "inputs": [
                    {"name": "text", "source": "/document/content"},
                    {
                        "name": "itemsToInsert",
                        "source": "/document/normalized_images/*/text",
                    },
                    {
                        "name": "offsets",
                        "source": "/document/normalized_images/*/contentOffset",
                    },
                ],
                "outputs": [{"name": "mergedText", "targetName": "mergedText"}],
            },
            {
                "@odata.type": "#Microsoft.Skills.Text.SplitSkill",
                "context": "/document",
                "textSplitMode": "pages",
                "maximumPageLength": 2000,
                "pageOverlapLength": 500,
                "inputs": [{"name": "text", "source": "/document/mergedText"}],
                "outputs": [{"name": "textItems", "targetName": "pages"}],
            },
            {
                "@odata.type": "#Microsoft.Skills.Text.AzureOpenAIEmbeddingSkill",
                "context": "/document/pages/*",
                "resourceUri": azure_oai_endpoint,
                "apiKey": azure_oai_key,
                "deploymentId": azure_oai_deployment,
                "modelName": "text-embedding-3-large",
                "inputs": [{"name": "text", "source": "/document/pages/*"}],
                "outputs": [{"name": "embedding", "targetName": "text_vector"}],
            },
        ],
        "cognitiveServices": {
            "@odata.type": "#Microsoft.Azure.Search.CognitiveServicesByKey",
            "description": "AI Services",
            "key": azure_ai_service_key,
        },
    }
    response = requests.post(
        f"{azure_search_endpoint}/skillsets?api-version=2024-07-01",
        headers=headers,
        json=skillset_payload,
    )
    return response


def create_index(name: str, fields: list, vector_search: dict):
    """Create an Azure Search index."""
    search_index = SearchIndex(
        name=name,
        fields=fields,
        similarity={"@odata.type": "#Microsoft.Azure.Search.BM25Similarity"},
        vector_search=vector_search,
    )
    response = index_client.create_index(search_index)
    return response


def create_indexer(
    name: str, datasource_name: str, target_index_name: str, skillset_name: str
):
    """Create an indexer for Azure Search."""
    payload = {
        "name": name,
        "description": "an indexer",
        "dataSourceName": datasource_name,
        "targetIndexName": target_index_name,
        "skillsetName": skillset_name,
        "parameters": {
            "configuration": {
                "dataToExtract": "contentAndMetadata",
                "parsingMode": "default",
                "imageAction": "generateNormalizedImages",
            }
        },
        "fieldMappings": [
            {"sourceFieldName": "metadata_storage_name", "targetFieldName": "title"},
        ],
    }
    response = requests.post(
        f"{azure_search_endpoint}/indexers?api-version=2024-07-01",
        json=payload,
        headers=headers,
    )
    return response


def execute_full_flow(connection_string: str, container_name: str):
    """Execute the full flow of creating a datasource, skillset, index, and indexer."""
    try:
        # Step 1: Create a data source
        datasource_response = create_datasource(
            name="test-datasource",
            connection_string=connection_string,
            container_name=container_name,
        )
        print("Datasource Response:", datasource_response.json())

        # Step 2: Create a skillset
        skillset_response = create_skillset(name="test-skillset")
        print("Skillset Response:", skillset_response.json())

        # Step 3: Create an index
        fields = create_search_fields(name="rag-pdf")
        vector_search = create_vector_search_config(name="rag-pdf")
        index_response = create_index(
            name="rag-index",
            fields=fields,
            vector_search=vector_search,
        )
        print("Index Response:", index_response)

        # Step 4: Create an indexer
        indexer_response = create_indexer(
            name="test-indexer",
            datasource_name="test-datasource",
            target_index_name="test-index",
            skillset_name="test-skillset",
        )
        print("Indexer Response:", indexer_response.json())

        print("Full flow executed successfully.")
    except Exception as e:
        print(f"An error occurred during the full flow execution: {e}")


def upload_new_document(connection_string: str, container_name: str, indexer_name: str):
    """Upload a new document to blob storage and run the indexer if the index already exists."""
    try:
        # Check if the index already exists
        index_check_response = requests.get(
            f"{azure_search_endpoint}/indexes/{azure_search_index_doc}?api-version=2023-10-01-Preview",
            headers=headers,
        )

        if index_check_response.status_code == 200:
            print("✅ Index already exists. Uploading new document to blob storage.")

            # Simulate document upload to blob storage
            print("Uploading document to blob storage...")
            # Add your blob upload logic here

            # Run the indexer
            run_response = requests.post(
                f"{azure_search_endpoint}/indexers/{indexer_name}/run?api-version=2023-10-01-Preview",
                headers=headers,
            )

            if run_response.status_code == 202:
                print("✅ Indexer started successfully (for new files only).")
            else:
                print("❌ Failed to run the indexer:", run_response.status_code)
                print(run_response.text)
        else:
            print("❌ Index does not exist. Please create the index first.")
    except Exception as e:
        print(
            f"An error occurred while uploading the document and running the indexer: {e}"
        )


def upload_or_create_index(
    connection_string: str, container_name: str, index_name: str, indexer_name: str
):
    """Upload a document to an existing index or create a new index if it doesn't exist."""
    try:
        # Check if the index already exists
        index_check_response = requests.get(
            f"{azure_search_endpoint}/indexes/{index_name}?api-version=2023-10-01-Preview",
            headers=headers,
        )

        if index_check_response.status_code == 200:
            print(
                f"✅ Index '{index_name}' already exists. Uploading new document to blob storage."
            )

            # Simulate document upload to blob storage
            print("Uploading document to blob storage...")
            # Add your blob upload logic here

            # Run the indexer
            run_response = requests.post(
                f"{azure_search_endpoint}/indexers/{indexer_name}/run?api-version=2023-10-01-Preview",
                headers=headers,
            )

            if run_response.status_code == 202:
                print("✅ Indexer started successfully (for new files only).")
            else:
                print("❌ Failed to run the indexer:", run_response.status_code)
                print(run_response.text)
        elif index_check_response.status_code == 404:
            print(f"❌ Index '{index_name}' not found. Creating a new index.")

            # Execute the full flow to create a new index
            execute_full_flow(connection_string, container_name)
        else:
            print(
                "❌ Failed to check index existence:", index_check_response.status_code
            )
            print(index_check_response.text)
    except Exception as e:
        print(
            f"An error occurred while uploading the document or creating the index: {e}"
        )


if __name__ == "__main__":
    # Example usage of the full flow
    execute_full_flow(
        connection_string="<your_connection_string>",
        container_name="<your_container_name>",
    )

    # Example usage of uploading a new document
    upload_new_document(
        connection_string="<your_connection_string>",
        container_name="<your_container_name>",
        indexer_name="rag-indexer",
    )

    # Example usage of uploading or creating an index
    upload_or_create_index(
        connection_string="<your_connection_string>",
        container_name="<your_container_name>",
        index_name="rag-index",
        indexer_name="rag-1757995320248-indexer",
    )
