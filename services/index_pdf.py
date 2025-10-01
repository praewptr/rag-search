import json
from pathlib import Path

import requests
from azure.search.documents.indexes.models import (
    SearchField,
    SearchFieldDataType,
    SearchIndex,
)

from config import (
    azure_ai_service_endpoint,
    azure_ai_service_key,
    azure_emb_oai_deployment,
    azure_emb_oai_endpoint,
    azure_emb_oai_key,
    azure_search_endpoint,
    azure_search_key,
    azure_storage_connection_str,
)
from services.client import index_client

headers = {
    "Content-Type": "application/json",
    "api-key": azure_search_key,
}


def create_search_fields(
    name: str, analyzer_name: str, include_vision: bool = True
) -> list:
    """
    Define the fields for the Azure Search index.

    Returns:
        list: A list of SearchField objects defining the schema.
    """
    if analyzer_name == "en":
        analyzer_name = "en.lucene"
    else:
        analyzer_name = "th"
        analyzer_name = "th.lucene"

    fields = [
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
            name="text_parent_id",
            type=SearchFieldDataType.String,
            searchable=False,
            filterable=True,
            retrievable=True,
            sortable=False,
            facetable=False,
            key=False,
        ),
        SearchField(
            name="image_parent_id",
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
            analyzer_name=analyzer_name,
        ),
        SearchField(
            name="title",
            type=SearchFieldDataType.String,
            searchable=True,
            filterable=False,
            retrievable=True,
            sortable=False,
            facetable=False,
            analyzer_name="en.lucene",
        ),
        SearchField(
            name="text_vector",
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            filterable=False,
            searchable=True,
            retrievable=True,
            vector_search_dimensions=3072,
            vector_search_profile_name=f"{name}-azureOpenAi-text-profile",
        ),
    ]

    # Only add image vector field if vision is enabled
    if include_vision:
        fields.append(
            SearchField(
                name="image_vector",
                type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                filterable=False,
                searchable=True,
                retrievable=True,
                vector_search_dimensions=1024,
                vector_search_profile_name=f"{name}-aiServicesVision-image-profile",
            )
        )

    return fields


def create_vector_search_config(name: str, include_vision: bool = False) -> dict:
    if include_vision:
        config_path = Path("vector_config_template.json")
    else:
        config_path = Path("vector_config_template_text_only.json")

    with open(config_path, "r", encoding="utf-8") as f:
        template = json.load(f)

    config_str = json.dumps(template)

    # Replace placeholders
    config_str = config_str.replace("__NAME__", name)
    config_str = config_str.replace("__RESOURCE_URI__", azure_emb_oai_endpoint)
    config_str = config_str.replace("__DEPLOYMENT_ID__", azure_emb_oai_deployment)
    config_str = config_str.replace("__API_KEY__", azure_emb_oai_key)
    config_str = config_str.replace("__MODEL_NAME__", azure_emb_oai_deployment)
    if include_vision:
        config_str = config_str.replace("__COGNITIVE_KEY__", azure_ai_service_key)
        config_str = config_str.replace(
            "__AI_SERVICE_RESOURCE_URI__", azure_ai_service_endpoint
        )

    return json.loads(config_str)


def create_datasource(name: str, container_name: str):
    """Create a data source for Azure Search."""
    datasource_payload = {
        "name": name,
        "type": "azureblob",
        "credentials": {"connectionString": azure_storage_connection_str},
        "container": {"name": container_name},
    }
    try:
        response = requests.put(
            f"{azure_search_endpoint}/datasources/{name}?api-version=2023-10-01-Preview",
            headers=headers,
            json=datasource_payload,
        )
        return response

    except Exception as e:
        print(f"Error creating datasource: {e}")
        return None


def create_skillset(name: str, target_index_name: str = None):
    """Create a skillset with optional vision capabilities."""

    template_path = Path("skillset_template_text_only.json")

    with open(template_path, "r", encoding="utf-8") as f:
        template = f.read()

    # Replace placeholders
    config_str = (
        template.replace("__NAME__", name)
        .replace("__RESOURCE_URI__", azure_emb_oai_endpoint)
        .replace("__API_KEY__", azure_emb_oai_key)
        .replace("__DEPLOYMENT_ID__", azure_emb_oai_deployment)
        .replace("__COGNITIVE_KEY__", azure_ai_service_key)
        .replace("__COGNITIVE_ENDPOINT__", azure_ai_service_endpoint)
    )

    # Replace index name if provided
    if target_index_name:
        config_str = config_str.replace("__INDEX_NAME__", target_index_name)

    try:
        skillset_payload = json.loads(config_str)

        response = requests.post(
            f"{azure_search_endpoint}/skillsets?api-version=2024-03-01-preview",
            headers=headers,
            json=skillset_payload,
        )

        if response.status_code not in [200, 201]:
            print(f"Error creating skillset: {response.status_code} - {response.text}")

        return response

    except json.JSONDecodeError as e:
        print(f"JSON parsing error in skillset template: {e}")
        return None
    except Exception as e:
        print(f"Error creating skillset: {e}")
        return None


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
            {
                "sourceFieldName": "metadata_storage_name",
                "targetFieldName": "title",
            }
        ],
    }
    response = requests.post(
        f"{azure_search_endpoint}/indexers?api-version=2023-10-01-Preview",
        json=payload,
        headers=headers,
    )
    return response
