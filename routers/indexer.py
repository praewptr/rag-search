import requests
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from config import azure_search_endpoint, azure_search_key

router = APIRouter()

headers = {
    "Content-Type": "application/json",
    "api-key": azure_search_key,
}


class UploadOrCreateIndexRequest(BaseModel):
    connection_string: str
    container_name: str
    index_name: str
    indexer_name: str


class CreateDataSourceRequest(BaseModel):
    name: str
    container_name: str = "documents"


class CreateSkillsetRequest(BaseModel):
    name: str
    description: str = "Document processing skillset"


class CreateIndexRequest(BaseModel):
    name: str


class CreateIndexerRequest(BaseModel):
    name: str
    datasource_name: str
    target_index_name: str
    skillset_name: str


@router.post("/run-indexer")
def upload_and_trigger_indexer(
    indexer_name: str,
):
    headers = {"Content-Type": "application/json", "api-key": azure_search_key}

    run_response = requests.post(
        f"{azure_search_endpoint}/indexers/{indexer_name}/run?api-version=2023-10-01-Preview",
        headers=headers,
    )

    if run_response.status_code == 202:
        return {"message": "✅ Upload success & Indexer started (new files only)."}
    else:
        raise HTTPException(
            status_code=run_response.status_code,
            detail=f"❌ Failed to run indexer: {run_response.text}",
        )
