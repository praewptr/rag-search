from dotenv import load_dotenv
from openai import AzureOpenAI
from chromadb import Client
from chromadb.config import Settings
from services.config import (
    azure_oai_endpoint,
    azure_oai_key,
    azure_oai_deployment,
)


# Azure OpenAI client
azure_openai_client = AzureOpenAI(
    base_url=f"{azure_oai_endpoint}/openai/deployments/{azure_oai_deployment}/extensions",
    api_key=azure_oai_key,
    api_version="2023-09-01-preview",
)

chroma_client = Client(
    Settings(
        persist_directory="./chroma_db",  # Directory to store the database
        anonymized_telemetry=False,
        is_persistent=True,
    )
)
