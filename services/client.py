from dotenv import load_dotenv
from openai import AzureOpenAI

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
