from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.storage.blob import BlobServiceClient
from langchain_community.retrievers.azure_ai_search import AzureAISearchRetriever
from langchain_openai import AzureChatOpenAI
from langchain_openai.embeddings import AzureOpenAIEmbeddings
from openai import AzureOpenAI

from config import (
    azure_emb_oai_deployment,
    azure_emb_oai_endpoint,
    azure_emb_oai_key,
    azure_oai_deployment,
    azure_oai_endpoint,
    azure_oai_key,
    azure_search_endpoint,
    azure_search_index_doc,
    azure_search_index_txt,
    azure_search_key,
    azure_storage_connection_str,
    azure_storage_container_name,
)

# project_dir = os.path.dirname(os.path.abspath(__file__))
# if project_dir not in sys.path:
#     sys.path.append(project_dir)


# --------- Initialize Azure OpenAI client---------#
azure_openai_client = AzureOpenAI(
    base_url=f"{azure_oai_endpoint}/openai/deployments/{azure_oai_deployment}/extensions",
    api_key=azure_oai_key,
    api_version="2023-09-01-preview",
)


# ------------Embedding Client-----------------#
embeddings = AzureOpenAIEmbeddings(
    deployment=azure_emb_oai_deployment,
    azure_endpoint=azure_emb_oai_endpoint,
    openai_api_key=azure_emb_oai_key,
    openai_api_version="2023-09-01-preview",
)

# ------------ Search Client -----------------#
search_client_pdf = SearchClient(
    endpoint=azure_search_endpoint,
    index_name=azure_search_index_doc,
    credential=AzureKeyCredential(azure_search_key),
)

search_client_text = SearchClient(
    endpoint=azure_search_endpoint,
    index_name=azure_search_index_txt,
    credential=AzureKeyCredential(azure_search_key),
)

# -------------llm------------------#
llm = AzureChatOpenAI(
    deployment_name=azure_oai_deployment,
    azure_endpoint=azure_oai_endpoint,  # Explicitly pass azure_endpoint
    openai_api_key=azure_oai_key,
    openai_api_type="azure",
    openai_api_version="2023-09-01-preview",
    temperature=0.3,
    max_tokens=1000,
)

# -------------langchain Retriver PDF------------------#
retriever_pdf = AzureAISearchRetriever(
    content_key="chunk",
    index_name=azure_search_index_doc,
    service_name=azure_search_endpoint.replace("https://", "").replace(
        ".search.windows.net", ""
    ),
    api_key=azure_search_key,
    api_version="2023-10-01-Preview",
)


# -------------langchain Retriver Text------------------#
retriever_text = AzureAISearchRetriever(
    content_key="content",
    top_k=3,
    index_name=azure_search_index_txt,
    service_name=azure_search_endpoint.replace("https://", "").replace(
        ".search.windows.net", ""
    ),
    api_key=azure_search_key,
    api_version="2023-10-01-Preview",
)


# ----------------Search Index Client-----------------#
index_client = SearchIndexClient(
    endpoint=azure_search_endpoint, credential=AzureKeyCredential(azure_search_key)
)


# ----------------Blob Storage Client-----------------#
blob_service_client = BlobServiceClient.from_connection_string(
    azure_storage_connection_str
)
blob_container_client = blob_service_client.get_container_client(
    azure_storage_container_name
)
