import logging
import os

from dotenv import load_dotenv

# Load environment variables
env_path = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(env_path)


azure_oai_key = os.getenv("AZURE_OAI_KEY")
azure_search_key = os.getenv("AZURE_SEARCH_KEY")
azure_oai_endpoint = os.getenv("AZURE_OAI_ENDPOINT")
azure_oai_deployment = os.getenv("AZURE_OAI_DEPLOYMENT")
azure_search_endpoint = os.getenv("AZURE_SEARCH_ENDPOINT")
azure_search_index_txt = os.getenv("AZURE_SEARCH_INDEX_TXT")
azure_search_index_doc = os.getenv("AZURE_SEARCH_INDEX_DOC")
azure_emb_oai_key = os.getenv("AZURE_EMB_OAI_KEY")
azure_emb_oai_deployment = os.getenv("AZURE_EMB_OAI_DEPLOYMENT")
azure_emb_oai_endpoint = os.getenv("AZURE_EMB_OAI_ENDPOINT")
azure_document_intelligence_key = os.getenv("AZURE_DOC_INT_KEY")
azure_document_intelligence_endpoint = os.getenv("AZURE_DOC_INT_ENDPOINT")

# Blob Storage Config
azure_storage_account_name = os.getenv("AZURE_STORAGE_ACCOUNT_NAME")
azure_storage_account_key = os.getenv("AZURE_STORAGE_ACCOUNT_KEY")
azure_storage_container_name = os.getenv("AZURE_STORAGE_CONTAINER_NAME")
azure_storage_connection_str = os.getenv("AZURE_STORAGE_CONNECTION_STR")

azure_ai_service_endpoint = os.getenv("AZURE_AI_SERVICE_ENDPOINT")
azure_ai_service_key = os.getenv("AZURE_AI_SERVICE_KEY")

# Oracle Database Configuration
oracle_db_user = os.getenv("ORACLE_DB_USER", "digital_test")
oracle_db_password = os.getenv("ORACLE_DB_PASSWORD", "test")
oracle_db_host = os.getenv("ORACLE_DB_HOST", "172.16.7.117")
oracle_db_port = int(os.getenv("ORACLE_DB_PORT", "1521"))
oracle_db_service_name = os.getenv("ORACLE_DB_SERVICE_NAME", "NYTG")

# Validate keys
if not azure_oai_key or not azure_search_key:
    raise ValueError("Missing required keys. Please check your .env file.")

# Validate Azure OpenAI Embeddings configuration (optional but recommended for vector search)
missing_embedding_vars = []
if not azure_emb_oai_key:
    missing_embedding_vars.append("AZURE_EMB_OAI_KEY")
if not azure_emb_oai_endpoint:
    missing_embedding_vars.append("AZURE_EMB_OAI_ENDPOINT")
if not azure_emb_oai_deployment:
    missing_embedding_vars.append("AZURE_EMB_OAI_DEPLOYMENT")

if missing_embedding_vars:
    print(
        f"Warning: Azure OpenAI Embeddings not fully configured. Missing: {', '.join(missing_embedding_vars)}"
    )
    print(
        "Vector search will be disabled for new indexes. Text search will still work."
    )
    print(
        "To enable vector search, add the missing environment variables to your .env file."
    )


# Logging Configuration
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()  # Default to INFO level
ENABLE_LOGGING = (
    os.getenv("ENABLE_LOGGING", "true").lower() == "true"
)  # Enable/disable logging


def setup_logging():
    """Configure logging for the application"""
    if not ENABLE_LOGGING:
        # Disable all logging
        logging.disable(logging.CRITICAL)
        print("Logging disabled")
        return

    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, LOG_LEVEL),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Set specific logger levels if needed
    # For example, to reduce Azure SDK logging:
    logging.getLogger("azure").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    print(f"Logging configured at level: {LOG_LEVEL}")
