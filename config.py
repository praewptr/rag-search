from dotenv import load_dotenv
import os
import logging

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


# Validate keys
if not azure_oai_key or not azure_search_key:
    raise ValueError("Missing required keys. Please check your .env file.")


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
