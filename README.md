# RAG PDF Question Answering System

## Overview
This project is a Retrieval-Augmented Generation (RAG) system for answering questions based on uploaded PDF documents. It combines FastAPI for the backend, Azure AI services for document processing and search, and provides a web interface for document management and question answering.

## Technology Stack
- **Backend**: FastAPI (Python web framework)
- **AI Services**: Azure OpenAI, Azure AI Search, Azure Document Intelligence
- **Storage**: Azure Blob Storage
- **Frontend**: Static HTML/CSS/JavaScript
- **Database**: Oracle Database (optional)

## Project Architecture

```
rag-search/
├── main.py                    # FastAPI application entry point
├── config.py                  # Configuration and environment variables
├── requirements.txt           # Python dependencies
├── data_manager_api.py        # Legacy API for data management
├── test_oracle.py            # Oracle database connection tests
├── test.ipynb                # Jupyter notebook for testing
├── rag_text.json            # Sample RAG text data
├── sample-blob.txt          # Sample text file for testing
├── sample.pdf               # Sample PDF for testing
├── models/                  # Data models and schemas
├── routers/                 # FastAPI route handlers
├── services/                # Business logic and integrations
├── static/                  # Frontend HTML/CSS/JS files
└── utils/                   # Utility functions
```

## Detailed File Descriptions

### Core Application Files

#### `main.py`
**Purpose**: FastAPI application entry point and server configuration.
**Key Functions**:
- Initializes the FastAPI application with CORS middleware
- Mounts static file serving for the frontend
- Includes all router modules (search, text management, Azure index, PDF upload, indexer creation)
- Serves the main dashboard at the root endpoint (`/`)
- Configures logging through the config module

**Important Notes**: This is the file to run to start the server (`python main.py` or `uvicorn main:app`)

#### `config.py`
**Purpose**: Central configuration management and environment variable loading.
**Key Components**:
- Loads environment variables from `.env` file
- Defines Azure service configurations (OpenAI, Search, Document Intelligence, Blob Storage)
- Sets up logging configuration
- Contains all API keys and endpoint configurations

**Required Environment Variables**:
- `AZURE_OAI_KEY`, `AZURE_OAI_ENDPOINT`, `AZURE_OAI_DEPLOYMENT`: Azure OpenAI configuration
- `AZURE_SEARCH_KEY`, `AZURE_SEARCH_ENDPOINT`: Azure AI Search configuration
- `AZURE_SEARCH_INDEX_TXT`, `AZURE_SEARCH_INDEX_DOC`: Search index names
- `AZURE_EMB_OAI_KEY`, `AZURE_EMB_OAI_ENDPOINT`, `AZURE_EMB_OAI_DEPLOYMENT`: Embedding model config
- `AZURE_DOC_INT_KEY`, `AZURE_DOC_INT_ENDPOINT`: Document Intelligence configuration
- `AZURE_STORAGE_*`: Blob storage configuration

#### `requirements.txt`
**Purpose**: Python package dependencies for the project.
**Key Dependencies**:
- `fastapi`: Web framework
- `uvicorn`: ASGI server
- `openai`: Azure OpenAI client
- `python-dotenv`: Environment variable management
- Additional supporting libraries for HTTP requests, data validation, etc.

### Router Files (`routers/`)

#### `routers/search.py`
**Purpose**: Handles question-answering API endpoints.
**Key Endpoints**:
- `POST /search/aisearch`: Main Q&A endpoint using Azure AI Search
- `POST /search/langchain`: Alternative Q&A using LangChain
**Functionality**: 
- Processes user questions
- Retrieves relevant documents from Azure Search
- Generates answers using Azure OpenAI
- Returns formatted responses with sources and citations

#### `routers/text_manage.py`
**Purpose**: Text document management and Azure Search integration.
**Key Endpoints**:
- `POST /text-manage/upload`: Upload text documents to Azure Search
- `GET /text-manage/documents`: List all documents in the index
- `DELETE /text-manage/delete/{doc_id}`: Delete specific documents
**Functionality**: 
- Manages text document lifecycle
- Handles document indexing in Azure Search
- Provides document CRUD operations

#### `routers/azure_index.py`
**Purpose**: Azure Search index management and browsing.
**Key Endpoints**:
- `GET /azure-index/indexes`: List all available Azure Search indexes
- `GET /azure-index/documents/{index_name}`: Browse documents in a specific index
- `DELETE /azure-index/documents/{index_name}/{doc_id}`: Delete documents from indexes
**Functionality**: 
- Provides interface to Azure Search indexes
- Allows browsing and managing indexed content
- Supports multiple index operations

#### `routers/upload_pdf.py`
**Purpose**: PDF document upload and processing.
**Key Endpoints**:
- `POST /upload-pdf/upload`: Upload and process PDF files
- `GET /upload-pdf/documents`: List uploaded PDF documents
**Functionality**: 
- Handles PDF file uploads
- Processes PDFs using Azure Document Intelligence
- Indexes PDF content in Azure Search
- Stores processed documents in Azure Blob Storage


### Service Files (`services/`)

#### `services/qa_engine.py`
**Purpose**: Core question-answering logic using Azure OpenAI and Search.
**Key Functions**:
- `get_response()`: Main Q&A function that combines search and generation
- Integrates Azure OpenAI with Azure Search data sources
- Handles citation extraction and formatting
- Manages conversation context and response generation

#### `services/client.py`
**Purpose**: Azure service client initialization and configuration.
**Key Components**:
- Azure OpenAI client setup
- Azure Search client configuration
- Document Intelligence client initialization
- Centralized client management for all Azure services

#### `services/langchain_flow.py`
**Purpose**: Alternative Q&A implementation using LangChain framework.
**Key Functions**:
- `generate_answer()`: LangChain-based answer generation
- Provides alternative processing pipeline
- Handles document retrieval and answer synthesis
- Useful for comparing different RAG approaches

#### `services/azure_upload.py`
**Purpose**: Azure Blob Storage integration for file management.
**Key Functions**:
- File upload to Azure Blob Storage
- Blob metadata management
- File retrieval and deletion operations
- Storage container management

#### `services/oracle_db.py`
**Purpose**: Oracle database integration (optional component).
**Key Functions**:
- Database connection management
- Data persistence operations
- Query execution and result processing
- Optional storage backend for application data

#### `services/utils.py`
**Purpose**: Common utility functions used throughout the application.
**Key Functions**:
- Text processing and cleaning utilities
- Citation marker removal
- Format conversion helpers
- Common data transformation functions

### Frontend Files (`static/`)

#### `static/dashboard.html`
**Purpose**: Main dashboard and landing page for the application.
**Features**:
- Navigation to all major features
- System status overview
- Quick access to document upload and search
- Responsive design for desktop and mobile

#### `static/data_manager.html`
**Purpose**: Text document management interface.
**Features**:
- Upload text documents
- View and manage uploaded documents
- Delete documents from the system
- Real-time document status updates

#### `static/azure_index.html`
**Purpose**: Azure Search index browser and management interface.
**Features**:
- Browse available Azure Search indexes
- View documents within indexes
- Search and filter indexed content
- Manage index documents

#### `static/upload_pdf.html`
**Purpose**: PDF document upload and processing interface.
**Features**:
- Drag-and-drop PDF upload
- Processing status indicators
- View uploaded PDF documents
- Integration with Azure Document Intelligence

#### `static/indexer_manager.html`
**Purpose**: Azure Search infrastructure management interface.
**Features**:
- Create and manage data sources
- Configure search indexes
- Set up automated indexers
- Monitor indexing operations

#### `static/chatbot.html`
**Purpose**: Interactive question-answering interface.
**Features**:
- Chat-like interface for asking questions
- Real-time answer generation
- Source citation display
- Conversation history management

### Model Files (`models/`)

#### `models/rag_search.py`
**Purpose**: Pydantic models for API request/response validation.
**Key Models**:
- `QuestionRequest`: Validates incoming question requests
- Data models for search responses
- Validation schemas for all API endpoints

#### `models/azure_index.py`
**Purpose**: Azure Search-specific data models and schemas.
**Key Models**:
- Index configuration models
- Document schema definitions
- Search result formatting models

### Test Files

#### `test_oracle.py`
**Purpose**: Oracle database connection and functionality tests.
**Key Tests**:
- Database connectivity verification
- Query execution tests
- Data persistence validation

#### `test.ipynb`
**Purpose**: Jupyter notebook for interactive testing and development.
**Contains**:
- API endpoint testing
- Data processing experiments
- Integration testing workflows
- Development and debugging tools

## Setup Instructions

### Prerequisites
- Python 3.8 or higher
- Azure subscription with the following services enabled:
  - Azure OpenAI
  - Azure AI Search
  - Azure Document Intelligence
  - Azure Blob Storage
- Oracle Database (optional)

### Installation Steps

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd rag-search
   ```

2. **Create virtual environment**:
   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # Linux/Mac
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables**:
   Create a `.env` file in the root directory with the following variables:
   ```env
   AZURE_OAI_KEY=your_azure_openai_key
   AZURE_OAI_ENDPOINT=your_azure_openai_endpoint
   AZURE_OAI_DEPLOYMENT=your_deployment_name
   AZURE_SEARCH_KEY=your_search_key
   AZURE_SEARCH_ENDPOINT=your_search_endpoint
   AZURE_SEARCH_INDEX_TXT=your_text_index_name
   AZURE_SEARCH_INDEX_DOC=your_document_index_name
   AZURE_EMB_OAI_KEY=your_embedding_key
   AZURE_EMB_OAI_ENDPOINT=your_embedding_endpoint
   AZURE_EMB_OAI_DEPLOYMENT=your_embedding_deployment
   AZURE_DOC_INT_KEY=your_document_intelligence_key
   AZURE_DOC_INT_ENDPOINT=your_document_intelligence_endpoint
   AZURE_STORAGE_ACCOUNT_NAME=your_storage_account
   AZURE_STORAGE_ACCOUNT_KEY=your_storage_key
   AZURE_STORAGE_CONTAINER_NAME=your_container_name
   AZURE_STORAGE_CONNECTION_STR=your_connection_string
   ```

5. **Run the application**:
   ```bash
   python main.py
   ```
   Or using uvicorn:
   ```bash
   uvicorn main:app --reload --host 0.0.0.0 --port 8000
   ```

6. **Access the application**:
   Open your browser and navigate to `http://localhost:8000`

## Usage Guide

### Uploading Documents
1. Navigate to the dashboard
2. Choose between text upload or PDF upload
3. Select your files and upload them
4. Wait for processing completion

### Asking Questions
1. Go to the chatbot interface
2. Type your question in the chat input
3. The system will search through uploaded documents
4. Review the generated answer and source citations

### Managing Documents
1. Use the data manager to view uploaded documents
2. Delete documents that are no longer needed
3. Browse Azure Search indexes to see processed content

### Creating Search Infrastructure
1. Use the indexer manager to create data sources
2. Configure search indexes for your document types
3. Set up automated indexers for continuous processing

## API Documentation
Once the application is running, visit `http://localhost:8000/docs` for the automatically generated API documentation (Swagger UI).

## Troubleshooting

### Common Issues
1. **Authentication Errors**: Verify all Azure service keys and endpoints in the `.env` file
2. **Index Not Found**: Create the required Azure Search indexes using the indexer manager
3. **Upload Failures**: Check Azure Blob Storage configuration and permissions
4. **Slow Responses**: Monitor Azure service quotas and performance tiers

### Logs
Check the application logs for detailed error messages and debugging information.

## Maintenance Notes

### Regular Tasks
- Monitor Azure service costs and usage
- Update Python dependencies regularly
- Clean up old documents and indexes
- Monitor storage usage in Azure Blob Storage

### Scaling Considerations
- Consider upgrading Azure service tiers for production use
- Implement caching for frequently asked questions
- Add load balancing for high-traffic scenarios

## Project Handoff Checklist

- [ ] Verify all environment variables are documented
- [ ] Ensure all Azure services are properly configured
- [ ] Test all API endpoints and frontend features
- [ ] Review and update dependencies
- [ ] Set up monitoring and logging
- [ ] Document any custom configurations or workarounds
- [ ] Provide access to Azure resources for the new team
- [ ] Schedule knowledge transfer sessions if needed

## Contact Information
For questions about this project, please refer to the commit history and documentation above.
