<!-- Use this file to provide workspace-specific custom instructions to Copilot. For more details, visit https://code.visualstudio.com/docs/copilot/copilot-customization#_use-a-githubcopilotinstructionsmd-file -->

# RAG PDF Question Answering System

This is a FastAPI-based RAG (Retrieval Augmented Generation) system for answering questions based on uploaded PDF documents.

## Key Components

- **FastAPI**: Web framework providing REST API endpoints
- **ChromaDB**: Vector database for storing document embeddings
- **sentence-transformers**: For generating text embeddings
- **PyPDF2**: For extracting text from PDF files (standard processing)
- **pdf2image & Pillow**: For converting PDFs to images (vision processing)
- **Azure OpenAI Vision**: For advanced document analysis and understanding
- **LangChain**: For text processing and splitting

## Code Style Guidelines

- Use async/await for all API endpoints and service methods
- Include proper error handling with try/except blocks
- Add type hints for all function parameters and return values
- Use Pydantic models for API request/response validation
- Follow Python naming conventions (snake_case for functions/variables)
- Include docstrings for all public methods

## Architecture

- `main.py`: FastAPI application with API endpoints
- `services/document_processor.py`: PDF processing and text extraction (standard)
- `services/image_processor.py`: PDF to image conversion and Azure OpenAI Vision processing
- `services/vector_store.py`: ChromaDB integration for embeddings
- `services/qa_engine.py`: Question answering logic with LLM integration

## API Endpoints

- `POST /upload`: Upload PDF documents (standard text extraction)
- `POST /upload-vision`: Upload PDF documents with Azure OpenAI Vision processing
- `POST /ask`: Ask questions based on uploaded documents
- `GET /documents`: List all uploaded documents
- `DELETE /documents/{document_id}`: Delete specific documents
