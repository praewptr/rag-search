# RAG PDF Question Answering API

A FastAPI-based Retrieval Augmented Generation (RAG) system that allows you to upload PDF documents and ask questions based on their content.

## Features

- **PDF Upload**: Upload PDF documents to build a knowledge base
- **Vision-Enhanced PDF Processing**: Upload PDFs that are converted to images and processed with Azure OpenAI Vision for advanced text extraction and document understanding
- **Question Answering**: Ask questions and get answers based on uploaded documents
- **Vector Search**: Uses ChromaDB and sentence transformers for semantic search
- **RESTful API**: Clean FastAPI endpoints for easy integration
- **Document Management**: List and delete uploaded documents

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Environment Configuration

Copy the example environment file and configure it:

```bash
cp .env.example .env
```

Edit `.env` file and add your API keys:

**For standard processing:**
```
OPENAI_API_KEY=your_openai_api_key_here
```

**For vision-enhanced processing (required for /upload-vision endpoint):**
```
AZURE_OPENAI_API_KEY=your_azure_openai_api_key_here
AZURE_OPENAI_ENDPOINT=https://your-resource-name.openai.azure.com/
AZURE_OPENAI_API_VERSION=2024-02-01
AZURE_OPENAI_VISION_MODEL=gpt-4-vision-preview
```

**Note:** The system works without OpenAI API keys using a fallback method, but Azure OpenAI is required for vision processing.

### 3. Run the Application

```bash
python main.py
```

Or using uvicorn directly:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at `http://localhost:8000`

## API Documentation

Once the server is running, you can access:
- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

## API Endpoints

### Upload Document (Standard)
```http
POST /upload
Content-Type: multipart/form-data
```
Upload a PDF file to the knowledge base using standard text extraction.

### Upload Document (Vision-Enhanced)
```http
POST /upload-vision
Content-Type: multipart/form-data
```
Upload a PDF file that will be converted to images and processed with Azure OpenAI Vision for enhanced text extraction and document understanding.

### Ask Question
```http
POST /ask
Content-Type: application/json

{
  "question": "What is the main topic of the document?",
  "document_ids": ["optional-document-id-filter"]
}
```

### List Documents
```http
GET /documents
```
Get a list of all uploaded documents with processing method information.

### Delete Document
```http
DELETE /documents/{document_id}
```
Delete a specific document from the knowledge base.

## Usage Example

### 1. Upload a PDF (Standard Processing)
```bash
curl -X POST "http://localhost:8000/upload" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@your_document.pdf"
```

### 2. Upload a PDF (Vision-Enhanced Processing)
```bash
curl -X POST "http://localhost:8000/upload-vision" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@your_document.pdf"
```

### 3. Ask a Question
```bash
curl -X POST "http://localhost:8000/ask" \
  -H "accept: application/json" \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the main findings?"}'
```

## How It Works

### Standard Processing Flow:
1. **Document Processing**: PDFs are processed using PyPDF2 to extract text
2. **Text Chunking**: Documents are split into manageable chunks using LangChain's text splitter
3. **Embedding Generation**: Text chunks are converted to embeddings using sentence-transformers
4. **Vector Storage**: Embeddings are stored in ChromaDB for efficient similarity search
5. **Question Answering**: When a question is asked, the system:
   - Converts the question to an embedding
   - Finds the most similar document chunks
   - Uses the context to generate an answer (with OpenAI API or fallback method)

### Vision-Enhanced Processing Flow:
1. **PDF to Images**: PDFs are converted to high-quality images using pdf2image
2. **Azure OpenAI Vision Analysis**: Each image is analyzed using Azure OpenAI's Vision API to:
   - Extract all text content with better accuracy
   - Understand document structure and visual elements
   - Identify key insights, topics, and entities
   - Analyze charts, tables, and diagrams
3. **Enhanced Text Chunking**: Vision analysis results are structured and chunked for better context
4. **Vector Storage**: Enhanced text chunks with vision insights are stored in ChromaDB
5. **Question Answering**: Same as standard flow but with richer context from vision analysis

## Technologies Used

- **FastAPI**: Web framework for the API
- **ChromaDB**: Vector database for storing embeddings
- **sentence-transformers**: For generating embeddings
- **PyPDF2**: For PDF text extraction (standard processing)
- **pdf2image**: For converting PDF pages to images (vision processing)
- **Pillow**: For image processing and manipulation
- **Azure OpenAI**: For vision-based document analysis and understanding
- **LangChain**: For text processing and splitting
- **OpenAI API** (optional): For advanced question answering

## Configuration

The system can be configured through environment variables:

- `OPENAI_API_KEY`: Your OpenAI API key (optional)
- `CHROMA_DB_PATH`: Path to store the ChromaDB database
- `HOST`: Server host (default: 0.0.0.0)
- `PORT`: Server port (default: 8000)
- `EMBEDDING_MODEL`: Sentence transformer model to use
- `CHUNK_SIZE`: Size of text chunks for processing
- `CHUNK_OVERLAP`: Overlap between text chunks

## Notes

- If no OpenAI API key is provided, the system uses a simple keyword-based fallback method for answering questions
- The vector database is persisted locally in the `chroma_db` directory
- Supports only PDF files currently
- CORS is enabled for all origins in development
