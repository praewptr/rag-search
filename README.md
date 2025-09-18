# RAG PDF Question Answering System

This repository contains a Retrieval-Augmented Generation (RAG) system designed to answer questions based on uploaded PDF documents. The system is built using FastAPI and integrates various tools for text extraction, embedding generation, and question answering.

## Features

- **PDF Upload and Processing**: Extract text and images from PDF files using PyPDF2 and pdf2image.
- **Azure OpenAI Vision Integration**: Advanced document analysis and understanding.
- **ChromaDB**: Vector database for storing and retrieving document embeddings.
- **Question Answering**: Leverages sentence-transformers and LangChain for embedding generation and text processing.
- **REST API**: FastAPI endpoints for seamless interaction.

## Project Structure

```
.
├── main.py                     # FastAPI application entry point
├── services/                   # Core service modules
│   ├── document_processor.py   # PDF text extraction logic
│   ├── image_processor.py      # PDF to image conversion and Azure Vision processing
│   ├── vector_store.py         # ChromaDB integration for embeddings
│   ├── qa_engine.py            # Question answering logic
├── models/                     # Data models
├── static/                     # Static files (e.g., HTML templates)
├── notebooks/                  # Jupyter notebooks for experimentation
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation
```

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/praewptr/rag-search.git
   cd rag-search
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   .\venv\Scripts\activate  # On Windows
   source venv/bin/activate   # On macOS/Linux
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Start the FastAPI server:
   ```bash
   python main.py
   ```

2. Open your browser and navigate to:
   ```
   http://127.0.0.1:8000/docs
   ```
   Use the interactive Swagger UI to test the API endpoints.

## API Endpoints

- `POST /upload`: Upload PDF documents for standard text extraction.
- `POST /upload-vision`: Upload PDF documents for Azure OpenAI Vision processing.
- `POST /ask`: Ask questions based on uploaded documents.
- `GET /documents`: List all uploaded documents.
- `DELETE /documents/{document_id}`: Delete specific documents.

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Commit your changes and push to your fork.
4. Submit a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [FastAPI](https://fastapi.tiangolo.com/)
- [ChromaDB](https://www.trychroma.com/)
- [Azure OpenAI](https://azure.microsoft.com/en-us/services/openai/)
- [LangChain](https://langchain.com/)
- [PyPDF2](https://pypi.org/project/PyPDF2/)
- [pdf2image](https://pypi.org/project/pdf2image/)

---

For any questions or issues, please open an issue in the repository or contact the maintainer.
