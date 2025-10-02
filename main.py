import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from config import setup_logging
from routers import (
    azure_index,
    create_service,
    indexer,
    rag_upload_text,
    search,
    upload_pdf,
)

# Load environment variables
env_path = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(env_path)

# Setup logging
setup_logging()

# Initialize FastAPI app
app = FastAPI(title="RAG PDF Question Answering API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


static_dir = Path(__file__).resolve().parent / "static"

app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Include routers
app.include_router(search.router, prefix="/search", tags=["AI RAG SEARCH"])
app.include_router(
    rag_upload_text.router,
    prefix="/rag-upload-text",
    tags=["Upload text from database to created index"],
)
app.include_router(
    azure_index.router, prefix="/azure-index", tags=["Azure Index Browser documents"]
)
app.include_router(
    upload_pdf.router,
    prefix="/upload-pdf",
    tags=["Upload PDF to created azure container"],
)
app.include_router(
    indexer.router,
    prefix="/indexer",
    tags=["Run selected indexer"],
)

app.include_router(
    create_service.router,
    prefix="/create-service",
    tags=["Create Indexer, Data Source, Skillset, Index"],
)




@app.get("/")
async def serve_dashboard():
    """Serve the main dashboard."""
    return FileResponse(static_dir / "dashboard.html")


@app.get("/rag-upload-text")
async def serve_data_manager():
    """Serve the data manager UI."""
    return FileResponse(static_dir / "rag_upload_text.html")


@app.get("/azure-index")
async def serve_azure_browser():
    """Serve the Azure Search index browser UI."""
    return FileResponse(static_dir / "azure_index.html")


@app.get("/pdf-upload")
async def serve_pdf_upload():
    """Serve the PDF upload UI."""
    return FileResponse(static_dir / "upload_pdf.html")


@app.get("/rag-upload-pdf")
async def serve_indexer_manager():
    """Serve the Indexer Manager UI."""
    return FileResponse(static_dir / "indexer.html")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=5099,
        access_log=False,  # Disable access logs
        # log_level="warning",  # Only show warnings and errors
    )
