import os

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from config import setup_logging
from routers import azure_index, search, text_manage

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

app.mount("/static", StaticFiles(directory="static"), name="static")

# Include routers
app.include_router(search.router, prefix="/search", tags=["Search"])
app.include_router(text_manage.router, prefix="/text-manage", tags=["Text Management"])
app.include_router(
    azure_index.router, prefix="/azure-index", tags=["Azure Index Browser"]
)


@app.get("/")
async def serve_dashboard():
    """Serve the main dashboard."""
    return FileResponse("static/dashboard.html")


@app.get("/text-manager")
async def serve_data_manager():
    """Serve the data manager UI."""
    return FileResponse("static/data_manager.html")


@app.get("/azure-browser")
async def serve_azure_browser():
    """Serve the Azure Search index browser UI."""
    return FileResponse("static/azure_index.html")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=5099)
