from typing import List, Optional
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

from services.document_processor import DocumentProcessor
from services.vector_store import VectorStore
from services.qa_engine import QAEngine
from services.image_processor import ImageProcessor

app = FastAPI(title="RAG PDF Question Answering API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
document_processor = DocumentProcessor()
image_processor = ImageProcessor()
vector_store = VectorStore()
qa_engine = QAEngine(vector_store)

class QuestionRequest(BaseModel):
    question: str
    document_ids: Optional[List[str]] = None

class UploadResponse(BaseModel):
    message: str
    document_id: str
    filename: str

class VisionUploadResponse(BaseModel):
    message: str
    document_id: str
    filename: str
    total_pages: int
    processing_method: str
    extracted_insights: List[str]

class AnswerResponse(BaseModel):
    answer: str
    sources: List[str]
    confidence: float

@app.get("/")
async def root():
    return {"message": "RAG PDF Question Answering API is running"}

@app.post("/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """
    Upload a PDF document to the knowledge base
    """
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")
    
    try:
        # Read file content
        content = await file.read()
        
        # Process document
        document_id = await document_processor.process_document(content, file.filename)
        
        # Extract text and create embeddings
        text_chunks = await document_processor.extract_text_chunks(content)
        
        # Store in vector database
        await vector_store.add_documents(document_id, text_chunks, file.filename)
        
        return UploadResponse(
            message="Document uploaded and processed successfully",
            document_id=document_id,
            filename=file.filename
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")

@app.post("/upload-vision", response_model=VisionUploadResponse)
async def upload_document_with_vision(file: UploadFile = File(...)):
    """
    Upload a PDF document, convert to images, and process with Azure OpenAI Vision
    """
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")
    
    try:
        # Read file content
        content = await file.read()
        
        # Process document with Azure OpenAI Vision
        vision_result = await image_processor.process_pdf_with_vision(content, file.filename)
        
        # Extract text chunks from vision analysis
        text_chunks = await image_processor.extract_text_chunks_from_vision_data(
            vision_result["extracted_data"]
        )
        
        # Store in vector database with vision processing metadata
        await vector_store.add_documents(
            vision_result["document_id"], 
            text_chunks, 
            file.filename,
            metadata={
                "processing_method": "azure_openai_vision",
                "total_pages": vision_result["total_pages"]
            }
        )
        
        # Extract key insights for response
        extracted_insights = []
        for page_data in vision_result["extracted_data"]:
            insights = page_data.get("vision_analysis", {}).get("key_insights", [])
            extracted_insights.extend(insights)
        
        return VisionUploadResponse(
            message="Document uploaded and processed with Azure OpenAI Vision successfully",
            document_id=vision_result["document_id"],
            filename=file.filename,
            total_pages=vision_result["total_pages"],
            processing_method="azure_openai_vision",
            extracted_insights=extracted_insights[:10]  # Limit to top 10 insights
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing document with vision: {str(e)}")

@app.post("/ask", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest):
    """
    Ask a question based on uploaded documents
    """
    try:
        # Get answer from QA engine
        result = await qa_engine.get_answer(
            question=request.question,
            document_ids=request.document_ids
        )
        
        return AnswerResponse(
            answer=result["answer"],
            sources=result["sources"],
            confidence=result["confidence"]
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating answer: {str(e)}")

@app.get("/documents")
async def list_documents():
    """
    List all uploaded documents
    """
    try:
        documents = await vector_store.list_documents()
        return {"documents": documents}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing documents: {str(e)}")

@app.delete("/documents/{document_id}")
async def delete_document(document_id: str):
    """
    Delete a document from the knowledge base
    """
    try:
        await vector_store.delete_document(document_id)
        return {"message": f"Document {document_id} deleted successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting document: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
