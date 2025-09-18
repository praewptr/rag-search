from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from services.utils import remove_citation_markers
from services.qa_engine import get_response
from services.client import azure_openai_client
from models.rag_search import QuestionRequest

app = FastAPI(title="RAG PDF Question Answering API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.post("/aisearch")
def ask_question(request: QuestionRequest):
    try:
        answer, citations = get_response(request.question, azure_openai_client)

        cleaned_answer = remove_citation_markers(answer)

        sources = []
        for c in citations:
            title = c.get("title") or c.get("filepath") or "Unknown Source"
            if title not in sources:
                sources.append(title)

        response_payload = {
            "answer": cleaned_answer,
            "sources": sources,
            # "citations": citations
        }
        return response_payload

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error generating answer: {str(e)}"
        )
