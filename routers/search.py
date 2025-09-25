from fastapi import APIRouter, HTTPException

from models.rag_search import QuestionRequest
from services.client import azure_openai_client
from services.langchain_flow import generate_answer
from services.qa_engine import get_response
from services.utils import remove_citation_markers

router = APIRouter()


@router.post("/aisearch")
def ask_question(request: QuestionRequest):
    try:
        answer, citations = get_response(
            request.question,
            azure_openai_client,
        )

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


@router.post("/test")
def langchain_search(request: QuestionRequest):
    try:
        answer = generate_answer(request.question)
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error processing request: {str(e)}"
        )
