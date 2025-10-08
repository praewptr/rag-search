from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Union
from models.rag_search import QuestionRequest
from services.client import azure_openai_client
from services.qa_engine import get_response,rag_pipeline,get_llm_answer
from services.utils import remove_citation_markers
from typing import Any, Dict, List, Optional




class QuestionRequest(BaseModel):
    text: str

class AnswerResponse(BaseModel):
    text: Union[str, list[str]]



router = APIRouter()


@router.post("/test")
def ask_question(request: QuestionRequest):
    try:
        answer, citations = get_response(
            request.text,
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


@router.post("/aisearch", response_model=AnswerResponse)
async def ai_search(request: QuestionRequest):
    """
    Receives a question, processes it through the full RAG pipeline,
    and returns a structured answer. The endpoint logic is now clean and simple.
    """
    try:
        # The endpoint now makes a single, logical call to the pipeline.
        final_answer = await rag_pipeline(request.text, azure_openai_client)
        return AnswerResponse(text=final_answer)
    except Exception as e:
        print(f"An error occurred in the ai_search endpoint: {e}")
        raise HTTPException(
            status_code=500,
            detail="An internal error occurred while processing your request."
        )
