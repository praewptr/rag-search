import logging
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Union
from models.rag_search import QuestionRequest, AnswerResponse
from services.client import azure_openai_client
from services.qa_engine import get_response,rag_pipeline,get_llm_answer
from services.utils import remove_citation_markers
from typing import Any, Dict, List, Optional



class ErrorResponse(BaseModel):
    detail: str


# Set up logger
logger = logging.getLogger(__name__)


router = APIRouter()


@router.post(
    "/aisearch",
    response_model=AnswerResponse,
    responses={
        500: {
            "model": ErrorResponse,
            "description": "Internal server error"
        }
    }
)
async def ai_search(request: QuestionRequest):
    """
    Receives a question, processes it through the full RAG pipeline,
    and returns a structured answer. The endpoint logic is now clean and simple.

    """
    try:
        final_answer = await rag_pipeline(request.text, azure_openai_client)
        logger.info("Successfully generated answer for /aisearch")
        return AnswerResponse(text=final_answer)
    except Exception as e:
        logger.error(f"An error occurred in the ai_search endpoint: {e}")
        raise HTTPException(
            status_code=500,
            detail="An internal error occurred while processing your request."
        )
