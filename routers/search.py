from fastapi import APIRouter, HTTPException

from models.rag_search import QuestionRequest
from services.client import azure_openai_client
from services.langchain_flow import generate_answer
from services.qa_engine import get_response,rag_pipeline,get_llm_answer
from services.utils import remove_citation_markers

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


@router.post("/aisearch")
def langchain_search(request: QuestionRequest):
    try:
        # answer = generate_answer(request.text)
        answer = rag_pipeline(request.text)
        if answer:
            final_answer = get_llm_answer(request.text, answer, azure_openai_client)
            cleaned_answer = remove_citation_markers(final_answer)
            return {"text": cleaned_answer}
        
        else:
            return {"text": []}

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error processing request: {str(e)}"
        )
