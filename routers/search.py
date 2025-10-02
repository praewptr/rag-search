from fastapi import APIRouter, HTTPException

from models.rag_search import QuestionRequest
from services.client import azure_openai_client
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
        # Get context from RAG pipeline
        context = rag_pipeline(request.text)
        
        # If no context found, return empty list
        if not context:
            return {"text": []}
        
        # Get LLM answer based on context
        final_answer = get_llm_answer(request.text, context, azure_openai_client)
        
        # If LLM couldn't generate an answer, return empty list
        if not final_answer:
            return {"text": []}
            
        # Clean and return the answer
        cleaned_answer = remove_citation_markers(final_answer)
        return {"text": cleaned_answer}

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error processing request: {str(e)}"
        )
