from pydantic import BaseModel, Field
from typing import Union


class QuestionRequest(BaseModel):
    text: str  = Field(..., description="The question text to search for")

class AnswerResponse(BaseModel):
    text: Union[str, list[str]] = Field(..., description="The answer text or a list of answer texts")

