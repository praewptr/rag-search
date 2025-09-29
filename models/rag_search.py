from pydantic import BaseModel


class QuestionRequest(BaseModel):
    text: str
