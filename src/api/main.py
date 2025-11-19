from fastapi import FastAPI
from pydantic import BaseModel

from src.rag.qa import UrbanRegulationsQA

app = FastAPI(
    title="Urban Regulations RAG Assistant",
    description="RAG chatbot for zoning and building regulations.",
)

qa_engine = UrbanRegulationsQA(k=5)


class AskRequest(BaseModel):
    question: str


class AskResponse(BaseModel):
    answer: str
    sources: list


@app.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest):
    result = qa_engine.answer(req.question)
    return AskResponse(**result)
