from fastapi import APIRouter,Form
from agent.agent import RAGAgent
from pydantic import BaseModel

class Query(BaseModel):
    query: str

router = APIRouter()

@router.post("/agent")
def agent(query: Query):
    rag_agent = RAGAgent()
    ai = rag_agent.query(query.query)
    print(ai)
    return ai