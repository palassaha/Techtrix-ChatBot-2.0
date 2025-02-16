import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from app.langchain_utils import vectorstore, rag_chain, fallback_responses, chat_history
import random

app = FastAPI(
    title="Techtrix-ChatBot-2.0",
    description="Techtrix-ChatBot-2.0",
    version="0.2.0"
)

# Enable Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], #change this to front end url
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str
    
class ChatResponse(BaseModel):
    respose: str
    status: str = "success"
    
    
@app.get("/")    
async def root():    
    return {"statue": "online", "message": "Techtrix 2025 Chatbot API is running"}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        docs_with_scores = vectorstore.similarity_search_with_score(request.message, k=3)
        
        if not docs_with_scores or docs_with_scores[0][1] > 1.2:
            fallback = random.choice(fallback_responses)
            logger.info(f"Fallback response triggered: {fallback}")
            return ChatResponse(response=fallback, status="success_fallback")
        
        chat_history.add_user_message(request.message)
        response = rag_chain.invoke(request.message)
        chat_history.add_ai_message(response)
        
        return ChatResponse(response=response)
        
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting TechFest 2025 Chatbot API...")
    uvicorn.run(app, host="0.0.0.0", port=8000)