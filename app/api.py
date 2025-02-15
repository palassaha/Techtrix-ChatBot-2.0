from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from app.langchain_utils import vectorstore, rag_chain, fallback_responses, chat_history
import random

# Initialize FastAPI app
app = FastAPI(
    title="TechFest 2025 Chatbot API",
    description="A chatbot API for answering questions about TechFest 2025",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change "*" to your frontend URL for security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str
    status: str = "success"

@app.get("/")
async def root():
    return {"status": "online", "message": "TechFest 2025 Chatbot API is running"}

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        docs_with_scores = vectorstore.similarity_search_with_score(request.message, k=3)
        
        if not docs_with_scores or docs_with_scores[0][1] > 1.2:
            return ChatResponse(
                response=random.choice(fallback_responses),
                status="success_fallback"
            )
        
        chat_history.add_user_message(request.message)
        response = rag_chain.invoke(request.message)
        chat_history.add_ai_message(response)
        
        return ChatResponse(response=response)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    print("Starting TechFest 2025 Chatbot API...")
    uvicorn.run(app, host="127.0.0.1", port=8000)