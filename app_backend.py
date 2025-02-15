from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import random
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_community.chat_message_histories import ChatMessageHistory
import google.generativeai as genai

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("Google API Key not found! Please check your .env file.")

# Initialize Google Gemini AI
genai.configure(api_key=GOOGLE_API_KEY)

# Initialize FastAPI app
app = FastAPI(
    title="TechFest 2025 Chatbot API",
    description="A chatbot API for answering questions about TechFest 2025",
    version="2.0.0"
)

# Initialize LangChain's Gemini wrapper
llm = GoogleGenerativeAI(
    google_api_key=GOOGLE_API_KEY,
    model="gemini-pro",
    temperature=0.7,
    top_p=0.95,
    top_k=40,
    max_output_tokens=2048,
)

# Function to process PDFs
def process_pdf(pdf_path: str):
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"File '{pdf_path}' not found!")

    try:
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
        )
        docs = text_splitter.split_documents(documents)

        for i, doc in enumerate(docs):
            doc.metadata["chunk_id"] = i
            doc.metadata["source"] = pdf_path

        return docs

    except Exception as e:
        raise RuntimeError(f"Error processing PDF: {str(e)}")

# Initialize embeddings and vector store
embedding_model = HuggingFaceEmbeddings(
    model_name='sentence-transformers/all-mpnet-base-v2',
    model_kwargs={'device': 'cpu'}
)

# Correct file path reference
pdf_path = "TechFest-2025-Event Details.pdf"

# Global variable to track PDF loading status
pdf_loaded = False

try:
    print("Loading PDF and initializing vector store...")
    docs = process_pdf(pdf_path)
    vectorstore = FAISS.from_documents(docs, embedding_model)
    pdf_loaded = True
    print("PDF loaded successfully!")
except Exception as e:
    print(f"Error loading PDF: {e}")
    vectorstore = None

# Initialize chat history
chat_history = ChatMessageHistory()

# Enhanced prompt template
prompt = PromptTemplate.from_template("""
    You are an AI assistant for a tech event. Use the following context to answer the question.
    If you don't know the answer, use one of the following fallback response from the given list.
     "I don't have enough information about that. Can you ask something specific about TechFest 2025?",
    "I'm focused on TechFest 2025. Would you like to know about the schedule, workshops, or competitions?",
    "Could you rephrase your question to relate to the event?"                                 
    
    Context: {context}
    
    Question: {question}
    
    Previous conversation: {chat_history}
    
    Answer:
""")

# Initialize the RAG chain only if vectorstore is valid
rag_chain = None
if vectorstore:
    print("Initializing RAG chain...")
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    
    rag_chain = (
        {"context": retriever, 
         "question": RunnablePassthrough(), 
         "chat_history": lambda _: str(chat_history.messages)}
        | prompt
        | llm
        | StrOutputParser()
    )
    print("RAG chain initialized successfully!")

fallback_responses = [
    "I don't have enough information about that. Can you ask something specific about TechFest 2025?",
    "I'm focused on TechFest 2025. Would you like to know about the schedule, workshops, or competitions?",
    "Could you rephrase your question to relate to the event?"
]

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str
    status: str = "success"

@app.get("/")
async def root():
    return {
        "status": "online",
        "message": "TechFest 2025 Chatbot API is running",
        "pdf_loaded": pdf_loaded,
        "rag_chain_initialized": rag_chain is not None
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "components": {
            "pdf_loader": "up" if pdf_loaded else "down",
            "vector_store": "up" if vectorstore is not None else "down",
            "rag_chain": "up" if rag_chain is not None else "down"
        }
    }

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    if not vectorstore or not rag_chain:
        raise HTTPException(status_code=500, detail="PDF processing failed. Chatbot is unavailable.")

    try:
        docs_with_scores = vectorstore.similarity_search_with_score(request.message, k=3)
        
        if not docs_with_scores or docs_with_scores[0][1] > 1.2:
            return ChatResponse(
                response=random.choice(fallback_responses),
                status="success_fallback"
            )
        
        # Add user message to chat history
        chat_history.add_user_message(request.message)
        
        # Get response from RAG chain
        response = rag_chain.invoke(request.message)
        
        # Add assistant response to chat history
        chat_history.add_ai_message(response)
        
        return ChatResponse(response=response)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    print("Starting TechFest 2025 Chatbot API...")
    uvicorn.run(app, host="127.0.0.1", port=8000)