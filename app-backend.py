from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import random
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAI

# Load API key from .env file
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("Google API Key not found! Please check your .env file.")

# Initialize FastAPI app
app = FastAPI()

# Define LLM
llm = GoogleGenerativeAI(
    google_api_key=GOOGLE_API_KEY, 
    model="gemini-pro",
    temperature=0.7
)

# Load and process PDF
pdf_path = "TechFest 2025 - Event Details.pdf"
loader = PyPDFLoader(pdf_path)
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
docs = text_splitter.split_documents(documents)

# Create FAISS vectorstore
# embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L12-v2')
embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-mpnet-base-v2')
vectorstore = FAISS.from_documents(docs, embedding_model)
retriever = vectorstore.as_retriever()

# Define RetrievalQA chain
chain = RetrievalQA.from_chain_type(llm=llm, chain_type='stuff', retriever=retriever)

# Fallback responses
fallback_responses = [
    "Oops! That question isn't in my study material. Try something from the PDF! 📚",
    "Hmmm... That sounds interesting, but it's out of my syllabus! 😅",
    "I wish I knew that! Maybe check with Google? 🌍",
    "Whoa! That’s beyond my knowledge base. How about something from the PDF?",
    "I'm just a humble chatbot with limited knowledge. Try asking something else! 🤖"
]

class ChatRequest(BaseModel):
    message: str

@app.post("/chat")
async def chat(request: ChatRequest):
    prompt = request.message
    docs_with_scores = vectorstore.similarity_search_with_score(prompt, k=3)

    if not docs_with_scores or docs_with_scores[0][1] > 1.2:
        response = random.choice(fallback_responses)
    else:
        response = chain.run(prompt)

    return {"response": response}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
