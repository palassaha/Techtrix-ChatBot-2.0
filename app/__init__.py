import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import GoogleGenerativeAI
from langchain_community.chat_message_histories import ChatMessageHistory
import google.generativeai as genai

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("❌ Google API Key not found! Check your .env file.")

# Configure Google Gemini AI
genai.configure(api_key=GOOGLE_API_KEY)

# Initialize Google Gemini LLM
llm = GoogleGenerativeAI(
    google_api_key=GOOGLE_API_KEY,
    model="gemini-pro",
    temperature=0.7,
    top_p=0.95,
    top_k=40,
    max_output_tokens=2048,
)

# Initialize embeddings model
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2",
    model_kwargs={"device": "cpu"},
)

# Define global variables for vector store & retriever
vectorstore = None
retriever = None
chat_history = ChatMessageHistory()

# Function to initialize FAISS vector store
def initialize_vectorstore(docs):
    global vectorstore, retriever
    vectorstore = FAISS.from_documents(docs, embedding_model)
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    print("✅ FAISS Vector Store Initialized!")

# Fallback responses for irrelevant queries
FALLBACK_RESPONSES = [
    "I don't have enough information about that. Can you ask something specific about TechFest 2025?",
    "I'm focused on TechFest 2025. Would you like to know about the schedule, workshops, or competitions?",
    "Could you rephrase your question to relate to the event?"
]
