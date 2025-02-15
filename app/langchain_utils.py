import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langchain_community.chat_message_histories import ChatMessageHistory
import google.generativeai as genai

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("Google API Key not found! Please check your .env file.")

# Initialize Google Gemini AI
genai.configure(api_key=GOOGLE_API_KEY)

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
pdf_path = "data\TechFest 2025 - Event Details.pdf"

# Load PDF and initialize vector store
docs = process_pdf(pdf_path)
vectorstore = FAISS.from_documents(docs, embedding_model)
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})

fallback_responses = [
    "I don't have enough information about that. Can you ask something specific about TechFest 2025?",
    "I'm focused on TechFest 2025. Would you like to know about the schedule, workshops, or competitions?",
    "Could you rephrase your question to relate to the event?"
]

chat_history = ChatMessageHistory()

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

rag_chain = (
    {"context": retriever, 
    "question": RunnablePassthrough(), 
    "chat_history": lambda _: str(chat_history.messages)}
    | prompt
    | llm
    | StrOutputParser()
)