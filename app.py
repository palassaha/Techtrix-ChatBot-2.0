import streamlit as st
import os
import random
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings  # Updated import
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAI

# Load API key from .env file
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Ensure API key is available
if not GOOGLE_API_KEY:
    st.error("Google API Key not found! Please check your .env file.")
    st.stop()

# Define LLM with temperature control
llm = GoogleGenerativeAI(
    google_api_key=GOOGLE_API_KEY, 
    model="gemini-pro",
    temperature=0.7
)

@st.cache_resource
def load_pdf():
    pdf_path = "TechFest 2025 - Event Details.pdf"

    # Load and process the PDF
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    # Split text into chunks for better retrieval
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
    docs = text_splitter.split_documents(documents)

    # Create FAISS vectorstore for persistence
    embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L12-v2')
    vectorstore = FAISS.from_documents(docs, embedding_model)

    return vectorstore

# Load the document into FAISS vectorstore
vectorstore = load_pdf()

# Define the RetrievalQA chain
chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type='stuff',
    retriever=vectorstore.as_retriever(),
)

# List of catchy fallback responses for out-of-context questions
fallback_responses = [
    "Oops! That question isn't in my study material. Try something from the PDF! 📚",
    "Hmmm... That sounds interesting, but it's out of my syllabus! 😅",
    "I wish I knew that! Maybe check with Google? 🌍",
    "Whoa! That’s beyond my knowledge base. How about something from the PDF?",
    "I'm just a humble chatbot with limited knowledge. Try asking something else! 🤖"
]

# Streamlit UI
st.title("📚 Ask Your PDF")

# Initialize chat history
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    st.chat_message(message['role']).markdown(message['content'])

# Input from user
prompt = st.chat_input("Ask something about the document...")

if prompt:
    st.chat_message('user').markdown(prompt)
    st.session_state.messages.append({'role': 'user', 'content': prompt})
    
    # Retrieve relevant documents
    docs_with_scores = vectorstore.similarity_search_with_score(prompt, k=3)
    
    # # Debug scores
    # st.write("Retrieved documents and their similarity scores:")
    # for doc, score in docs_with_scores:
    #     st.write(f"Score: {score}, Content: {doc.page_content[:200]}...")  # Show preview

    if not docs_with_scores or docs_with_scores[0][1] > 1.2:  # Higher score = lower relevance distance
        response = random.choice(fallback_responses)
    else:
        response = chain.run(prompt)

    st.chat_message('assistant').markdown(response)
    st.session_state.messages.append({'role': 'assistant', 'content': response})

# running : streamlit run app.py