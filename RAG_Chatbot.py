from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from sentence_transformers import SentenceTransformer
import faiss
import google.generativeai as genai
from pathlib import Path
import logging

# Configure logging to write to a file
logging.basicConfig(
    filename="rag_chatbot.log",  # Log file name
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    filemode="a",  # Append mode; change to "w" to overwrite each time
)

logger = logging.getLogger("RAG_Chatbot")

class ChatMessage(BaseModel):
    question: str

class ChatResponse(BaseModel):
    answer: str

app = FastAPI()

class RAGChatbot:
    def __init__(self, pdf_path: str, embedding_model: str = "all-MiniLM-L6-v2", embedding_dim: int = 384):
        """Initialize the chatbot with a PDF knowledge base."""
        try:
            self.embedding_model = SentenceTransformer(embedding_model)
            self.index = faiss.IndexFlatL2(embedding_dim)
            self.documents = []
            
            self._initialize_with_pdf(pdf_path)
            logger.info(f"Chatbot initialized with PDF: {pdf_path}")
        except Exception as e:
            logger.error(f"Error initializing chatbot: {str(e)}")
            raise ValueError("Failed to initialize chatbot") from e

    def _initialize_with_pdf(self, pdf_path: str):
        """Process and index the PDF content."""
        try:
            from PyPDF2 import PdfReader
            
            if not Path(pdf_path).exists():
                raise FileNotFoundError(f"PDF file not found: {pdf_path}")
            
            reader = PdfReader(pdf_path)
            processed_texts = []
            
            # Process each page and create meaningful chunks
            for page in reader.pages:
                text = page.extract_text()
                # Split into paragraphs and filter empty ones
                paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
                processed_texts.extend(paragraphs)
            
            self._add_documents(processed_texts)
            logger.info(f"Processed {len(processed_texts)} paragraphs from PDF")
            
        except Exception as e:
            logger.error(f"Error processing PDF: {str(e)}")
            raise ValueError("Failed to process PDF") from e

    def _add_documents(self, documents):
        """Add documents to the FAISS index."""
        try:
            embeddings = self.embedding_model.encode(documents)
            self.index.add(embeddings)
            self.documents.extend(documents)
            logger.info(f"Added {len(documents)} documents to index")
        except Exception as e:
            logger.error(f"Error adding documents: {str(e)}")
            raise ValueError("Failed to add documents") from e

    def _retrieve_context(self, query: str, k: int = 3):
        """Retrieve relevant context for the query."""
        try:
            query_embedding = self.embedding_model.encode([query])
            distances, indices = self.index.search(query_embedding, k)
            return [self.documents[i] for i in indices[0] if i < len(self.documents)]
        except Exception as e:
            logger.error(f"Error retrieving context: {str(e)}")
            raise ValueError("Failed to retrieve context") from e

    async def generate_response(self, question: str) -> str:
        """Generate a response based on the question and document context."""
        try:
            genai.configure(api_key="AIzaSyAj0r-73_rsVDnt1gkcGEIG6XE8vnqyjx0")
            
            # Get relevant context from the PDF
            contexts = self._retrieve_context(question)
            combined_context = " ".join(contexts)

            prompt = f"""You are a helpful assistant answering questions based on the provided document. 
            Use the following context to answer the question naturally and accurately.

            Context:
            {combined_context}

            Question: {question}

            Instructions:
            - Answer based on the context provided
            - If the context doesn't contain relevant information, say so
            - Be concise but informative
            - Maintain a conversational tone

            Response:"""

            model = genai.GenerativeModel('gemini-1.5-flash')
            response = model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.7,
                    max_output_tokens=200,  # Reasonable length for chat responses
                ),
            )
            return response.text.strip()
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            raise ValueError(f"Failed to generate response: {str(e)}")

# Initialize the chatbot with a static PDF
PDF_PATH = "./TechFest 2025-Event Details.pdf"  # Replace with your PDF path
chatbot = RAGChatbot(PDF_PATH)

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(message: ChatMessage):
    """Endpoint for chat interactions."""
    try:
        response = await chatbot.generate_response(
            question=message.question,
        )
        return ChatResponse(answer=response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)