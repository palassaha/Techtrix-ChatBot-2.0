import uvicorn
from app.api import app  # Import FastAPI instance from api.py

if __name__ == "__main__":
    print("🚀 Starting TechFest 2025 Chatbot API...")
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=True)
