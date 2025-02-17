

import pytest
from fastapi.testclient import TestClient
from app.api import app
from app.langchain_utils import fallback_responses, chat_history

client = TestClient(app)

def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"status": "online", "message": "TechFest 2025 Chatbot API is running"}

def test_chat_endpoint_valid_query():
    response = client.post("/chat", json={"message": "What events are happening at TechFest 2025?"})
    assert response.status_code == 200
    data = response.json()
    assert "response" in data
    assert data["status"] in ["success", "success_fallback"]

def test_chat_endpoint_fallback():
    response = client.post("/chat", json={"message": "Tell me about quantum physics"})
    assert response.status_code == 200
    data = response.json()
    assert data["response"] in fallback_responses
    assert data["status"] == "success_fallback"

def test_chat_endpoint_invalid_request():
    response = client.post("/chat", json={})
    assert response.status_code == 422  # Unprocessable entity

def test_chat_history():
    chat_history.clear()
    chat_history.add_user_message("Hello")
    chat_history.add_ai_message("Hi there!")
    
    assert len(chat_history.messages) == 2
    assert chat_history.messages[0].content == "Hello"
    assert chat_history.messages[1].content == "Hi there!"
