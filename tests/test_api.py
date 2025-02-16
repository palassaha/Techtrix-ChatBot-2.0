import pytest
from app import app  # Import your Flask/FastAPI app
from fastapi.testclient import TestClient

client = TestClient(app)  # Use TestClient for FastAPI, or use Flask test client

def test_root():
    """Test the root endpoint"""
    response = client.get("/")
    assert response.status_code == 200
    assert "Welcome" in response.text  # Adjust based on your actual response

def test_chatbot_response():
    """Test chatbot response API"""
    payload = {"message": "Hello"}
    response = client.post("/chatbot", json=payload)  # Adjust URL if needed
    assert response.status_code == 200
    assert "reply" in response.json()  # Ensure chatbot reply exists

def test_404_error():
    """Test an invalid endpoint"""
    response = client.get("/invalid-url")
    assert response.status_code == 404
