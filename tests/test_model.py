import pytest
from app.model import ChatbotModel  # Import your ML model class

def test_model_prediction():
    """Ensure chatbot model returns a response"""
    model = ChatbotModel()
    response = model.get_response("Hello")  # Adjust based on your model method
    assert isinstance(response, str)
    assert len(response) > 0  # Model should return a valid response
