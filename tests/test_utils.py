import pytest
from app.utils import preprocess_text  # Import a utility function

def test_preprocess_text():
    """Test text preprocessing function"""
    input_text = "   Hello, WORLD!   "
    expected_output = "hello, world!"
    assert preprocess_text(input_text) == expected_output
