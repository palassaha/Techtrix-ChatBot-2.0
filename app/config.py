
import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    if not GOOGLE_API_KEY:
        raise ValueError("Google API Key not found! Please check your .env file.")
    
    PDF_PATH = "data/TechFest2025-EventDetails.pdf"