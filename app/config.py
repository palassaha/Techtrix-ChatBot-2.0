import os
from dotenv import load_dotenv

load_dotenv()

PDF_PATH = os.path.join("data", "TechFest2025-EventDetails.pdf")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
