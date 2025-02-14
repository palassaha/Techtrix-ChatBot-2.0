import google.generativeai as genai
from config import GEMINI_API_KEY

# Configure Gemini API
genai.configure(api_key=GEMINI_API_KEY)

# Use Gemini 1.5 Flash model
model = genai.GenerativeModel('gemini-1.5-flash')

def analyze_query(query):
    prompt = f"Classify the following query as either 'general' or 'registration': {query}"
    
    response = model.generate_content(prompt)
    classification = response.text.lower()
    
    if "registration" in classification:
        return "registration"
    return "general"
