from fastapi import FastAPI
from pydantic import BaseModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage
from typing import Dict, Optional
import json
import os

# Set up your Gemini API Key
os.environ["GOOGLE_API_KEY"] = "AIzaSyABvCxsNHwoePoveLn-WMe2PFIIh8QUZ-k"

app = FastAPI()

# Initialize Gemini model
model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)

# Define required data points
REQUIRED_DATA_POINTS = ["name", "email", "phone number"]

class ChatRequest(BaseModel):
    user_id: str
    message: str

class ChatMemory:
    def __init__(self):
        self.conversations: Dict[str, list] = {}
        self.collected_data: Dict[str, Dict[str, Optional[str]]] = {}
    
    def add_message(self, user_id: str, message: str, is_user: bool = True):
        if user_id not in self.conversations:
            self.conversations[user_id] = []
        
        message_obj = HumanMessage(content=message) if is_user else AIMessage(content=message)
        self.conversations[user_id].append(message_obj)
    
    def get_history(self, user_id: str) -> list:
        return self.conversations.get(user_id, [])
    
    def update_collected_data(self, user_id: str, new_data: Dict[str, Optional[str]]):
        if user_id not in self.collected_data:
            self.collected_data[user_id] = {data: None for data in REQUIRED_DATA_POINTS}
        
        # Update only non-None values from new_data
        for key, value in new_data.items():
            if value is not None:
                self.collected_data[user_id][key] = value
    
    def get_collected_data(self, user_id: str) -> Dict[str, Optional[str]]:
        return self.collected_data.get(user_id, {data: None for data in REQUIRED_DATA_POINTS})

# Initialize chat memory
memory = ChatMemory()

async def extract_data_with_ai(message: str) -> Dict[str, Optional[str]]:
    """Use Gemini to extract structured data from a message."""
    prompt = f"""
    Extract ONLY the following details from this message if they exist:
    - Name (should be a proper name, not just any word)
    - Email (should be in valid email format)
    - Phone Number (should be a sequence of numbers, possibly with formatting)

    Respond strictly in this JSON format:
    {{
        "name": "extracted name or null if not found",
        "email": "extracted email or null if not found",
        "phone number": "extracted phone number or null if not found"
    }}

    Message: {message}
    """

    try:
        response = model.invoke(prompt)
        response_text = response.content.strip()
        
        # Extract JSON part if response contains additional text
        start_idx = response_text.find('{')
        end_idx = response_text.rfind('}') + 1
        if start_idx != -1 and end_idx != 0:
            json_str = response_text[start_idx:end_idx]
            extracted_data = json.loads(json_str)
        else:
            raise ValueError("No valid JSON found in response")
        
        # Validate and clean extracted data
        return {
            "name": extracted_data.get("name") if extracted_data.get("name") != "" else None,
            "email": extracted_data.get("email") if extracted_data.get("email") != "" else None,
            "phone number": extracted_data.get("phone number") if extracted_data.get("phone number") != "" else None
        }
    
    except Exception as e:
        print(f"Error in data extraction: {e}")
        return {data: None for data in REQUIRED_DATA_POINTS}

def generate_response(current_data: Dict[str, Optional[str]], missing_data: list) -> str:
    """Generate a conversational response based on collected and missing data."""
    name = current_data.get("name")
    email = current_data.get("email")
    phone = current_data.get("phone number")
    
    # First message if we have a name but missing other details
    if name and len(missing_data) > 0:
        missing_items = " and ".join(missing_data)
        return f"Hello {name}! Could you please share your {missing_items}?"
    
    # If we have all the data
    if not missing_data:
        return f"Thank you {name}! I have confirmed your details:\nEmail: {email}\nPhone: {phone}"
    
    # If we just got email and phone but still missing some data
    if email or phone:
        collected = []
        if email:
            collected.append(f"email ({email})")
        if phone:
            collected.append(f"phone number ({phone})")
        
        if collected and missing_data:
            missing_items = " and ".join(missing_data)
            return f"I've got your {' and '.join(collected)}. Could you please provide your {missing_items}?"
    
    # Default case for missing data
    missing_items = " and ".join(missing_data)
    return f"Could you please provide your {missing_items}?"

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    user_message = request.message
    user_id = request.user_id

    # Store user message in memory
    memory.add_message(user_id, user_message, is_user=True)

    # Extract available data using AI
    extracted_data = await extract_data_with_ai(user_message)
    
    # Update stored data with new information
    memory.update_collected_data(user_id, extracted_data)
    
    # Get all collected data so far
    current_data = memory.get_collected_data(user_id)
    
    # Check what data is still missing
    missing_data = [data for data in REQUIRED_DATA_POINTS if not current_data.get(data)]

    # Generate appropriate response
    response = generate_response(current_data, missing_data)
    memory.add_message(user_id, response, is_user=False)

    return {
        "response": response,
        "collected_data": current_data,
        "missing_data": missing_data
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)