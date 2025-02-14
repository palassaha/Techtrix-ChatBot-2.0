from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.memory import BaseMemory
from typing import Dict, List, Any

class ChatMemory(BaseMemory):
    def __init__(self):
        self.messages: Dict[str, List[Any]] = {}
        
    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
        user_id = inputs["user_id"]
        message = outputs["message"]
        
        if user_id not in self.messages:
            self.messages[user_id] = []
            
        # Store as a HumanMessage or AIMessage depending on the context
        message_obj = HumanMessage(content=message) if "user_message" in inputs else AIMessage(content=message)
        self.messages[user_id].append(message_obj)
    
    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, List[Any]]:
        user_id = inputs["user_id"]
        return {
            "chat_history": self.messages.get(user_id, [])
        }
        
    def clear(self, user_id: str = None) -> None:
        if user_id:
            self.messages.pop(user_id, None)
        else:
            self.messages.clear()

# Create memory instance
memory = ChatMemory()

def update_chat_memory(user_id: str, message: str, is_user: bool = True) -> None:
    """
    Update the chat memory with a new message
    
    Args:
        user_id (str): The unique identifier for the user
        message (str): The message content to store
        is_user (bool): Whether the message is from the user (True) or AI (False)
    """
    context = {
        "user_id": user_id,
        "user_message" if is_user else "ai_message": message
    }
    memory.save_context(context, {"message": message})

def get_chat_memory(user_id: str) -> List[Any]:
    """
    Retrieve the chat history for a specific user
    
    Args:
        user_id (str): The unique identifier for the user
        
    Returns:
        List[Any]: List of message objects representing the chat history
    """
    return memory.load_memory_variables({"user_id": user_id})["chat_history"]

# Example usage:
if __name__ == "__main__":
    # Store some messages
    update_chat_memory("user123", "Hello!", is_user=True)
    update_chat_memory("user123", "Hi there!", is_user=False)
    
    # Retrieve chat history
    history = get_chat_memory("user123")
    for message in history:
        print(f"{'User' if isinstance(message, HumanMessage) else 'AI'}: {message.content}")