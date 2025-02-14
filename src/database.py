from pymongo import MongoClient
from config import MONGO_URI

client = MongoClient(MONGO_URI)
db = client["event_chatbot"]
registrations_collection = db["registrations"]

def is_student_registered(email):
    return registrations_collection.find_one({"email": email})

def register_student(data):
    if is_student_registered(data.email):
        return {"message": "Student is already registered for this event."}
    
    new_registration = {
        "name": data.name,
        "email": data.email,
        "college_roll": data.college_roll,
        "college_name": data.college_name,
        "event_name": data.event_name,
        "team_members": data.team_members if data.team_members else None
    }
    
    registrations_collection.insert_one(new_registration)
    return {"message": "Registration successful!"}
