import json
from database import is_student_registered, register_student
from models import StudentRegistration

with open("events/event_details.json", "r") as f:
    event_data = json.load(f)

def get_event_type(event_name):
    for event in event_data["events"]:
        if event["event_name"].lower() == event_name.lower():
            return event["type"]
    return None

def handle_registration(data: StudentRegistration):
    event_type = get_event_type(data.event_name)

    if event_type is None:
        return {"message": "Event not found."}

    if event_type == "single":
        return register_student(data)

    return {"response": "Please provide your team member details."}
