from pydantic import BaseModel
from typing import List, Optional

class QueryRequest(BaseModel):
    user_id: str
    query: str

class StudentRegistration(BaseModel):
    name: str
    email: str
    college_roll: str
    college_name: str
    event_name: str
    team_members: Optional[List[dict]] = None  # For team events
