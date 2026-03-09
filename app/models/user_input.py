from pydantic import BaseModel

class UserInput(BaseModel):
    topic: str
    difficulty: str
    subtopic: str | None = None