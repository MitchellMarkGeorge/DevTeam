from pydantic import BaseModel, Field

class FeedbackItem(BaseModel):
    type: str
    description: str
    severity: str
    file: str
    line_number_range: tuple[int, int] # should this be the actual code sample

class Review(BaseModel):
    feedback_items: list[FeedbackItem] = Field(default_factory=list)
    is_approved: bool = False