
from typing import Optional
from pydantic import BaseModel, Field
from datetime import datetime
from uuid import uuid4

class ImplementationPlanStep(BaseModel):
    """A step in an implementation plan."""
    title: str
    content: str
    code_example: Optional[str] = None
    reference_files: Optional[list[str]] = None # a list of at most 5 file paths that can be referenced
    created_at: datetime = Field(default_factory=datetime.now) # do I need this
    
class ImplementationPlan(BaseModel):
    """An implementation plan for a task."""
    
    id: str = Field(default_factory=lambda: str(uuid4()))
    version: int = 0
    steps: list[ImplementationPlanStep] = Field(default_factory=list)
    is_approved: bool = False