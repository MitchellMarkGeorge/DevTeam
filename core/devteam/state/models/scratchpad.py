
from typing import Optional
from pydantic import BaseModel, Field
from datetime import datetime

class ScratchpadEntry(BaseModel):
    """An entry in the Architect's scratchpad."""
    timestamp: datetime = Field(default_factory=datetime.now)
    category: str  # e.g., "finding", "question", "pattern", "dependency"
    content: str
    source_files: Optional[list[str]] = None # a list of at most 5 file paths that can be referenced