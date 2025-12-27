from enum import StrEnum
from typing import Optional
from pydantic import BaseModel, Field
from datetime import datetime

class TaskState(StrEnum):
    INITIAL = "INITIAL"
    GATHERING_INITIAL_CONTEXT = "GATHERING_INITIAL_CONTEXT"
    PLANNING = "planning"
    AWAITING_PLAN_APPROVAL = "AWAITING_PLAN_APPROVAL"
    IMPLEMENTING = "IMPLEMENTING"
    TESTING = "TESTING"
    QA_REVIEW = "QA_REVIEW"
    ADDRESSING_FEEDBACK = "ADDRESSING_FEEDBACK"
    AWAITING_USER_APPROVAL = "AWAITING_USER_APPROVAL"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"

class TaskInfo(BaseModel):
    id: str
    title: str
    description: str
    initial_prompt: str
    state: TaskState = TaskState.INITIAL
    created_at: datetime = Field(default_factory=datetime.now)
    started_at: datetime = Field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    last_stopped_at: Optional[datetime] = None