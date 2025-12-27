from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field

from devteam.llm.models import Message
from devteam.state.models.plan import ImplementationPlan, ImplementationPlanStep
from devteam.state.models.reviews import Review
from devteam.state.models.scratchpad import ScratchpadEntry
from devteam.state.models.task import TaskInfo

class NoImplementationPlanError(Exception):
    pass


class CodebaseInfo(BaseModel):
    # only required fields
    path: Path
    tree: str
    # should they be separated?
    languages_and_frameworks: list[str] = Field(default_factory=list)
    package_manager: Optional[str] = None
    has_tests: bool = False


class Context(BaseModel):
    task: TaskInfo

    # # this will be current step that the Devteam is working on
    # current_step: Optional[str] = None

    codebase: CodebaseInfo

    # this is a record of all the messages exchanged between the Devteam and the user (all agents)
    # this is for more forward facing purposes
    conversation_history: list[Message] = Field(default_factory=list)

    # there should be a limit to the number of entries per category
    scratchpad: list[ScratchpadEntry] = Field(default_factory=list)  # think about this

    current_implementation_plan_id: Optional[str] = None

    # should this be managed externally?
    implementation_plans: list[ImplementationPlan] = Field(default_factory=list)

    reviews: Optional[list[Review]] = None
    
    # How can I optimize this?
    read_files: Optional[dict[str, str]] = (
        None  # keeping track of files that have been read
    )
    edited_files: Optional[dict[str, str]] = (
        None  # keeping track of files that have been written
    )
    deleted_files: Optional[dict[str, str]] = (
        None  # keeping track of files that have been deleted
    )

    @property
    def current_implementation_plan(self) -> Optional[ImplementationPlan]:
        if self.current_implementation_plan_id is None:
            return None
        return next(
            (
                plan
                for plan in self.implementation_plans
                if plan.id == self.current_implementation_plan_id
            ),
            None,
        )

    def add_message_to_conversation(self, message: Message):
        self.conversation_history.append(message)

    def add_scratchpad_entry(self, entry: ScratchpadEntry):
        # validate limits and entry
        self.scratchpad.append(entry)
        
    def edit_scratchpad_entry(self, index: int, entry: ScratchpadEntry):
        # think about this
        self.scratchpad[index] = entry

    def add_step_to_implementation_plan(self, step: ImplementationPlanStep):
        if self.current_implementation_plan is None:
            raise NoImplementationPlanError("No current implementation plan exists")
            
        self.current_implementation_plan.steps.append(step)

    def create_new_implementation_plan(self):
        # think about this?
        new_implementation_plan = ImplementationPlan()
        self.implementation_plans.append(new_implementation_plan)
        self.current_implementation_plan_id = new_implementation_plan.id

