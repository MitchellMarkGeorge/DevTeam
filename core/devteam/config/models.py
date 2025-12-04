from pydantic import BaseModel
from devteam.config import settings

from enum import StrEnum

class ApprovalMode(StrEnum):
    STRICT = "strict"
    NORMAL = "normal"
    RELAXED = "relaxed"


class FileOperationApprovalSettings(BaseModel):
    require_for_create: bool
    require_for_modify: bool
    require_for_delete: bool


class CommandOperationApprovalSettings(BaseModel):
    privileged_commands: bool
    build_commands: bool
    package_commands: bool
    destructive_commands: bool


class ApprovalSettings(BaseModel):
    require_for_architect_plan: bool = True
    file_operations: FileOperationApprovalSettings
    command_operations: CommandOperationApprovalSettings


class SandboxSettings(BaseModel):
    enabled: bool = False


class SecuritySettings(BaseModel):
    forbidden_commands: list[str]
    # think about this
    # max_file_size: int


class AuditSettings(BaseModel):
    enabled: bool
    # audit_log_path: str


class AIAgentSettings(BaseModel):
    model: str


class AgentsSettings(BaseModel):
    manager: AIAgentSettings
    architect: AIAgentSettings
    developer: AIAgentSettings
    qa: AIAgentSettings


class LLMModelSettings(BaseModel):
    enabled: bool
    api_key: str | None = None


class ModelSettings(BaseModel):
    anthropic: LLMModelSettings = LLMModelSettings(
        enabled=True if settings.anthropic_api_key else False,
        api_key=settings.anthropic_api_key,
    )
    openai: LLMModelSettings = LLMModelSettings(
        enabled=True if settings.openai_api_key else False,
        api_key=settings.openai_api_key,
    )
    gemeni: LLMModelSettings = LLMModelSettings(
        enabled=True if settings.gemeni_api_key else False,
        api_key=settings.gemeni_api_key,
    )


class ArchitectToolsSettings(BaseModel):
    web_search: bool


class ToolsSettings(BaseModel):
    architect: ArchitectToolsSettings


class UserSettings(BaseModel):
    name: str
    email: str
    git_name: str
    git_email: str