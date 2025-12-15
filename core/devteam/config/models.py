from enum import StrEnum

from pydantic import BaseModel, model_validator

from devteam.config import settings
from devteam.llm.llm_models import ModelProvider, validate_model


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
    model_provider: ModelProvider
    manager: AIAgentSettings
    architect: AIAgentSettings
    developer: AIAgentSettings
    qa: AIAgentSettings

    @model_validator(mode="after")
    def validate_agents_settings(self):
        """Validates that individual model settings for each agent are valid based on the given model family"""

        if not validate_model(self.model_provider, self.manager.model):
            raise ValueError(
                f"Unsupported model for Manager agent: {self.manager.model}"
            )
        if not validate_model(self.model_provider, self.architect.model):
            raise ValueError(
                f"Unsupported model for Architect agent: {self.architect.model}"
            )
        if not validate_model(self.model_provider, self.developer.model):
            raise ValueError(
                f"Unsupported model for Developer agent: {self.developer.model}"
            )
        if not validate_model(self.model_provider, self.qa.model):
            raise ValueError(f"Unsupported model for QA agent: {self.qa.model}")

        return self


class LLMModelSettings(BaseModel):
    api_key: str | None = None


class ModelSettings(BaseModel):
    anthropic: LLMModelSettings = LLMModelSettings(
        api_key=settings.anthropic_api_key,
    )
    openai: LLMModelSettings = LLMModelSettings(
        api_key=settings.openai_api_key,
    )
    gemeni: LLMModelSettings = LLMModelSettings(
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
