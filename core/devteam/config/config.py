from pathlib import Path
from typing import Any

import yaml

from devteam.config import settings
from devteam.config.models import (
    ApprovalMode,
    ApprovalSettings,
    FileOperationApprovalSettings,
    CommandOperationApprovalSettings,
    SandboxSettings,
    SecuritySettings,
    AuditSettings,
    AIAgentSettings,
    ArchitectToolsSettings,
    LLMModelSettings,
    ModelSettings,
    AgentsSettings,
    ToolsSettings,
    UserSettings
)
from pydantic import BaseModel
from aiofiles import open

from devteam.utils import merge_dicts

def get_default_file_operations(approval_mode: ApprovalMode) -> FileOperationApprovalSettings:
    """Get default file operation approval settings based on approval mode."""
    is_strict_mode = approval_mode == ApprovalMode.STRICT
    is_relaxed_mode = approval_mode == ApprovalMode.RELAXED

    return FileOperationApprovalSettings(
        require_for_create=True if is_strict_mode else False,
        require_for_modify=True if is_strict_mode else False,
        require_for_delete=False if is_relaxed_mode else True,
    )


def get_default_command_operations(approval_mode: ApprovalMode) -> CommandOperationApprovalSettings:
    """Get default command operation approval settings based on approval mode."""
    is_strict_mode = approval_mode == ApprovalMode.STRICT
    is_relaxed_mode = approval_mode == ApprovalMode.RELAXED

    return CommandOperationApprovalSettings(
        privileged_commands=False if is_relaxed_mode else True,
        build_commands=True if is_strict_mode else False,
        package_commands=False if is_relaxed_mode else True,
        destructive_commands=False if is_relaxed_mode else True,
    )

def get_default_approval_settings(approval_mode: ApprovalMode) -> ApprovalSettings:
    is_relaxed_mode = approval_mode == ApprovalMode.RELAXED

    return ApprovalSettings(
        require_for_architect_plan=False if is_relaxed_mode else True,
        file_operations=get_default_file_operations(approval_mode),
        command_operations=get_default_command_operations(approval_mode),
    )


def get_default_sandbox_settings() -> SandboxSettings:
    return SandboxSettings(enabled=False)


def get_default_security_settings() -> SecuritySettings:
    forbidden_commands = ["rm -rf /", "dd if=*"]
    return SecuritySettings(forbidden_commands=forbidden_commands)


def get_default_audit_settings() -> AuditSettings:
    return AuditSettings(enabled=True)


def get_default_agents_settings() -> AgentsSettings:
    # will use model mapper/manager for this
    manager = AIAgentSettings(model="claude-haiku-3-5")
    architect = AIAgentSettings(model="claude-sonnet-4-5")
    developer = AIAgentSettings(model="claude-sonnet-4-5")
    qa = AIAgentSettings(model="claude-sonnet-4-5")

    return AgentsSettings(
        manager=manager, architect=architect, developer=developer, qa=qa
    )


def get_default_model_settings() -> ModelSettings:
    anthropic = LLMModelSettings(
        enabled=True if settings.anthropic_api_key else False,
        api_key=settings.anthropic_api_key,
    )
    openai = LLMModelSettings(
        enabled=True if settings.openai_api_key else False,
        api_key=settings.openai_api_key,
    )
    gemeni = LLMModelSettings(
        enabled=True if settings.gemeni_api_key else False,
        api_key=settings.gemeni_api_key,
    )

    return ModelSettings(anthropic=anthropic, openai=openai, gemeni=gemeni)


def get_default_tool_settings() -> ToolsSettings:
    architect = ArchitectToolsSettings(web_search=True)
    return ToolsSettings(architect=architect)
    
def preprocess_config_data(data: dict[str, Any]) -> dict[str, Any]:
    """
    Preprocess config data to merge partial configs with defaults.
    Returns a new dict without mutating the input.
    
    This function is called BEFORE model_validate(), keeping the validator clean
    and avoiding any mutation issues.
    """
    # Create a new dict to avoid any possibility of mutating input
    result = {}
    
    # Get approval_mode first as it affects default approval settings
    approval_mode = data.get('approval_mode', ApprovalMode.NORMAL)
    result['approval_mode'] = approval_mode
    
    # Handle approvals with proper defaults based on approval_mode
    if 'approvals' in data and data['approvals'] is not None:
        default_approvals = get_default_approval_settings(approval_mode)
        
        # Merge user-provided approvals with defaults
        if isinstance(data['approvals'], dict):
            approvals_data = {}
            
            # Handle file_operations
            if 'file_operations' in data['approvals']:
                file_ops_defaults = default_approvals.file_operations.model_dump()
                if isinstance(data['approvals']['file_operations'], dict):
                    approvals_data['file_operations'] = merge_dicts(
                        file_ops_defaults, 
                        data['approvals']['file_operations']
                    )
                else:
                    approvals_data['file_operations'] = data['approvals']['file_operations']
            else:
                approvals_data['file_operations'] = default_approvals.file_operations.model_dump()
            
            # Handle command_operations
            if 'command_operations' in data['approvals']:
                cmd_ops_defaults = default_approvals.command_operations.model_dump()
                if isinstance(data['approvals']['command_operations'], dict):
                    approvals_data['command_operations'] = merge_dicts(
                        cmd_ops_defaults,
                        data['approvals']['command_operations']
                    )
                else:
                    approvals_data['command_operations'] = data['approvals']['command_operations']
            else:
                approvals_data['command_operations'] = default_approvals.command_operations.model_dump()
            
            # Handle require_for_architect_plan
            if 'require_for_architect_plan' in data['approvals']:
                approvals_data['require_for_architect_plan'] = data['approvals']['require_for_architect_plan']
            else:
                approvals_data['require_for_architect_plan'] = default_approvals.require_for_architect_plan
            
            result['approvals'] = approvals_data
        else:
            result['approvals'] = data['approvals']
    else:
        result['approvals'] = get_default_approval_settings(approval_mode).model_dump()
    
    # Handle sandbox
    if 'sandbox' in data and data['sandbox'] is not None:
        result['sandbox'] = data['sandbox']
    else:
        result['sandbox'] = get_default_sandbox_settings().model_dump()
    
    # Handle security
    if 'security' in data and data['security'] is not None:
        if isinstance(data['security'], dict):
            default_security = get_default_security_settings().model_dump()
            result['security'] = merge_dicts(default_security, data['security'])
        else:
            result['security'] = data['security']
    else:
        result['security'] = get_default_security_settings().model_dump()
    
    # Handle audit
    if 'audit' in data and data['audit'] is not None:
        if isinstance(data['audit'], dict):
            default_audit = get_default_audit_settings().model_dump()
            result['audit'] = merge_dicts(default_audit, data['audit'])
        else:
            result['audit'] = data['audit']
    else:
        result['audit'] = get_default_audit_settings().model_dump()
    
    # Handle agents
    if 'agents' in data and data['agents'] is not None:
        if isinstance(data['agents'], dict):
            default_agents = get_default_agents_settings().model_dump()
            result['agents'] = merge_dicts(default_agents, data['agents'])
        else:
            result['agents'] = data['agents']
    else:
        result['agents'] = get_default_agents_settings().model_dump()
    
    # Handle models
    if 'models' in data and data['models'] is not None:
        if isinstance(data['models'], dict):
            default_models = get_default_model_settings().model_dump()
            result['models'] = merge_dicts(default_models, data['models'])
        else:
            result['models'] = data['models']
    else:
        result['models'] = get_default_model_settings().model_dump()
    
    # Handle tools
    if 'tools' in data and data['tools'] is not None:
        if isinstance(data['tools'], dict):
            default_tools = get_default_tool_settings().model_dump()
            result['tools'] = merge_dicts(default_tools, data['tools'])
        else:
            result['tools'] = data['tools']
    else:
        result['tools'] = get_default_tool_settings().model_dump()
    
    # Handle user (optional field)
    if 'user' in data:
        result['user'] = data['user']
    
    return result

class DevTeamConfig(BaseModel):
    approval_mode: ApprovalMode = ApprovalMode.NORMAL
    approvals: ApprovalSettings
    sandbox: SandboxSettings
    security: SecuritySettings
    audit: AuditSettings
    agents: AgentsSettings
    models: ModelSettings
    tools: ToolsSettings
    user: UserSettings | None = None
    

    @staticmethod
    async def from_config_file(
        config_file_path: Path = settings.config_file
    ):
        if not config_file_path.exists():
            raise FileNotFoundError(f"Error: Config file not found at {config_file_path}")

        async with open(config_file_path, "r") as f:
            try:
                yaml_data = await f.read()
            except Exception as e:
                raise ValueError(f"Error: Unable to read config file at {config_file_path}: {e}")

        return DevTeamConfig.from_yaml(yaml_data)

    @staticmethod
    def from_yaml(yaml_data: str):
        data: dict = yaml.safe_load(yaml_data)
        
        if data is None:
            data = {}
            
        processed_config = preprocess_config_data(data)
        return DevTeamConfig.model_validate(processed_config)
